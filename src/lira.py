# the strongest known attack for working out who was in the training data.

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch

from train_baseline import (
    run_dir,
    TARGET_COLUMN,
    load_data,
    positive_class_weight,
    predict,
    scale_using_public_bounds,
    train_arm,
)

ROOT = Path(__file__).resolve().parents[1]

N_SHADOW = 64
SHADOW_VAL_FRACTION = 0.2
EPS = 1e-12


def logit_scale(probabilities, labels):
    # stretch out the model's confidence in the correct answer.
    # use more decimal places first. with fewer, a confidence of 0.999... rounds to exactly 1 and the maths breaks.
    p_true = np.where(labels == 1, probabilities, 1.0 - probabilities).astype(np.float64)
    p_true = np.clip(p_true, EPS, 1.0 - EPS)
    return np.log(p_true) - np.log1p(-p_true)


def train_shadows(features, labels, pos_weight, arm, n_shadow, fit_size, rng, seed_offset):
    # train many copies of the model, each on a different random half of the patients.
    n_records = len(labels)
    phi = np.zeros((n_shadow, n_records), dtype=np.float64)
    was_in = np.zeros((n_shadow, n_records), dtype=bool)

    for index in range(n_shadow):
        chosen = rng.choice(n_records, size=fit_size, replace=False)
        mask = np.zeros(n_records, dtype=bool)
        mask[chosen] = True

        n_val = int(SHADOW_VAL_FRACTION * fit_size)
        val_idx, fit_idx = chosen[:n_val], chosen[n_val:]

        model, _ = train_arm(
            arm,
            features[fit_idx],
            labels[fit_idx],
            features[val_idx],
            labels[val_idx],
            pos_weight,
            seed=seed_offset + index,
        )

        probabilities, _ = predict(model, features, labels)
        phi[index] = logit_scale(probabilities, labels)
        was_in[index] = mask

        if (index + 1) % 8 == 0:
            print(f"    {arm}: {index + 1}/{n_shadow} shadow models trained")

    return phi, was_in


def lira_scores(phi_shadow, was_in, phi_target, min_observations=4):
    # for each patient, compare how the model acts when it has seen them and when it has not.
    n_records = phi_shadow.shape[1]
    scores = np.zeros(n_records)

    global_std = float(phi_shadow.std())
    variance_floor = (0.1 * global_std) ** 2

    for record in range(n_records):
        in_values = phi_shadow[was_in[:, record], record]
        out_values = phi_shadow[~was_in[:, record], record]

        if len(in_values) < min_observations or len(out_values) < min_observations:
            scores[record] = 0.0
            continue

        mu_in, mu_out = in_values.mean(), out_values.mean()
        # keep the spread above a minimum. with few models it can look far too small and fake a confident answer.
        var_in = max(float(np.nan_to_num(in_values.var(ddof=1))), variance_floor)
        var_out = max(float(np.nan_to_num(out_values.var(ddof=1))), variance_floor)

        target = phi_target[record]
        log_in = -0.5 * (np.log(2 * np.pi * var_in) + (target - mu_in) ** 2 / var_in)
        log_out = -0.5 * (np.log(2 * np.pi * var_out) + (target - mu_out) ** 2 / var_out)
        scores[record] = log_in - log_out

    return scores


def main():
    parser = argparse.ArgumentParser(description="the strongest known attack for working out who was in the training data.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shadows", type=int, default=N_SHADOW)
    parser.add_argument("--arms", nargs="+", default=["overfit", "tuned"])
    args = parser.parse_args()

    # the shadow draw follows the run seed, so each seed gets its own shadows.
    rng = np.random.default_rng(1000 + args.seed)
    out = run_dir(args.seed)

    data = load_data(args.seed)
    features = scale_using_public_bounds(data)
    labels = data[TARGET_COLUMN].to_numpy(dtype=np.float32)

    is_fit = (data["split"] == "fit").to_numpy()
    pos_weight = positive_class_weight(labels[is_fit])
    fit_size = int(is_fit.sum())

    baseline_predictions = pd.read_csv(out / "baseline_predictions.csv")

    frames = []
    for arm in args.arms:
        print(f"seed {args.seed}: training {args.shadows} shadow models for '{arm}'")
        phi_shadow, was_in = train_shadows(
            features, labels, pos_weight, arm, args.shadows, fit_size, rng,
            seed_offset=10_000 * (args.seed + 1),
        )

        arm_predictions = baseline_predictions[baseline_predictions["arm"] == arm]
        lookup = arm_predictions.set_index("record_id")
        aligned = data["record_id"].map(lookup["predicted_probability"])

        phi_target = logit_scale(aligned.to_numpy(dtype=float), labels)

        # only score the rows we are attacking. the "val" rows have no prediction to work from.
        keep = data["record_id"].isin(arm_predictions["record_id"]).to_numpy()
        scores = lira_scores(phi_shadow[:, keep], was_in[:, keep], phi_target[keep])

        undecidable = int((~np.isfinite(scores)).sum())
        if undecidable:
            print(f"  {undecidable} records had too few shadow observations; scored 0")
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        frames.append(
            pd.DataFrame(
                {
                    "record_id": data["record_id"].to_numpy()[keep],
                    "target": f"Non-private ({arm})",
                    "lira_score": scores,
                }
            )
        )

        evaluated = frames[-1].merge(
            arm_predictions[["record_id", "is_member"]], on="record_id"
        )
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(evaluated["is_member"], evaluated["lira_score"])
        print(f"  LiRA AUC against '{arm}': {auc:.4f}")

    pd.concat(frames, ignore_index=True).to_csv(out / "lira_scores.csv", index=False)
    print(f"saved {out / 'lira_scores.csv'}")


if __name__ == "__main__":
    main()
