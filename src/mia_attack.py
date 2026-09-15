# attacks that try to work out which patients were used to train each model.

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"

SEED = 2026
N_BOOTSTRAP = 400
FPR_TARGETS = (0.001, 0.01, 0.1)
# a shared x grid so the roc curves from different seeds can be averaged.
FPR_GRID = np.logspace(-3.5, 0.0, 120)
TARGET_ORDER = [
    "Non-private (overfit)",
    "Non-private (tuned)",
    "DP eps=8",
    "DP eps=3",
    "DP eps=1",
    "DP eps=0.5",
]
DELTA = 1e-5
EPS = 1e-12

C_ATTACK = "#eb6834"
C_UTILITY = "#2a78d6"
C_THIRD = "#1baf7a"
C_FOURTH = "#4a3aa7"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"

# some attackers also know whether the patient really was readmitted. those ones are stronger.
THREAT_MODELS = {
    "confidence": "label_agnostic",
    "neg_entropy": "label_agnostic",
    "yeom_neg_loss": "label_aware",
    "modified_entropy": "label_aware",
    "class_calibrated_loss": "label_aware",
    "learned_ensemble": "label_aware",
    "lira": "label_aware",
}


def binary_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def attack_scores(frame, calibration_reference):
    # score how strongly the attacker believes each patient was used in training.
    p = frame["predicted_probability"].to_numpy(dtype=float)
    y = frame["true_label"].to_numpy(dtype=int)
    loss = frame["prediction_loss"].to_numpy(dtype=float)

    p_true = np.where(y == 1, p, 1.0 - p)
    p_true = np.clip(p_true, EPS, 1 - EPS)

    scores = {
        "confidence": np.maximum(p, 1.0 - p),
        "neg_entropy": -binary_entropy(p),
        "yeom_neg_loss": -loss,
        # with only two outcomes this orders patients exactly like the loss does. the scores should match.
        "modified_entropy": 2.0 * (1.0 - p_true) * np.log(p_true),
    }

    ref_loss = calibration_reference["prediction_loss"].to_numpy(dtype=float)
    ref_y = calibration_reference["true_label"].to_numpy(dtype=int)
    class_median = {c: np.median(ref_loss[ref_y == c]) for c in (0, 1)}
    baseline_loss = np.array([class_median[c] for c in y])
    # some patients are just easy to predict. take that away, so easy is not mistaken for memorised.
    scores["class_calibrated_loss"] = -(loss - baseline_loss)

    return scores


def fit_learned_attacker(calibration, reference_for_calibration):
    # an attacker that learns from examples where it already knows the answer.
    feats = attack_feature_matrix(calibration, reference_for_calibration)
    model = GradientBoostingClassifier(random_state=SEED, max_depth=2)
    model.fit(feats, calibration["is_member"].to_numpy(dtype=int))
    return model


def attack_feature_matrix(frame, calibration_reference):
    s = attack_scores(frame, calibration_reference)
    return np.column_stack(
        [
            s["confidence"],
            s["neg_entropy"],
            s["yeom_neg_loss"],
            s["class_calibrated_loss"],
            frame["true_label"].to_numpy(dtype=float),
        ]
    )


def tpr_at_fpr(y_true, score, target_fpr):
    # how many real members the attack catches while staying under a false-alarm limit.
    fpr, tpr, _ = roc_curve(y_true, score)
    usable = fpr <= target_fpr
    return float(tpr[usable].max()) if usable.any() else 0.0


def bootstrap_ci(y_true, score, statistic, rng, n=N_BOOTSTRAP):
    # re-draw the data many times to get an error bar.
    member_idx = np.flatnonzero(y_true == 1)
    other_idx = np.flatnonzero(y_true == 0)
    draws = []
    for _ in range(n):
        pick = np.concatenate(
            [
                rng.choice(member_idx, member_idx.size, replace=True),
                rng.choice(other_idx, other_idx.size, replace=True),
            ]
        )
        try:
            draws.append(statistic(y_true[pick], score[pick]))
        except ValueError:
            continue
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def balanced_accuracy_at(y_true, score, threshold):
    guess = score >= threshold
    tpr = guess[y_true == 1].mean()
    tnr = (~guess[y_true == 0]).mean()
    return float((tpr + tnr) / 2.0)


def best_threshold(y_true, score):
    # pick the cut-off that separates the two groups best.
    fpr, tpr, thresholds = roc_curve(y_true, score)
    return float(thresholds[np.argmax(tpr - fpr)])


def clopper_pearson(successes, trials, alpha=0.05):
    # an honest error range for a count out of a total.
    lo = beta.ppf(alpha / 2, successes, trials - successes + 1) if successes > 0 else 0.0
    hi = (
        beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
        if successes < trials
        else 1.0
    )
    return float(lo), float(hi)


def audit_epsilon(y_true, score, delta=DELTA):
    # turn how well the attack did into a measured privacy loss.
    n_member = int((y_true == 1).sum())
    n_other = int((y_true == 0).sum())
    best = 0.0

    for threshold in np.unique(score):
        guess = score >= threshold
        fp = int(guess[y_true == 0].sum())
        fn = int((~guess[y_true == 1]).sum())

        # always take the least flattering estimate, so we never claim more certainty than we have.
        _, fpr_hi = clopper_pearson(fp, n_other)
        _, fnr_hi = clopper_pearson(fn, n_member)

        if fnr_hi > 0 and (1 - delta - fpr_hi) > 0:
            best = max(best, float(np.log((1 - delta - fpr_hi) / fnr_hi)))

        _, tnr_hi = clopper_pearson(n_other - fp, n_other)
        _, tpr_hi = clopper_pearson(n_member - fn, n_member)
        if tpr_hi > 0 and (1 - delta - tnr_hi) > 0:
            best = max(best, float(np.log((1 - delta - tnr_hi) / tpr_hi)))

    return max(best, 0.0)


def evaluate_target(frame, target_name, rng, lira_lookup=None):
    # run every attack against one model and score how well each one did.
    fold_a, fold_b = train_test_split(
        frame,
        test_size=0.5,
        stratify=frame[["is_member", "true_label"]],
        random_state=SEED,
    )

    scored_frames, scored = [], {}
    # split the data in two. each half is attacked by an attacker that never saw it.
    for calibration, evaluation in ((fold_a, fold_b), (fold_b, fold_a)):
        reference = calibration[calibration["is_member"] == 0]

        fold_scores = attack_scores(evaluation, reference)
        learned = fit_learned_attacker(calibration, reference)
        fold_scores["learned_ensemble"] = learned.predict_proba(
            attack_feature_matrix(evaluation, reference)
        )[:, 1]

        scored_frames.append(evaluation)
        for name, values in fold_scores.items():
            scored.setdefault(name, []).append(values)

    evaluation = pd.concat(scored_frames, ignore_index=True)
    scores = {name: np.concatenate(parts) for name, parts in scored.items()}

    # lira is slow and runs in its own file. its scores come in here so every attack is judged the same way.
    if lira_lookup is not None:
        scores["lira"] = evaluation["record_id"].map(lira_lookup).to_numpy(dtype=float)

    y = evaluation["is_member"].to_numpy(dtype=int)

    rows, curves = [], {}
    for name, score in scores.items():
        auc = roc_auc_score(y, score)
        auc_lo, auc_hi = bootstrap_ci(y, score, roc_auc_score, rng)

        threshold = best_threshold(y, score)

        row = {
            "target": target_name,
            "attack": name,
            "threat_model": THREAT_MODELS[name],
            "auc": float(auc),
            "auc_lo": auc_lo,
            "auc_hi": auc_hi,
            "balanced_accuracy_optimistic": balanced_accuracy_at(y, score, threshold),
            "empirical_epsilon": audit_epsilon(y, score),
            "n_eval": int(len(y)),
            "fpr_resolution_floor": 1.0 / float((y == 0).sum()),
        }
        for target_fpr in FPR_TARGETS:
            row[f"tpr_at_fpr_{target_fpr:g}"] = tpr_at_fpr(y, score, target_fpr)
        lo, hi = bootstrap_ci(y, score, lambda a, b: tpr_at_fpr(a, b, 0.01), rng)
        row["tpr_at_fpr_0.01_lo"] = lo
        row["tpr_at_fpr_0.01_hi"] = hi
        rows.append(row)

        fpr, tpr, _ = roc_curve(y, score)
        curves[name] = (fpr, tpr)

    return rows, curves, evaluation, scores


def subgroup_vulnerability(evaluation, score, patients, target_name, attack_name, rng):
    # which patients are easiest to identify. an average hides the people at risk.
    overlap = [c for c in patients.columns if c != "record_id" and c in evaluation.columns]
    joined = evaluation.drop(columns=overlap).merge(
        patients, on="record_id", how="left", validate="one_to_one"
    )
    joined = joined.assign(score=score)

    non_members = joined[joined["is_member"] == 0]
    members = joined[joined["is_member"] == 1]

    groups = {
        "age": pd.cut(members["age"], [17, 45, 60, 75, 90],
                      labels=["18-45", "46-60", "61-75", "76-90"]),
        "comorbidities": pd.cut(members["comorbidity_count"], [-1, 0, 2, 4, 8],
                                labels=["0", "1-2", "3-4", "5+"]),
        "sex": members["sex"],
        "hospital": members["entity_id"],
        "readmitted": members["true_label"].map({0: "no", 1: "yes"}),
    }

    rows = []
    for dimension, series in groups.items():
        for level, idx in members.groupby(series, observed=True).groups.items():
            subset = members.loc[idx]
            if len(subset) < 30:
                continue
            y = np.concatenate([np.ones(len(subset), int), np.zeros(len(non_members), int)])
            values = np.concatenate([subset["score"].to_numpy(), non_members["score"].to_numpy()])
            auc = float(roc_auc_score(y, values))
            lo, hi = bootstrap_ci(y, values, roc_auc_score, rng, n=400)
            rows.append({
                "target": target_name,
                "attack": attack_name,
                "dimension": dimension,
                "group": str(level),
                "n_members": int(len(subset)),
                "attack_auc": auc,
                "ci_lo": lo,
                "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=13, color=INK, pad=16, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_MUTED)
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c4")
    ax.tick_params(length=0, colors=INK_MUTED, labelsize=9)


def plot_loglog_roc(roc, path):
    # one thin line per seed plus the average. the spread is the run to run noise.
    targets = [t for t in TARGET_ORDER if t in set(roc["target"])]
    panels = [t for t in targets if t in (
        "Non-private (overfit)", "Non-private (tuned)", "DP eps=3")]

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5),
                             facecolor=SURFACE, sharey=True)
    colors = [C_ATTACK, C_THIRD, "#eda100", "#e87ba4", C_FOURTH, C_UTILITY]
    redundant = ("modified_entropy", "neg_entropy")

    for ax, target in zip(axes, panels):
        panel = roc[roc["target"] == target]
        attacks = [a for a in panel["attack"].unique() if a not in redundant]

        for attack, color in zip(attacks, colors):
            curves = panel[panel["attack"] == attack]
            for _, seed_curve in curves.groupby("seed"):
                ax.plot(seed_curve["fpr"], np.maximum(seed_curve["tpr"], 1e-4),
                        lw=1, color=color, alpha=0.35, zorder=2)

            mean = curves.groupby("fpr", as_index=False)["tpr"].mean()
            ax.plot(mean["fpr"], np.maximum(mean["tpr"], 1e-4),
                    lw=2.6 if attack == "lira" else 2.0, color=color,
                    label=attack, zorder=4 if attack == "lira" else 3)

        ax.plot([1e-4, 1], [1e-4, 1], ls=":", lw=1, color=INK_MUTED, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(3e-4, 1)
        ax.set_ylim(3e-4, 1)
        style(ax, target, "False positive rate", "True positive rate")

    for ax in axes[1:]:
        ax.set_ylabel("")
    axes[0].text(3.6e-4, 4.4e-4, "random guess", fontsize=8, color=INK_MUTED)

    richest = max(axes, key=lambda a: len(a.get_legend_handles_labels()[0]))
    handles, labels = richest.get_legend_handles_labels()
    axes[-1].legend(handles, labels, frameon=False, fontsize=9,
                    loc="lower right", labelcolor=INK_MUTED)

    seeds = roc["seed"].nunique()
    fig.suptitle(
        f"Attack ROC on log-log axes, {seeds} independent runs per curve",
        fontsize=12, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_tradeoff(tradeoff, path, ceiling, seeds):
    # error bars are the spread over seeds, not a bootstrap of one run.
    fig, ax = plt.subplots(figsize=(9.5, 5.5), facecolor=SURFACE)
    x = np.arange(len(tradeoff))

    ax.errorbar(x, tradeoff["attack_auc_mean"], yerr=tradeoff["attack_auc_std"],
                color=C_ATTACK, lw=2, marker="o", ms=8, mec=SURFACE, mew=2,
                capsize=4, elinewidth=1.4,
                label=f"Privacy risk: strongest attack AUC (mean \u00b1 SD, {seeds} runs)")
    ax.errorbar(x, tradeoff["target_test_auc_mean"], yerr=tradeoff["target_test_auc_std"],
                color=C_UTILITY, lw=2, marker="s", ms=8, mec=SURFACE, mew=2,
                capsize=4, elinewidth=1.4,
                label=f"Utility: target model test AUC (mean \u00b1 SD, {seeds} runs)")

    ax.axhline(0.5, color=INK_MUTED, lw=1, ls=":", zorder=0)
    ax.text(len(x) - 0.55, 0.508, "attack no better than guessing",
            ha="right", fontsize=8, color=INK_MUTED)

    ax.axhline(ceiling, color=C_THIRD, lw=1.5, ls="--", zorder=0)
    ax.text(-0.35, ceiling + 0.008, f"Bayes-optimal ceiling ({ceiling:.3f})",
            fontsize=9, color=C_THIRD)

    for i, row in tradeoff.reset_index(drop=True).iterrows():
        ax.annotate(f"{row['attack_auc_mean']:.2f}", (i, row["attack_auc_mean"]),
                    textcoords="offset points", xytext=(0, 14), ha="center",
                    fontsize=9, color=INK)
        ax.annotate(f"{row['target_test_auc_mean']:.2f}", (i, row["target_test_auc_mean"]),
                    textcoords="offset points", xytext=(0, -20), ha="center",
                    fontsize=9, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(tradeoff["label"], fontsize=10, color=INK)
    ax.set_xlim(-0.4, len(x) - 0.6)
    ax.set_ylim(0.42, 0.86)
    style(ax, "Privacy / utility across the privacy budget",
          "\u2190 weaker privacy guarantee        stronger privacy guarantee \u2192", "AUC")
    ax.legend(frameon=False, fontsize=10, loc="upper right", labelcolor=INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_epsilon_audit(audit, path, unprotected_epsilon, seeds):
    # every protected model sits at zero, so label the points and add a reference.
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)

    ax.plot(audit["theoretical_epsilon"], audit["theoretical_epsilon"], ls=":", lw=1.5,
            color=INK_MUTED, zorder=1, label="if the accounting were tight")

    if unprotected_epsilon:
        ax.axhline(unprotected_epsilon, color=C_FOURTH, lw=1.5, ls="--", zorder=1)
        ax.text(audit["theoretical_epsilon"].min(), unprotected_epsilon + 0.18,
                f"unprotected model is caught at eps >= {unprotected_epsilon:.2f}",
                fontsize=9, color=C_FOURTH)

    ax.errorbar(audit["theoretical_epsilon"], audit["empirical_epsilon_mean"],
                yerr=audit["empirical_epsilon_std"], color=C_ATTACK, lw=2.5,
                marker="o", ms=10, mec=SURFACE, mew=2, capsize=4, zorder=3,
                label=f"strongest attack, measured (mean \u00b1 SD, {seeds} runs)")

    for _, row in audit.iterrows():
        ax.annotate(f"{row['empirical_epsilon_mean']:.2f}",
                    (row["theoretical_epsilon"], row["empirical_epsilon_mean"]),
                    textcoords="offset points", xytext=(0, 14), ha="center",
                    fontsize=10, color=C_ATTACK, fontweight="bold")

    ax.set_xscale("log")
    ax.set_ylim(-0.4, max(float(audit["theoretical_epsilon"].max()),
                          unprotected_epsilon or 0) * 1.15)
    style(ax, "Privacy audit: no attack exceeds the accounted budget",
          "theoretical epsilon (Opacus PRV accountant)", "epsilon lower bound from attack")
    ax.legend(frameon=False, fontsize=10, loc="upper left", labelcolor=INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_subgroups(subgroups, path, seeds):
    # chart which kinds of patient are easiest to identify.
    data = subgroups[subgroups["dimension"].isin(
        ["age", "comorbidities", "readmitted"])].copy()
    fig, ax = plt.subplots(figsize=(7.2, 7.6), facecolor=SURFACE)
    labels = data["dimension"] + " = " + data["group"]
    y = np.arange(len(data))
    palette = {"age": C_ATTACK, "comorbidities": C_UTILITY, "readmitted": C_FOURTH}
    colors = [palette[d] for d in data["dimension"]]

    ax.barh(y, data["attack_auc_mean"] - 0.5, left=0.5, color=colors, height=0.62)
    ax.errorbar(data["attack_auc_mean"], y, xerr=data["attack_auc_std"],
                fmt="none", ecolor=INK_MUTED, elinewidth=1, capsize=3)
    ax.axvline(0.5, color=INK_MUTED, lw=1, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0.45, max(0.80, float((data["attack_auc_mean"] + data["attack_auc_std"]).max()) + 0.03))
    style(ax, f"Which patients are identifiable (mean of {seeds} runs)",
          "attack AUC: that group's members vs all non-members", "")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def load_patients():
    frames = [
        pd.read_csv(DATA_DIR / name)
        for name in ("hospital_a.csv", "hospital_b.csv")
    ]
    patients = pd.concat(frames, ignore_index=True)
    return patients[["record_id", "age", "sex", "comorbidity_count", "entity_id"]]


def bayes_ceiling(seed):
    # the best score any model could ever reach on this data.
    oracle = pd.read_csv(DATA_DIR / "oracle.csv")
    splits = pd.read_csv(DATA_DIR / f"splits_seed{seed}.csv")
    labels = pd.concat(
        [pd.read_csv(DATA_DIR / n)[["record_id", "readmitted_30d"]]
         for n in ("hospital_a.csv", "hospital_b.csv")],
        ignore_index=True,
    )
    joined = labels.merge(oracle, on="record_id").merge(splits, on="record_id")
    test = joined[joined["split"] == "test"]
    return float(roc_auc_score(test["readmitted_30d"], test["true_readmission_probability"]))


def roc_on_grid(curves, target, seed):
    # save each curve on a shared x grid so seeds can be averaged later.
    rows = []
    for attack, (fpr, tpr) in curves.items():
        rows.append(pd.DataFrame({
            "seed": seed,
            "target": target,
            "attack": attack,
            "fpr": FPR_GRID,
            "tpr": np.interp(FPR_GRID, fpr, tpr),
        }))
    return pd.concat(rows, ignore_index=True)


def run_one_seed(seed):
    # run every attack against every model for a single seed.
    rng = np.random.default_rng(SEED + seed)
    out = RUNS_DIR / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(out / "baseline_predictions.csv")
    dp = pd.read_csv(out / "dp_predictions.csv")
    patients = load_patients()

    baseline_utility = json.loads((out / "baseline_metrics.json").read_text())
    dp_utility = pd.read_csv(out / "dp_metrics.csv").set_index("target_epsilon")
    ceiling = bayes_ceiling(seed)

    lira_path = out / "lira_scores.csv"
    if lira_path.exists():
        frame = pd.read_csv(lira_path)
        lira = {t: g.set_index("record_id")["lira_score"]
                for t, g in frame.groupby("target")}
    else:
        print(f"  seed {seed}: no lira_scores.csv, skipping lira")
        lira = {}

    all_rows, all_subgroups, all_roc = [], [], []
    utility, theoretical = {}, {}

    for arm in ("overfit", "tuned"):
        name = f"Non-private ({arm})"
        rows, curves, evaluation, scores = evaluate_target(
            baseline[baseline["arm"] == arm], name, rng, lira.get(name)
        )
        best = max(rows, key=lambda r: r["auc"])["attack"]
        all_rows += rows
        all_subgroups.append(
            subgroup_vulnerability(evaluation, scores[best], patients, name, best, rng)
        )
        all_roc.append(roc_on_grid(curves, name, seed))
        utility[name] = baseline_utility[arm]["test_auc"]
        theoretical[name] = np.inf

    for epsilon, group in sorted(dp.groupby("target_epsilon"), reverse=True):
        name = f"DP eps={epsilon:g}"
        rows, curves, evaluation, scores = evaluate_target(group, name, rng, lira.get(name))
        best = max(rows, key=lambda r: r["auc"])["attack"]
        all_rows += rows
        all_subgroups.append(
            subgroup_vulnerability(evaluation, scores[best], patients, name, best, rng)
        )
        all_roc.append(roc_on_grid(curves, name, seed))
        utility[name] = float(dp_utility.loc[epsilon, "test_auc"])
        theoretical[name] = float(dp_utility.loc[epsilon, "actual_epsilon"])

    results = pd.DataFrame(all_rows).assign(seed=seed)
    results["target_test_auc"] = results["target"].map(utility)
    results["theoretical_epsilon"] = results["target"].map(theoretical)
    results["bayes_ceiling"] = ceiling

    results.to_csv(out / "mia_results.csv", index=False)
    pd.concat(all_subgroups, ignore_index=True).assign(seed=seed).to_csv(
        out / "subgroup_vulnerability.csv", index=False)
    pd.concat(all_roc, ignore_index=True).to_csv(out / "roc_points.csv", index=False)

    strongest = results.sort_values("auc", ascending=False).groupby("target").first()
    print(f"  seed {seed}: " + ", ".join(
        f"{t.replace('Non-private ', '')} {strongest.loc[t, 'auc']:.3f}"
        for t in TARGET_ORDER if t in strongest.index))


def aggregate(seeds):
    # average every metric over the seeds and report the spread.
    OUT_DIR.mkdir(exist_ok=True)

    results = pd.concat(
        [pd.read_csv(RUNS_DIR / f"seed{s}" / "mia_results.csv") for s in seeds],
        ignore_index=True)
    subgroups = pd.concat(
        [pd.read_csv(RUNS_DIR / f"seed{s}" / "subgroup_vulnerability.csv") for s in seeds],
        ignore_index=True)
    roc = pd.concat(
        [pd.read_csv(RUNS_DIR / f"seed{s}" / "roc_points.csv") for s in seeds],
        ignore_index=True)

    n_seeds = results["seed"].nunique()
    ceiling = float(results["bayes_ceiling"].mean())

    attacks = (results
               .groupby(["target", "attack", "threat_model"], as_index=False)
               .agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                    tpr_001_mean=("tpr_at_fpr_0.001", "mean"),
                    tpr_001_std=("tpr_at_fpr_0.001", "std"),
                    tpr_01_mean=("tpr_at_fpr_0.01", "mean"),
                    tpr_01_std=("tpr_at_fpr_0.01", "std"),
                    empirical_epsilon_mean=("empirical_epsilon", "mean"),
                    empirical_epsilon_std=("empirical_epsilon", "std"),
                    n_seeds=("seed", "nunique")))

    # pandas gives NaN for the spread of a single value. a single run has no
    # spread, so record zero and let the seed count speak for itself.
    std_columns = [c for c in attacks.columns if c.endswith("_std")]
    attacks[std_columns] = attacks[std_columns].fillna(0.0)
    attacks.to_csv(OUT_DIR / "mia_results.csv", index=False)

    # pick the strongest attack on its average, not on whichever seed got lucky.
    strongest = (attacks.sort_values("auc_mean", ascending=False)
                 .groupby("target", as_index=False).first())

    utility = (results.groupby("target", as_index=False)
               .agg(target_test_auc_mean=("target_test_auc", "mean"),
                    target_test_auc_std=("target_test_auc", "std"),
                    theoretical_epsilon=("theoretical_epsilon", "mean")))
    utility["target_test_auc_std"] = utility["target_test_auc_std"].fillna(0.0)

    tradeoff = strongest.merge(utility, on="target")
    tradeoff = tradeoff.rename(columns={"auc_mean": "attack_auc_mean",
                                        "auc_std": "attack_auc_std"})
    tradeoff["pct_of_bayes_ceiling"] = tradeoff["target_test_auc_mean"] / ceiling
    tradeoff["label"] = (tradeoff["target"]
                         .str.replace("Non-private ", "Non-private\n")
                         .str.replace("DP eps=", "DP\neps="))
    order = [t for t in TARGET_ORDER if t in set(tradeoff["target"])]
    tradeoff = tradeoff.set_index("target").loc[order].reset_index()
    tradeoff.to_csv(OUT_DIR / "privacy_utility_tradeoff.csv", index=False)

    groups = (subgroups
              .groupby(["target", "dimension", "group"], as_index=False)
              .agg(n_members=("n_members", "mean"),
                   attack_auc_mean=("attack_auc", "mean"),
                   attack_auc_std=("attack_auc", "std"))
              .sort_values("attack_auc_mean", ascending=False))
    groups["attack_auc_std"] = groups["attack_auc_std"].fillna(0.0)
    groups.to_csv(OUT_DIR / "subgroup_vulnerability.csv", index=False)

    private = tradeoff[np.isfinite(tradeoff["theoretical_epsilon"])]
    unprotected = float(tradeoff.loc[
        ~np.isfinite(tradeoff["theoretical_epsilon"]), "empirical_epsilon_mean"].max())

    plot_tradeoff(tradeoff, ROOT / "privacy_utility_tradeoff.png", ceiling, n_seeds)
    plot_loglog_roc(roc, OUT_DIR / "mia_roc_loglog.png")
    plot_epsilon_audit(private, OUT_DIR / "epsilon_audit.png", unprotected, n_seeds)
    plot_subgroups(groups[groups["target"] == "Non-private (overfit)"],
                   OUT_DIR / "subgroup_vulnerability.png", n_seeds)

    pd.set_option("display.width", 220)
    print(f"\nATTACK SUITE  (mean +/- SD over {n_seeds} runs)")
    print("=" * 104)
    view = attacks.assign(
        AUC=lambda d: d["auc_mean"].map("{:.4f}".format) + " +/- " + d["auc_std"].map("{:.4f}".format),
        TPR_at_1pct=lambda d: d["tpr_01_mean"].map("{:.4f}".format) + " +/- " + d["tpr_01_std"].map("{:.4f}".format),
    )[["target", "attack", "threat_model", "AUC", "TPR_at_1pct", "empirical_epsilon_mean"]]
    print(view.to_string(index=False))

    print(f"\n\nHEADLINE  (Bayes-optimal test AUC = {ceiling:.4f}, {n_seeds} runs)")
    print("=" * 104)
    headline = tradeoff.assign(
        attack_AUC=lambda d: d["attack_auc_mean"].map("{:.4f}".format) + " +/- " + d["attack_auc_std"].map("{:.4f}".format),
        test_AUC=lambda d: d["target_test_auc_mean"].map("{:.4f}".format) + " +/- " + d["target_test_auc_std"].map("{:.4f}".format),
        pct_ceiling=lambda d: (100 * d["pct_of_bayes_ceiling"]).map("{:.1f}%".format),
    )[["target", "attack", "attack_AUC", "test_AUC", "pct_ceiling",
       "theoretical_epsilon", "empirical_epsilon_mean"]]
    print(headline.to_string(index=False))

    print("\n\nPRIVACY AUDIT")
    print("=" * 104)
    for _, row in tradeoff.iterrows():
        if not np.isfinite(row["theoretical_epsilon"]):
            print(f"  {row['target']:22s}  no formal guarantee; "
                  f"attack certifies eps >= {row['empirical_epsilon_mean']:.2f}")
            continue
        verdict = "OK" if row["empirical_epsilon_mean"] <= row["theoretical_epsilon"] else "VIOLATION"
        print(f"  {row['target']:22s}  accounted eps={row['theoretical_epsilon']:.2f}  "
              f"empirical eps>={row['empirical_epsilon_mean']:.2f}  [{verdict}]")

    print(f"\n\nMOST EXPOSED PATIENT GROUPS (overfit model, mean of {n_seeds} runs)")
    print("=" * 104)
    top = groups[groups["target"] == "Non-private (overfit)"].head(8)
    print(top[["dimension", "group", "n_members", "attack_auc_mean", "attack_auc_std"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    floor = float(results["fpr_resolution_floor"].iloc[0])
    print(f"\nNOTE: with {int(1 / floor)} non-members the finest resolvable FPR is {floor:.2%}.")
    print(f"Wrote {OUT_DIR}/ and privacy_utility_tradeoff.png")


def main():
    parser = argparse.ArgumentParser(description="attack the models and report the results.")
    parser.add_argument("--seed", type=int, help="score one run")
    parser.add_argument("--aggregate", type=int, nargs="+", metavar="SEED",
                        help="combine the runs for these seeds")
    args = parser.parse_args()

    if args.aggregate:
        aggregate(args.aggregate)
    elif args.seed is not None:
        run_one_seed(args.seed)
    else:
        parser.error("give either --seed N or --aggregate N [N ...]")


if __name__ == "__main__":
    main()
