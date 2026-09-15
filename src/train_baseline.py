# two ordinary models: one trained carelessly, one trained properly.

from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_SEED = 0
BATCH_SIZE = 64
LEARNING_RATE = 0.002

OVERFIT_EPOCHS = 200
TUNED_MAX_EPOCHS = 200
TUNED_WEIGHT_DECAY = 1e-2
TUNED_PATIENCE = 20

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "systolic_bp",
    "glucose",
    "cholesterol",
    "hba1c",
    "smoker",
    "comorbidity_count",
    "heart_failure",
    "kidney_disease",
    "visits_last_year",
    "length_of_stay",
]

TARGET_COLUMN = "readmitted_30d"

# use ranges from a public medical reference. ranges measured from our own patients would leak facts about them.
PUBLIC_BOUNDS = {
    "age": (18, 90),
    "sex": (0, 1),
    "bmi": (16, 50),
    "systolic_bp": (85, 200),
    "glucose": (60, 300),
    "cholesterol": (100, 350),
    "hba1c": (4.0, 14.0),
    "smoker": (0, 1),
    "comorbidity_count": (0, 8),
    "heart_failure": (0, 1),
    "kidney_disease": (0, 1),
    "visits_last_year": (0, 15),
    "length_of_stay": (1, 21),
}

PUBLIC_MINIMUMS = np.array(
    [PUBLIC_BOUNDS[c][0] for c in FEATURE_COLUMNS], dtype=np.float32
)
PUBLIC_MAXIMUMS = np.array(
    [PUBLIC_BOUNDS[c][1] for c in FEATURE_COLUMNS], dtype=np.float32
)


class ReadmissionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features):
        return self.network(features)


def run_dir(seed):
    # every seed writes its own folder. the aggregator reads them all back.
    path = ROOT / "runs" / f"seed{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def scale_using_public_bounds(data):
    # rescale each measurement using public medical ranges.
    features = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    features = np.clip(features, PUBLIC_MINIMUMS, PUBLIC_MAXIMUMS)
    scaled = (features - PUBLIC_MINIMUMS) / (PUBLIC_MAXIMUMS - PUBLIC_MINIMUMS)
    return (2.0 * scaled - 1.0).astype(np.float32)


def load_data(seed=DEFAULT_SEED):
    # load the patients and mark which group each one belongs to.
    frames = [
        pd.read_csv(DATA_DIR / name)
        for name in ("hospital_a.csv", "hospital_b.csv")
    ]
    data = pd.concat(frames, ignore_index=True)
    data["sex"] = data["sex"].map({"Female": 0, "Male": 1})

    split_file = DATA_DIR / f"splits_seed{seed}.csv"
    if not split_file.exists():
        raise FileNotFoundError(f"{split_file} is missing; rerun generate_data.py")

    splits = pd.read_csv(split_file)
    data = data.merge(splits, on="record_id", how="left", validate="one_to_one")

    if data["split"].isnull().any():
        raise ValueError(f"{split_file} does not cover every record; rerun generate_data.py")
    if data[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
        raise ValueError("The dataset contains missing or invalid values.")

    return data


def positive_class_weight(labels):
    # work out how much to weight the rarer outcome so it is not ignored.
    positives = float(labels.sum())
    return float((len(labels) - positives) / positives)


def predict(model, features, labels):
    model.eval()

    feature_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32)

    with torch.no_grad():
        logits = model(feature_tensor).squeeze(1)
        probabilities = torch.sigmoid(logits)
        losses = F.binary_cross_entropy_with_logits(
            logits, label_tensor, reduction="none"
        )

    return probabilities.numpy(), losses.numpy()


def train_arm(arm, fit_features, fit_labels, val_features, val_labels, pos_weight, seed=DEFAULT_SEED):
    # train one model. the careless one runs to the end. the tuned one stops early.
    set_seed(seed)

    model = ReadmissionModel(input_size=len(FEATURE_COLUMNS))
    weight_decay = TUNED_WEIGHT_DECAY if arm == "tuned" else 0.0
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    loader = DataLoader(
        TensorDataset(
            torch.tensor(fit_features, dtype=torch.float32),
            torch.tensor(fit_labels, dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    max_epochs = TUNED_MAX_EPOCHS if arm == "tuned" else OVERFIT_EPOCHS
    best_val_auc, best_state, best_epoch, stale = -np.inf, None, 0, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_features).squeeze(1), batch_labels)
            loss.backward()
            optimizer.step()

        if arm != "tuned":
            continue

        val_probabilities, _ = predict(model, val_features, val_labels)
        val_auc = roc_auc_score(val_labels, val_probabilities)

        if val_auc > best_val_auc:
            best_val_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= TUNED_PATIENCE:
                break

    if arm == "tuned":
        model.load_state_dict(best_state)
        print(f"  tuned: stopped at epoch {epoch}, best val AUC {best_val_auc:.4f} (epoch {best_epoch})")
    else:
        best_epoch = max_epochs
        print(f"  overfit: ran all {max_epochs} epochs, no early stopping")

    return model, best_epoch


def evaluate(model, data, features, labels, arm):
    # score every patient, then keep only the ones the attack uses.
    probabilities, losses = predict(model, features, labels)

    scored = pd.DataFrame(
        {
            "record_id": data["record_id"].to_numpy(),
            "entity_id": data["entity_id"].to_numpy(),
            "arm": arm,
            "split": data["split"].to_numpy(),
            "true_label": labels.astype(int),
            "predicted_probability": probabilities,
            "prediction_loss": losses,
        }
    )

    # the model never learned from "val" rows, but it did look at them. they fit neither group, so leave them out.
    fit = scored[scored["split"] == "fit"]
    test = scored[scored["split"] == "test"]

    metrics = {
        "arm": arm,
        "fit_accuracy": float(accuracy_score(fit["true_label"], fit["predicted_probability"] >= 0.5)),
        "test_accuracy": float(accuracy_score(test["true_label"], test["predicted_probability"] >= 0.5)),
        "fit_auc": float(roc_auc_score(fit["true_label"], fit["predicted_probability"])),
        "test_auc": float(roc_auc_score(test["true_label"], test["predicted_probability"])),
        "fit_loss": float(fit["prediction_loss"].mean()),
        "test_loss": float(test["prediction_loss"].mean()),
        "fit_records": int(len(fit)),
        "test_records": int(len(test)),
    }
    metrics["accuracy_gap"] = metrics["fit_accuracy"] - metrics["test_accuracy"]
    metrics["loss_gap"] = metrics["test_loss"] - metrics["fit_loss"]

    # the best a simple attack could ever score. we check the real attack against it.
    metrics["loss_attack_auc_ceiling"] = 0.5 + metrics["accuracy_gap"] / 2.0

    membership = pd.concat([fit, test], ignore_index=True).drop(columns="split")
    membership.insert(3, "is_member", (membership["record_id"].isin(fit["record_id"])).astype(int))

    return membership, metrics


def main():
    parser = argparse.ArgumentParser(description="train the two non-private models.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    out = run_dir(args.seed)
    data = load_data(args.seed)

    features = scale_using_public_bounds(data)
    labels = data[TARGET_COLUMN].to_numpy(dtype=np.float32)

    is_fit = (data["split"] == "fit").to_numpy()
    is_val = (data["split"] == "val").to_numpy()

    pos_weight = positive_class_weight(labels[is_fit])
    print(f"seed {args.seed}: pos_weight {pos_weight:.4f}, {int(is_fit.sum())} fit records")

    all_predictions, all_metrics = [], []

    for arm in ("overfit", "tuned"):
        model, epochs_used = train_arm(
            arm,
            features[is_fit],
            labels[is_fit],
            features[is_val],
            labels[is_val],
            pos_weight,
            seed=args.seed,
        )

        membership, metrics = evaluate(model, data, features, labels, arm)
        metrics["seed"] = args.seed
        metrics["epochs_used"] = epochs_used
        metrics["pos_weight"] = pos_weight
        metrics["weight_decay"] = TUNED_WEIGHT_DECAY if arm == "tuned" else 0.0

        all_predictions.append(membership)
        all_metrics.append(metrics)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "arm": arm,
                "input_size": len(FEATURE_COLUMNS),
                "feature_columns": FEATURE_COLUMNS,
                "public_bounds": PUBLIC_BOUNDS,
                "seed": args.seed,
            },
            out / f"baseline_model_{arm}.pt",
        )

        print(
            f"  {arm:8s} fit AUC {metrics['fit_auc']:.4f} | test AUC {metrics['test_auc']:.4f} | "
            f"gap {metrics['accuracy_gap']:+.4f}"
        )

    pd.concat(all_predictions, ignore_index=True).to_csv(
        out / "baseline_predictions.csv", index=False
    )
    (out / "baseline_metrics.json").write_text(
        json.dumps({m["arm"]: m for m in all_metrics}, indent=4)
    )

    print(f"saved to {out}")


if __name__ == "__main__":
    main()
