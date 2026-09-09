from pathlib import Path
import json
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
EPOCHS = 200
BATCH_SIZE = 64

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "systolic_bp",
    "glucose",
    "cholesterol",
    "smoker",
    "comorbidity_count",
    "visits_last_year",
    "length_of_stay",
]

TARGET_COLUMN = "readmitted_30d"


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


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_data():
    hospital_a = pd.read_csv(DATA_DIR / "hospital_a.csv")
    hospital_b = pd.read_csv(DATA_DIR / "hospital_b.csv")

    data = pd.concat([hospital_a, hospital_b], ignore_index=True)

    data["sex"] = data["sex"].map(
        {
            "Female": 0,
            "Male": 1,
        }
    )

    if data[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
        raise ValueError("The dataset contains missing or invalid values.")

    return data


def predict(model, features, labels):
    model.eval()

    feature_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32)

    with torch.no_grad():
        logits = model(feature_tensor).squeeze(1)
        probabilities = torch.sigmoid(logits)

        losses = F.binary_cross_entropy_with_logits(
            logits,
            label_tensor,
            reduction="none",
        )

    return probabilities.numpy(), losses.numpy()


def main():
    set_seed()
    data = load_data()

    features = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    labels = data[TARGET_COLUMN].to_numpy(dtype=np.float32)
    indices = np.arange(len(data))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.5,
        random_state=SEED,
        stratify=labels,
    )

    scaler = StandardScaler()

    train_features = scaler.fit_transform(features[train_indices])
    test_features = scaler.transform(features[test_indices])

    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    train_dataset = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.float32),
    )

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = ReadmissionModel(input_size=len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()

            logits = model(batch_features).squeeze(1)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Training loss: {np.mean(epoch_losses):.4f}"
            )

    train_probabilities, train_losses = predict(
        model,
        train_features,
        train_labels,
    )

    test_probabilities, test_losses = predict(
        model,
        test_features,
        test_labels,
    )

    train_accuracy = accuracy_score(
        train_labels,
        train_probabilities >= 0.5,
    )

    test_accuracy = accuracy_score(
        test_labels,
        test_probabilities >= 0.5,
    )

    metrics = {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "train_auc": float(
            roc_auc_score(train_labels, train_probabilities)
        ),
        "test_auc": float(
            roc_auc_score(test_labels, test_probabilities)
        ),
        "train_loss": float(np.mean(train_losses)),
        "test_loss": float(np.mean(test_losses)),
        "accuracy_gap": float(train_accuracy - test_accuracy),
        "loss_gap": float(
            np.mean(test_losses) - np.mean(train_losses)
        ),
        "training_records": int(len(train_indices)),
        "testing_records": int(len(test_indices)),
    }

    train_predictions = pd.DataFrame(
        {
            "record_id": data.iloc[train_indices]["record_id"].values,
            "true_label": train_labels.astype(int),
            "is_member": 1,
            "predicted_probability": train_probabilities,
            "prediction_loss": train_losses,
        }
    )

    test_predictions = pd.DataFrame(
        {
            "record_id": data.iloc[test_indices]["record_id"].values,
            "true_label": test_labels.astype(int),
            "is_member": 0,
            "predicted_probability": test_probabilities,
            "prediction_loss": test_losses,
        }
    )

    membership_predictions = pd.concat(
        [train_predictions, test_predictions],
        ignore_index=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(FEATURE_COLUMNS),
            "feature_columns": FEATURE_COLUMNS,
            "seed": SEED,
        },
        ROOT / "baseline_model.pt",
    )

    with open(ROOT / "baseline_scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    with open(ROOT / "baseline_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    membership_predictions.to_csv(
        ROOT / "baseline_predictions.csv",
        index=False,
    )

    print("\nBaseline training completed.")
    print(json.dumps(metrics, indent=4))
    print("\nSaved baseline_model.pt")
    print("Saved baseline_scaler.pkl")
    print("Saved baseline_metrics.json")
    print("Saved baseline_predictions.csv")


if __name__ == "__main__":
    main()
