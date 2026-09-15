# train one shared model across two hospitals without moving any patient records.

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from train_baseline import (
    run_dir,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ReadmissionModel,
    load_data,
    positive_class_weight,
    predict,
    scale_using_public_bounds,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]

TARGET_EPSILONS = [0.5, 1.0, 3.0, 8.0]
DEPLOYMENT_EPSILON = 3.0

# the tiny chance the privacy promise fails. it must stay far below 1 in 400, the patients per hospital.
DELTA = 1e-5
MAX_GRAD_NORM = 1.0

FEDERATED_ROUNDS = 8
LOCAL_EPOCHS = 2
TOTAL_LOCAL_EPOCHS = FEDERATED_ROUNDS * LOCAL_EPOCHS

BATCH_SIZE = 64
LEARNING_RATE = 0.01
DEFAULT_SEED = 0


def load_clients(seed):
    # one client per hospital. each one trains only on its own patients.
    data = load_data(seed)
    data["_features"] = list(scale_using_public_bounds(data))

    clients = []
    for name, hospital in data.groupby("entity_id", sort=True):
        features = np.stack(hospital["_features"].to_numpy()).astype(np.float32)
        labels = hospital[TARGET_COLUMN].to_numpy(dtype=np.float32)
        is_fit = (hospital["split"] == "fit").to_numpy()

        clients.append(
            {
                "name": name,
                "data": hospital.drop(columns="_features").reset_index(drop=True),
                "all_features": features,
                "all_labels": labels,
                "train_dataset": TensorDataset(
                    torch.tensor(features[is_fit], dtype=torch.float32),
                    torch.tensor(labels[is_fit], dtype=torch.float32),
                ),
                "train_size": int(is_fit.sum()),
                "positive_rate": float(labels[is_fit].mean()),
            }
        )

    return clients


def copy_state_dict(state_dict):
    return {name: value.detach().clone() for name, value in state_dict.items()}


def average_models(state_dicts, client_weights):
    # blend the two hospital models. the one with more patients counts for more.
    total_weight = float(sum(client_weights))
    averaged = {}

    for parameter_name in state_dicts[0]:
        accumulator = torch.zeros_like(state_dicts[0][parameter_name])
        for state, weight in zip(state_dicts, client_weights):
            accumulator += state[parameter_name] * (weight / total_weight)
        averaged[parameter_name] = accumulator

    return averaged


def remove_opacus_prefix(private_state):
    # remove the privacy library wrapper so a plain model can load the weights.
    return {
        (name[len("_module.") :] if name.startswith("_module.") else name): value.detach().clone()
        for name, value in private_state.items()
    }


def create_private_clients(clients, target_epsilon, initial_model_state, pos_weight, seed):
    private_clients = []

    for client in clients:
        local_model = ReadmissionModel(input_size=len(FEATURE_COLUMNS))
        local_model.load_state_dict(initial_model_state)

        local_optimizer = torch.optim.Adam(local_model.parameters(), lr=LEARNING_RATE)

        local_loader = DataLoader(
            client["train_dataset"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        privacy_engine = PrivacyEngine(accountant="prv", secure_mode=False)

        private_model, private_optimizer, private_loader = (
            privacy_engine.make_private_with_epsilon(
                module=local_model,
                optimizer=local_optimizer,
                data_loader=local_loader,
                epochs=TOTAL_LOCAL_EPOCHS,
                target_epsilon=target_epsilon,
                target_delta=DELTA,
                max_grad_norm=MAX_GRAD_NORM,
            )
        )

        private_clients.append(
            {
                "name": client["name"],
                "model": private_model,
                "optimizer": private_optimizer,
                "loader": private_loader,
                "privacy_engine": privacy_engine,
                "train_size": client["train_size"],
                "noise_multiplier": private_optimizer.noise_multiplier,
            }
        )

    return private_clients


def train_federated_model(clients, target_epsilon, pos_weight, seed):
    # fix the randomness here so every run gives the same answer. do not reset it later.
    set_seed(seed)

    initial_model_state = copy_state_dict(
        ReadmissionModel(input_size=len(FEATURE_COLUMNS)).state_dict()
    )

    private_clients = create_private_clients(
        clients, target_epsilon, initial_model_state, pos_weight, seed
    )

    global_state = copy_state_dict(private_clients[0]["model"].state_dict())
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    for round_number in range(1, FEDERATED_ROUNDS + 1):
        local_states, local_weights = [], []

        for client in private_clients:
            client["model"].load_state_dict(global_state)
            client["model"].train()

            for _ in range(LOCAL_EPOCHS):
                for batch_features, batch_labels in client["loader"]:
                    client["optimizer"].zero_grad()
                    loss = criterion(
                        client["model"](batch_features).squeeze(1), batch_labels
                    )
                    loss.backward()
                    client["optimizer"].step()

            local_states.append(copy_state_dict(client["model"].state_dict()))
            local_weights.append(client["train_size"])

        global_state = average_models(local_states, local_weights)

    epsilon_by_hospital = {
        client["name"]: float(client["privacy_engine"].get_epsilon(DELTA))
        for client in private_clients
    }

    return (
        global_state,
        # no patient is at both hospitals, so the total privacy cost is the larger of the two, not the sum.
        max(epsilon_by_hospital.values()),
        epsilon_by_hospital,
        [client["noise_multiplier"] for client in private_clients],
    )


def evaluate_model(private_state, clients, target_epsilon, actual_epsilon, noise_multipliers):
    global_model = ReadmissionModel(input_size=len(FEATURE_COLUMNS))
    global_model.load_state_dict(remove_opacus_prefix(private_state))

    frames = []
    for client in clients:
        probabilities, losses = predict(
            global_model, client["all_features"], client["all_labels"]
        )
        frames.append(
            pd.DataFrame(
                {
                    "record_id": client["data"]["record_id"].to_numpy(),
                    "entity_id": client["data"]["entity_id"].to_numpy(),
                    "split": client["data"]["split"].to_numpy(),
                    "true_label": client["all_labels"].astype(int),
                    "predicted_probability": probabilities,
                    "prediction_loss": losses,
                    "target_epsilon": target_epsilon,
                    "actual_epsilon": actual_epsilon,
                    "delta": DELTA,
                }
            )
        )

    scored = pd.concat(frames, ignore_index=True)

    fit = scored[scored["split"] == "fit"]
    test = scored[scored["split"] == "test"]

    metrics = {
        "target_epsilon": target_epsilon,
        "actual_epsilon": actual_epsilon,
        "delta": DELTA,
        "noise_multiplier": float(np.mean(noise_multipliers)),
        "fit_accuracy": float(accuracy_score(fit["true_label"], fit["predicted_probability"] >= 0.5)),
        "test_accuracy": float(accuracy_score(test["true_label"], test["predicted_probability"] >= 0.5)),
        "fit_auc": float(roc_auc_score(fit["true_label"], fit["predicted_probability"])),
        "test_auc": float(roc_auc_score(test["true_label"], test["predicted_probability"])),
        "fit_loss": float(fit["prediction_loss"].mean()),
        "test_loss": float(test["prediction_loss"].mean()),
    }
    metrics["accuracy_gap"] = metrics["fit_accuracy"] - metrics["test_accuracy"]
    metrics["loss_gap"] = metrics["test_loss"] - metrics["fit_loss"]
    metrics["loss_attack_auc_ceiling"] = 0.5 + metrics["accuracy_gap"] / 2.0

    predictions = pd.concat([fit, test], ignore_index=True).drop(columns="split")
    predictions.insert(
        3, "is_member", predictions["record_id"].isin(fit["record_id"]).astype(int)
    )

    return global_model, predictions, metrics


def main():
    parser = argparse.ArgumentParser(description="train the federated private models.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    out = run_dir(args.seed)
    clients = load_clients(args.seed)

    for client in clients:
        print(
            f"{client['name']}: {client['train_size']} fit records, "
            f"{client['positive_rate']:.1%} readmitted"
        )

    data = load_data(args.seed)
    pos_weight = positive_class_weight(
        data.loc[data["split"] == "fit", TARGET_COLUMN].to_numpy(dtype=np.float32)
    )
    print(f"seed {args.seed}: pos_weight {pos_weight:.4f}")

    all_predictions, all_metrics = [], []

    for target_epsilon in TARGET_EPSILONS:
        state, actual_epsilon, epsilon_by_hospital, noise_multipliers = (
            train_federated_model(clients, target_epsilon, pos_weight, args.seed)
        )

        global_model, predictions, metrics = evaluate_model(
            state, clients, target_epsilon, actual_epsilon, noise_multipliers
        )
        metrics["seed"] = args.seed

        print(
            f"  eps={target_epsilon:<4g} spent {actual_epsilon:.3f} | "
            f"test AUC {metrics['test_auc']:.4f} | gap {metrics['accuracy_gap']:+.4f}"
        )

        all_predictions.append(predictions)
        all_metrics.append(metrics)

        if target_epsilon == DEPLOYMENT_EPSILON:
            torch.save(
                {
                    "model_state_dict": global_model.state_dict(),
                    "feature_columns": FEATURE_COLUMNS,
                    "target_epsilon": target_epsilon,
                    "actual_epsilon": actual_epsilon,
                    "delta": DELTA,
                    "max_grad_norm": MAX_GRAD_NORM,
                    "federated_rounds": FEDERATED_ROUNDS,
                    "local_epochs": LOCAL_EPOCHS,
                    "seed": args.seed,
                },
                out / "dp_model.pt",
            )

    pd.concat(all_predictions, ignore_index=True).to_csv(
        out / "dp_predictions.csv", index=False
    )
    pd.DataFrame(all_metrics).to_csv(out / "dp_metrics.csv", index=False)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
