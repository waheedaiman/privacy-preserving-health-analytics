from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from train_baseline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ReadmissionModel,
    predict,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TARGET_EPSILONS = [0.5, 1.0, 3.0, 8.0]
DEPLOYMENT_EPSILON = 3.0

DELTA = 1e-5
MAX_GRAD_NORM = 1.0

FEDERATED_ROUNDS = 8
LOCAL_EPOCHS = 2
TOTAL_LOCAL_EPOCHS = FEDERATED_ROUNDS * LOCAL_EPOCHS

BATCH_SIZE = 64
LEARNING_RATE = 0.01
MODEL_SEED = 42



PUBLIC_MINIMUMS = np.array(
    [
        18,   # age
        0,    # sex
        16,   # bmi
        85,   # systolic_bp
        60,   # glucose
        100,  # cholesterol
        0,    # smoker
        0,    # comorbidity_count
        0,    # visits_last_year
        1,    # length_of_stay
    ],
    dtype=np.float32,
)

PUBLIC_MAXIMUMS = np.array(
    [
        90,
        1,
        50,
        200,
        300,
        350,
        1,
        8,
        15,
        21,
    ],
    dtype=np.float32,
)


def scale_using_public_bounds(data):

    features = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    features = np.clip(
        features,
        PUBLIC_MINIMUMS,
        PUBLIC_MAXIMUMS,
    )

    scaled = (
        (features - PUBLIC_MINIMUMS)
        / (PUBLIC_MAXIMUMS - PUBLIC_MINIMUMS)
    )

    return (2.0 * scaled - 1.0).astype(np.float32)


def load_clients():

    membership = pd.read_csv(
        ROOT / "baseline_predictions.csv"
    )[["record_id", "is_member"]]

    clients = []

    for filename in ["hospital_a.csv", "hospital_b.csv"]:
        hospital_data = pd.read_csv(DATA_DIR / filename)

        hospital_data["sex"] = hospital_data["sex"].map(
            {
                "Female": 0,
                "Male": 1,
            }
        )

        hospital_data = hospital_data.merge(
            membership,
            on="record_id",
            how="left",
            validate="one_to_one",
        )

        if hospital_data["is_member"].isnull().any():
            raise ValueError(
                f"Membership information is missing for {filename}."
            )

        all_features = scale_using_public_bounds(hospital_data)

        all_labels = hospital_data[TARGET_COLUMN].to_numpy(
            dtype=np.float32
        )

        member_mask = (
            hospital_data["is_member"].to_numpy() == 1
        )

        train_features = all_features[member_mask]
        train_labels = all_labels[member_mask]

        train_dataset = TensorDataset(
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.float32),
        )

        clients.append(
            {
                "name": hospital_data["entity_id"].iloc[0],
                "data": hospital_data,
                "all_features": all_features,
                "all_labels": all_labels,
                "train_dataset": train_dataset,
                "train_size": len(train_dataset),
            }
        )

    return clients


def copy_state_dict(state_dict):
    return {
        name: value.detach().clone()
        for name, value in state_dict.items()
    }


def average_models(state_dicts, client_weights):

    total_weight = float(sum(client_weights))
    averaged_state = {}

    for parameter_name in state_dicts[0]:
        averaged_parameter = torch.zeros_like(
            state_dicts[0][parameter_name]
        )

        for state, weight in zip(
            state_dicts,
            client_weights,
        ):
            averaged_parameter += (
                state[parameter_name]
                * (weight / total_weight)
            )

        averaged_state[parameter_name] = averaged_parameter

    return averaged_state


def remove_opacus_prefix(private_state):

    cleaned_state = {}

    for name, value in private_state.items():
        if name.startswith("_module."):
            clean_name = name[len("_module."):]
        else:
            clean_name = name

        cleaned_state[clean_name] = value.detach().clone()

    return cleaned_state


def create_private_clients(
    clients,
    target_epsilon,
    initial_model_state,
):
    private_clients = []

    for client in clients:
        local_model = ReadmissionModel(
            input_size=len(FEATURE_COLUMNS)
        )

        local_model.load_state_dict(initial_model_state)

        local_optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=LEARNING_RATE,
        )

        local_loader = DataLoader(
            client["train_dataset"],
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        privacy_engine = PrivacyEngine(
            accountant="prv",
            secure_mode=False,
        )

        (
            private_model,
            private_optimizer,
            private_loader,
        ) = privacy_engine.make_private_with_epsilon(
            module=local_model,
            optimizer=local_optimizer,
            data_loader=local_loader,
            epochs=TOTAL_LOCAL_EPOCHS,
            target_epsilon=target_epsilon,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )

        private_clients.append(
            {
                "name": client["name"],
                "model": private_model,
                "optimizer": private_optimizer,
                "loader": private_loader,
                "privacy_engine": privacy_engine,
                "train_size": client["train_size"],
                "noise_multiplier": (
                    private_optimizer.noise_multiplier
                ),
            }
        )

    return private_clients


def train_federated_model(clients, target_epsilon):
    torch.manual_seed(MODEL_SEED)

    initial_model = ReadmissionModel(
        input_size=len(FEATURE_COLUMNS)
    )

    initial_model_state = copy_state_dict(
        initial_model.state_dict()
    )

    torch.seed()

    private_clients = create_private_clients(
        clients,
        target_epsilon,
        initial_model_state,
    )

    global_private_state = copy_state_dict(
        private_clients[0]["model"].state_dict()
    )

    criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([1.75])
    )

    for round_number in range(
        1,
        FEDERATED_ROUNDS + 1,
    ):
        local_states = []
        local_weights = []

        print(
            f"\nEpsilon {target_epsilon} | "
            f"Federated round {round_number}/"
            f"{FEDERATED_ROUNDS}"
        )

        for client in private_clients:
            client["model"].load_state_dict(
                global_private_state
            )

            client["model"].train()
            local_losses = []

            for _ in range(LOCAL_EPOCHS):
                for batch_features, batch_labels in client["loader"]:
                    client["optimizer"].zero_grad()

                    logits = client["model"](
                        batch_features
                    ).squeeze(1)

                    loss = criterion(
                        logits,
                        batch_labels,
                    )

                    loss.backward()
                    client["optimizer"].step()

                    local_losses.append(loss.item())

            current_epsilon = client[
                "privacy_engine"
            ].get_epsilon(DELTA)

            print(
                f"  {client['name']}: "
                f"loss={np.mean(local_losses):.4f}, "
                f"spent ε={current_epsilon:.3f}"
            )

            local_states.append(
                copy_state_dict(
                    client["model"].state_dict()
                )
            )

            local_weights.append(
                client["train_size"]
            )

        global_private_state = average_models(
            local_states,
            local_weights,
        )

    epsilon_by_hospital = {
        client["name"]: float(
            client["privacy_engine"].get_epsilon(
                DELTA
            )
        )
        for client in private_clients
    }

    actual_epsilon = max(
        epsilon_by_hospital.values()
    )

    noise_multipliers = [
        client["noise_multiplier"]
        for client in private_clients
    ]

    return (
        global_private_state,
        actual_epsilon,
        epsilon_by_hospital,
        noise_multipliers,
    )


def evaluate_model(
    private_state,
    clients,
    target_epsilon,
    actual_epsilon,
    noise_multipliers,
):
    global_model = ReadmissionModel(
        input_size=len(FEATURE_COLUMNS)
    )

    global_model.load_state_dict(
        remove_opacus_prefix(private_state)
    )

    prediction_frames = []

    for client in clients:
        probabilities, losses = predict(
            global_model,
            client["all_features"],
            client["all_labels"],
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "record_id": client["data"][
                        "record_id"
                    ].values,
                    "entity_id": client["data"][
                        "entity_id"
                    ].values,
                    "true_label": client[
                        "all_labels"
                    ].astype(int),
                    "is_member": client["data"][
                        "is_member"
                    ].astype(int).values,
                    "predicted_probability": probabilities,
                    "prediction_loss": losses,
                    "target_epsilon": target_epsilon,
                    "actual_epsilon": actual_epsilon,
                    "delta": DELTA,
                }
            )
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    train_results = predictions[
        predictions["is_member"] == 1
    ]

    test_results = predictions[
        predictions["is_member"] == 0
    ]

    train_accuracy = accuracy_score(
        train_results["true_label"],
        train_results["predicted_probability"] >= 0.5,
    )

    test_accuracy = accuracy_score(
        test_results["true_label"],
        test_results["predicted_probability"] >= 0.5,
    )

    metrics = {
        "target_epsilon": target_epsilon,
        "actual_epsilon": actual_epsilon,
        "delta": DELTA,
        "noise_multiplier": float(
            np.mean(noise_multipliers)
        ),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "train_auc": float(
            roc_auc_score(
                train_results["true_label"],
                train_results["predicted_probability"],
            )
        ),
        "test_auc": float(
            roc_auc_score(
                test_results["true_label"],
                test_results["predicted_probability"],
            )
        ),
        "train_loss": float(
            train_results["prediction_loss"].mean()
        ),
        "test_loss": float(
            test_results["prediction_loss"].mean()
        ),
        "accuracy_gap": float(
            train_accuracy - test_accuracy
        ),
        "loss_gap": float(
            test_results["prediction_loss"].mean()
            - train_results["prediction_loss"].mean()
        ),
    }

    return global_model, predictions, metrics


def main():
    clients = load_clients()

    all_predictions = []
    all_metrics = []

    for target_epsilon in TARGET_EPSILONS:
        print(
            "\n"
            + "=" * 60
            + f"\nTraining private model for ε={target_epsilon}"
        )

        (
            private_state,
            actual_epsilon,
            epsilon_by_hospital,
            noise_multipliers,
        ) = train_federated_model(
            clients,
            target_epsilon,
        )

        (
            global_model,
            predictions,
            metrics,
        ) = evaluate_model(
            private_state,
            clients,
            target_epsilon,
            actual_epsilon,
            noise_multipliers,
        )

        for hospital, epsilon in epsilon_by_hospital.items():
            print(
                f"{hospital} final privacy budget: "
                f"(ε={epsilon:.3f}, δ={DELTA})"
            )

        print(
            f"Private test accuracy: "
            f"{metrics['test_accuracy']:.3f}"
        )

        print(
            f"Private test AUC: "
            f"{metrics['test_auc']:.3f}"
        )

        all_predictions.append(predictions)
        all_metrics.append(metrics)

        if target_epsilon == DEPLOYMENT_EPSILON:
            torch.save(
                {
                    "model_state_dict": (
                        global_model.state_dict()
                    ),
                    "feature_columns": FEATURE_COLUMNS,
                    "target_epsilon": target_epsilon,
                    "actual_epsilon": actual_epsilon,
                    "delta": DELTA,
                    "max_grad_norm": MAX_GRAD_NORM,
                    "federated_rounds": FEDERATED_ROUNDS,
                    "local_epochs": LOCAL_EPOCHS,
                },
                ROOT / "dp_model.pt",
            )

    combined_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics_table = pd.DataFrame(all_metrics)

    combined_predictions.to_csv(
        ROOT / "dp_predictions.csv",
        index=False,
    )

    metrics_table.to_csv(
        ROOT / "dp_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("DP training completed.")
    print("\nPrivacy–utility results:")
    print(metrics_table.to_string(index=False))

    print("\nSaved dp_model.pt")
    print("Saved dp_predictions.csv")
    print("Saved dp_metrics.csv")


if __name__ == "__main__":
    main()
