from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

SEED = 2026
CALIBRATION_SIZE = 0.5


def add_membership_score(data):
    """
    Build a confidence-based membership score.

    A higher confidence prediction is treated as stronger
    evidence that the record was present during training.
    """

    data = data.copy()

    data["membership_score"] = np.maximum(
        data["predicted_probability"],
        1.0 - data["predicted_probability"],
    )

    return data


def evaluate_attack(data, model_name):
    """
    Evaluate a confidence-based membership inference attack.

    The attack threshold is selected using a calibration subset
    and evaluated on a separate attack-test subset.
    """

    data = add_membership_score(data)

    calibration, attack_test = train_test_split(
        data,
        test_size=CALIBRATION_SIZE,
        stratify=data["is_member"],
        random_state=SEED,
    )

    # ROC-AUC is threshold independent.
    auc = roc_auc_score(
        attack_test["is_member"],
        attack_test["membership_score"],
    )

    # Select the decision threshold using ONLY calibration data.
    fpr, tpr, thresholds = roc_curve(
        calibration["is_member"],
        calibration["membership_score"],
    )

    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)

    threshold = thresholds[best_index]

    attack_predictions = (
        attack_test["membership_score"] >= threshold
    ).astype(int)

    attack_accuracy = accuracy_score(
        attack_test["is_member"],
        attack_predictions,
    )

    true_members = attack_test["is_member"] == 1
    true_nonmembers = attack_test["is_member"] == 0

    member_confidence = attack_test.loc[
        true_members,
        "membership_score",
    ].mean()

    nonmember_confidence = attack_test.loc[
        true_nonmembers,
        "membership_score",
    ].mean()

    return {
        "model": model_name,
        "mia_auc": float(auc),
        "attack_accuracy": float(attack_accuracy),
        "threshold": float(threshold),
        "member_mean_confidence": float(member_confidence),
        "nonmember_mean_confidence": float(nonmember_confidence),
        "records_tested": int(len(attack_test)),
    }


def main():

    results = []

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    baseline = pd.read_csv(
        ROOT / "baseline_predictions.csv"
    )

    baseline_result = evaluate_attack(
        baseline,
        "Baseline",
    )

    results.append(baseline_result)

    # ---------------------------------------------------------
    # Differential privacy models
    # ---------------------------------------------------------

    dp = pd.read_csv(
        ROOT / "dp_predictions.csv"
    )

    for epsilon, group in dp.groupby(
        "target_epsilon"
    ):

        result = evaluate_attack(
            group,
            f"DP ε={epsilon:g}",
        )

        result["target_epsilon"] = float(epsilon)

        results.append(result)

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        ROOT / "mia_results.csv",
        index=False,
    )

    print("\nMembership Inference Attack Results")
    print("=" * 55)

    for result in results:
        print(
            f"{result['model']:12s} | "
            f"AUC: {result['mia_auc']:.3f} | "
            f"Accuracy: {result['attack_accuracy']:.3f}"
        )

    print(
        f"\nSaved results to: "
        f"{ROOT / 'mia_results.csv'}"
    )


if __name__ == "__main__":
    main()
