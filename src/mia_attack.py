
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
FEATURES = ["predicted_probability", "prediction_loss"]

SERIES_ATTACK = "#eb6834"
SERIES_UTILITY = "#2a78d6"
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def run_attack(data, name):
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data["is_member"],
        test_size=0.5,
        stratify=data["is_member"],
        random_state=SEED,
    )

    attacker = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,  
        random_state=SEED,
    ).fit(X_train, y_train)

    scores = attacker.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "attack_accuracy": accuracy_score(y_test, attacker.predict(X_test)),
        "attack_auc": roc_auc_score(y_test, scores),
        "records_tested": len(y_test),
    }


def main():
    baseline = pd.read_csv(ROOT / "baseline_predictions.csv")
    dp = pd.read_csv(ROOT / "dp_predictions.csv")

    baseline_utility = json.loads(
        (ROOT / "baseline_metrics.json").read_text()
    )["test_accuracy"]
    dp_utility = pd.read_csv(ROOT / "dp_metrics.csv").set_index("target_epsilon")[
        "test_accuracy"
    ]

    rows = [{**run_attack(baseline, "Baseline\n(no DP)"), "utility": baseline_utility}]

    for epsilon, group in sorted(dp.groupby("target_epsilon"), reverse=True):
        rows.append(
            {
                **run_attack(group, f"DP\nε={epsilon:g}"),
                "utility": dp_utility[epsilon],
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "mia_results.csv", index=False)

    print("\nMembership Inference Attack")
    print("=" * 62)
    print(f"{'target model':16s} {'attack acc':>11s} {'attack AUC':>11s} {'utility':>9s}")
    for r in rows:
        print(
            f"{r['model'].replace(chr(10), ' '):16s} "
            f"{r['attack_accuracy']:11.3f} {r['attack_auc']:11.3f} {r['utility']:9.3f}"
        )
    print(f"\nSaved {ROOT / 'mia_results.csv'}")

    plot(results)


def plot(results):
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    x = range(len(results))
    ax.plot(x, results["attack_auc"], color=SERIES_ATTACK, lw=2, marker="o", ms=8,
            mec="#fcfcfb", mew=2, label="Attack success (ROC-AUC)")
    ax.plot(x, results["utility"], color=SERIES_UTILITY, lw=2, marker="s", ms=8,
            mec="#fcfcfb", mew=2, label="Target model utility (test accuracy)")

    ax.axhline(0.5, color=INK_MUTED, lw=1, ls=":", zorder=0)
    ax.text(len(results) - 0.5, 0.505, "random guess", ha="right", va="bottom",
            fontsize=9, color=INK_MUTED)

    for i, row in results.iterrows():
        ax.annotate(f"{row['attack_auc']:.2f}", (i, row["attack_auc"]),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    fontsize=9, color=INK)
        ax.annotate(f"{row['utility']:.2f}", (i, row["utility"]),
                    textcoords="offset points", xytext=(0, -18), ha="center",
                    fontsize=9, color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels(results["model"], fontsize=10, color=INK)
    ax.set_xlim(-0.4, len(results) - 0.6)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("rate", fontsize=10, color=INK_MUTED)
    ax.set_xlabel("← weaker privacy guarantee        stronger privacy guarantee →",
                  fontsize=10, color=INK_MUTED, labelpad=12)
    ax.set_title("Privacy / utility trade-off under differentially private training",
                 fontsize=13, color=INK, pad=18, loc="left")

    ax.grid(axis="y", color="#e6e5e1", lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c4")
    ax.tick_params(length=0, colors=INK_MUTED)

    ax.legend(frameon=False, loc="upper right", fontsize=10, labelcolor=INK_MUTED)

    fig.tight_layout()
    out = ROOT / "privacy_utility_tradeoff.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
