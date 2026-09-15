# checks on the things that would quietly make our privacy claims wrong.

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lira import logit_scale
from mia_attack import audit_epsilon, clopper_pearson, tpr_at_fpr

RUNS = ROOT / "runs"
DATA = ROOT / "data"


def seeds():
    return sorted(int(p.name.removeprefix("seed")) for p in RUNS.glob("seed*"))


def patients():
    return pd.concat(
        [pd.read_csv(DATA / n) for n in ("hospital_a.csv", "hospital_b.csv")],
        ignore_index=True,
    )


def bayes_ceiling(seed):
    oracle = pd.read_csv(DATA / "oracle.csv")
    splits = pd.read_csv(DATA / f"splits_seed{seed}.csv")
    labels = patients()[["record_id", "readmitted_30d"]]
    joined = labels.merge(oracle, on="record_id").merge(splits, on="record_id")
    test = joined[joined["split"] == "test"]
    return roc_auc_score(test["readmitted_30d"], test["true_readmission_probability"])


def test_splits_are_complete_and_exclusive():
    everyone = set(patients()["record_id"])
    for seed in seeds():
        splits = pd.read_csv(DATA / f"splits_seed{seed}.csv")
        assert splits["record_id"].is_unique
        assert set(splits["record_id"]) == everyone
        assert set(splits["split"]) == {"fit", "val", "test"}


def test_each_seed_trains_on_a_different_set():
    # if the seeds shared one split, repeating the run would prove nothing.
    members = []
    for seed in seeds():
        splits = pd.read_csv(DATA / f"splits_seed{seed}.csv")
        members.append(frozenset(splits.loc[splits["split"] == "fit", "record_id"]))
    assert len(set(members)) == len(members), "two seeds share the same fit set"


def test_membership_labels_match_the_canonical_split():
    # the one assumption that every attack result depends on.
    for seed in seeds():
        splits = pd.read_csv(DATA / f"splits_seed{seed}.csv").set_index("record_id")["split"]
        for name in ("baseline_predictions.csv", "dp_predictions.csv"):
            predictions = pd.read_csv(RUNS / f"seed{seed}" / name)
            actual = predictions["record_id"].map(splits)
            assert (actual[predictions["is_member"] == 1] == "fit").all(), (seed, name)
            assert (actual[predictions["is_member"] == 0] == "test").all(), (seed, name)
            assert not (actual == "val").any(), f"seed{seed}/{name} leaks val records"


def test_both_arms_share_one_membership_definition():
    # if the two models trained on different patients, comparing them means nothing.
    for seed in seeds():
        predictions = pd.read_csv(RUNS / f"seed{seed}" / "baseline_predictions.csv")
        members = {
            arm: frozenset(group.loc[group["is_member"] == 1, "record_id"])
            for arm, group in predictions.groupby("arm")
        }
        assert len(set(members.values())) == 1, seed


def test_logit_scale_is_finite_at_saturation():
    # a confidence can round to exactly 1. the maths must still work.
    probabilities = np.array([1.0, 0.0, 0.5, 1e-9], dtype=np.float32)
    labels = np.array([1, 0, 1, 0])
    assert np.isfinite(logit_scale(probabilities, labels)).all()


def test_epsilon_audit_is_quiet_for_a_useless_attack():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 500)
    assert audit_epsilon(y, rng.normal(size=1000)) < 0.5


def test_epsilon_audit_fires_on_a_perfect_attack():
    y = np.repeat([0, 1], 500)
    assert audit_epsilon(y, y.astype(float)) > 3.0


def test_clopper_pearson_brackets_the_estimate():
    lo, hi = clopper_pearson(5, 100)
    assert lo < 0.05 < hi
    assert clopper_pearson(0, 100)[0] == 0.0
    assert clopper_pearson(100, 100)[1] == 1.0


def test_tpr_at_fpr_is_monotone_in_its_budget():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 500)
    score = rng.normal(size=1000) + y
    assert 0.0 <= tpr_at_fpr(y, score, 0.01) <= 1.0
    assert tpr_at_fpr(y, score, 0.01) <= tpr_at_fpr(y, score, 0.1)


def test_no_model_beats_the_bayes_ceiling():
    # a model that beats the theoretical best is leaking, not learning.
    for seed in seeds():
        ceiling = bayes_ceiling(seed)
        metrics = json.loads((RUNS / f"seed{seed}" / "baseline_metrics.json").read_text())
        for arm, values in metrics.items():
            assert values["test_auc"] <= ceiling + 0.02, (seed, arm)
        for _, row in pd.read_csv(RUNS / f"seed{seed}" / "dp_metrics.csv").iterrows():
            assert row["test_auc"] <= ceiling + 0.02, (seed, row["target_epsilon"])


def test_spent_epsilon_never_exceeds_its_target():
    for seed in seeds():
        dp = pd.read_csv(RUNS / f"seed{seed}" / "dp_metrics.csv")
        assert (dp["actual_epsilon"] <= dp["target_epsilon"] + 1e-6).all(), seed
        # delta must stay far below 1 over the records held by one hospital.
        assert (dp["delta"] < 1.0 / 1000).all(), seed


def test_no_attack_exceeds_the_accounted_budget():
    path = ROOT / "results" / "privacy_utility_tradeoff.csv"
    if not path.exists():
        print("    (skipped: run the aggregate step first)", end=" ")
        return
    tradeoff = pd.read_csv(path)
    private = tradeoff[np.isfinite(tradeoff["theoretical_epsilon"])]
    assert (private["empirical_epsilon_mean"] <= private["theoretical_epsilon"]).all()


def test_aggregate_covers_every_seed():
    # a silently dropped run would shrink the error bars for free.
    path = ROOT / "results" / "mia_results.csv"
    if not path.exists():
        print("    (skipped: run the aggregate step first)", end=" ")
        return
    assert (pd.read_csv(path)["n_seeds"] == len(seeds())).all()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL  {test.__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
