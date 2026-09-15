# build the synthetic hospital cohorts and the train/test splits for each seed.

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


COHORT_SEED = 42
ROWS_PER_HOSPITAL = 3000
DEFAULT_SEEDS = [0, 1, 2]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def generate_hospital(
    hospital_name,
    prefix,
    age_mean,
    comorbidity_rate,
    visit_rate,
    smoking_rate,
    risk_shift,
    rng,
):
    n = ROWS_PER_HOSPITAL

    age = np.clip(rng.normal(age_mean, 15, n), 18, 90).round().astype(int)
    sex = rng.choice(["Female", "Male"], n)

    comorbidities = np.clip(
        rng.poisson(comorbidity_rate, n), 0, 8
    )

    bmi = np.clip(
        rng.normal(25 + (comorbidities * 1.2), 4.5, n),
        16,
        50,
    ).round(1)

    systolic_bp = np.clip(
        rng.normal(
            115 + (age * 0.2) + (comorbidities * 3),
            12,
            n,
        ),
        85,
        200,
    ).round().astype(int)

    glucose = np.clip(
        rng.normal(
            92 + (comorbidities * 7),
            18,
            n,
        ),
        60,
        300,
    ).round().astype(int)

    cholesterol = np.clip(
        rng.normal(
            160 + (age * 0.45) + (comorbidities * 5),
            25,
            n,
        ),
        100,
        350,
    ).round().astype(int)

    # long term blood sugar. it tracks glucose but drifts on its own.
    hba1c = np.clip(
        rng.normal(4.4 + (glucose * 0.019), 0.6, n),
        4.0,
        14.0,
    ).round(1)

    smoker = rng.binomial(1, smoking_rate, n)
    visits_last_year = np.clip(rng.poisson(visit_rate, n), 0, 15)

    # two long term conditions. both get more common with age and other illness.
    heart_failure = rng.binomial(
        1, sigmoid(-4.2 + 0.035 * (age - 50) + 0.45 * comorbidities)
    )
    kidney_disease = rng.binomial(
        1, sigmoid(-4.6 + 0.40 * comorbidities + 0.018 * (systolic_bp - 120))
    )

    length_of_stay = np.clip(
        rng.normal(
            2 + (comorbidities * 0.8) + (visits_last_year * 0.25) + (1.1 * heart_failure),
            1.5,
            n,
        ),
        1,
        21,
    ).round().astype(int)

    risk_score = (
        -2.9
        + 0.03 * (age - 50)
        + 0.035 * (bmi - 25)
        + 0.012 * (systolic_bp - 120)
        + 0.010 * (glucose - 100)
        + 0.22 * (hba1c - 5.7)
        + 0.45 * smoker
        + 0.28 * comorbidities
        + 0.15 * visits_last_year
        + 0.08 * length_of_stay
        + 0.60 * heart_failure
        + 0.45 * kidney_disease
        + risk_shift
    )

    readmission_probability = sigmoid(risk_score)
    readmitted_30d = rng.binomial(1, readmission_probability)

    record_ids = [f"{prefix}-{number:05d}" for number in range(1, n + 1)]

    oracle = pd.DataFrame(
        {
            "record_id": record_ids,
            "true_readmission_probability": readmission_probability,
        }
    )

    records = pd.DataFrame(
        {
            "record_id": record_ids,
            "entity_id": hospital_name,
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "glucose": glucose,
            "cholesterol": cholesterol,
            "hba1c": hba1c,
            "smoker": smoker,
            "comorbidity_count": comorbidities,
            "heart_failure": heart_failure,
            "kidney_disease": kidney_disease,
            "visits_last_year": visits_last_year,
            "length_of_stay": length_of_stay,
            "readmitted_30d": readmitted_30d,
        }
    )

    return records, oracle


def validate_dataset(data):
    assert not data.isnull().any().any()
    assert data["record_id"].is_unique
    assert data["age"].between(18, 90).all()
    assert data["bmi"].between(16, 50).all()
    assert data["hba1c"].between(4.0, 14.0).all()
    assert set(data["heart_failure"].unique()).issubset({0, 1})
    assert set(data["kidney_disease"].unique()).issubset({0, 1})
    assert set(data["readmitted_30d"].unique()).issubset({0, 1})


def assign_splits(datasets, seed):
    # decide which patients are used for what. every script reads this.
    rng = np.random.default_rng(seed)
    rows = []

    for dataset in datasets:
        for _, stratum in dataset.groupby(["entity_id", "readmitted_30d"]):
            ids = rng.permutation(stratum["record_id"].to_numpy())

            # the model learns from "fit" rows. "val" rows only say when to stop.
            # it never sees "test" rows.
            n_test = round(0.5 * len(ids))
            n_val = round(0.1 * len(ids))

            labels = (
                ["test"] * n_test
                + ["val"] * n_val
                + ["fit"] * (len(ids) - n_test - n_val)
            )
            rows.append(pd.DataFrame({"record_id": ids, "split": labels}))

    splits = pd.concat(rows, ignore_index=True).sort_values("record_id")
    return splits.reset_index(drop=True)


def splits_path(seed):
    return OUTPUT_DIR / f"splits_seed{seed}.csv"


def main():
    parser = argparse.ArgumentParser(
        description="build the synthetic cohorts and one split per seed."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    # the patients themselves stay fixed. a hospital has the patients it has.
    # only the split changes per seed, so membership changes and the attack is
    # measured against a different training set each run.
    rng = np.random.default_rng(COHORT_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    hospital_a, oracle_a = generate_hospital(
        hospital_name="Hospital A",
        prefix="HA",
        age_mean=45,
        comorbidity_rate=1.2,
        visit_rate=2.0,
        smoking_rate=0.18,
        risk_shift=-0.1,
        rng=rng,
    )

    hospital_b, oracle_b = generate_hospital(
        hospital_name="Hospital B",
        prefix="HB",
        age_mean=57,
        comorbidity_rate=2.2,
        visit_rate=3.2,
        smoking_rate=0.22,
        risk_shift=0.1,
        rng=rng,
    )

    for filename, dataset in [
        ("hospital_a.csv", hospital_a),
        ("hospital_b.csv", hospital_b),
    ]:
        validate_dataset(dataset)
        dataset.to_csv(OUTPUT_DIR / filename, index=False)

        rate = dataset["readmitted_30d"].mean() * 100
        print(f"{filename}: {len(dataset)} records, {rate:.1f}% readmitted")

    # keep the real answer in a separate file, so it can never be fed to the model
    # by mistake.
    pd.concat([oracle_a, oracle_b], ignore_index=True).to_csv(
        OUTPUT_DIR / "oracle.csv", index=False
    )

    for seed in args.seeds:
        splits = assign_splits([hospital_a, hospital_b], seed)
        splits.to_csv(splits_path(seed), index=False)

        counts = splits["split"].value_counts().to_dict()
        print(
            f"splits_seed{seed}.csv: "
            + ", ".join(f"{name}={counts[name]}" for name in ("fit", "val", "test"))
        )


if __name__ == "__main__":
    main()
