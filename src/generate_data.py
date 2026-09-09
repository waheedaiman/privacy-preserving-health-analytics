from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
ROWS_PER_HOSPITAL = 1000
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

    smoker = rng.binomial(1, smoking_rate, n)
    visits_last_year = np.clip(rng.poisson(visit_rate, n), 0, 15)

    length_of_stay = np.clip(
        rng.normal(
            2 + (comorbidities * 0.8) + (visits_last_year * 0.25),
            1.5,
            n,
        ),
        1,
        21,
    ).round().astype(int)

    risk_score = (
        -2.3
        + 0.03 * (age - 50)
        + 0.035 * (bmi - 25)
        + 0.012 * (systolic_bp - 120)
        + 0.010 * (glucose - 100)
        + 0.45 * smoker
        + 0.28 * comorbidities
        + 0.15 * visits_last_year
        + 0.08 * length_of_stay
        + risk_shift
    )

    readmission_probability = sigmoid(risk_score)
    readmitted_30d = rng.binomial(1, readmission_probability)

    return pd.DataFrame(
        {
            "record_id": [
                f"{prefix}-{number:04d}" for number in range(1, n + 1)
            ],
            "entity_id": hospital_name,
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "glucose": glucose,
            "cholesterol": cholesterol,
            "smoker": smoker,
            "comorbidity_count": comorbidities,
            "visits_last_year": visits_last_year,
            "length_of_stay": length_of_stay,
            "readmitted_30d": readmitted_30d,
        }
    )


def validate_dataset(data):
    assert not data.isnull().any().any()
    assert data["record_id"].is_unique
    assert data["age"].between(18, 90).all()
    assert data["bmi"].between(16, 50).all()
    assert set(data["readmitted_30d"].unique()).issubset({0, 1})


def main():
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    hospital_a = generate_hospital(
        hospital_name="Hospital A",
        prefix="HA",
        age_mean=45,
        comorbidity_rate=1.2,
        visit_rate=2.0,
        smoking_rate=0.18,
        risk_shift=-0.1,
        rng=rng,
    )

    hospital_b = generate_hospital(
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

    print("Synthetic datasets generated successfully.")


if __name__ == "__main__":
    main()
