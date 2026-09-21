"""
Generates a statistically-calibrated SYNTHETIC placeholder dataset that reproduces the
distributional patterns described in the report (class/loyalty/delay effects on
satisfaction), for use when the real Kaggle CSV is unavailable in this environment.

To use the REAL dataset instead: download it from
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
and save it as data/airline_passenger_satisfaction.csv (same column names as below),
then skip this script entirely.

Run: python data/generate_placeholder_dataset.py
Produces: data/airline_passenger_satisfaction.csv (raw schema, ready for src/preprocessing.py)
"""
import numpy as np
import pandas as pd

OUT_CSV = "airline_passenger_satisfaction.csv"


def to_rating(x):
    return np.clip(np.round(x), 0, 5).astype(int)


def generate(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_type = rng.choice(["Loyal Customer", "disloyal Customer"], size=n, p=[0.82, 0.18])
    type_of_travel = rng.choice(["Business travel", "Personal Travel"], size=n, p=[0.69, 0.31])

    travel_class = np.empty(n, dtype=object)
    biz_mask = type_of_travel == "Business travel"
    travel_class[biz_mask] = rng.choice(["Eco", "Eco Plus", "Business"], size=biz_mask.sum(),
                                         p=[0.30, 0.08, 0.62])
    travel_class[~biz_mask] = rng.choice(["Eco", "Eco Plus", "Business"], size=(~biz_mask).sum(),
                                          p=[0.75, 0.15, 0.10])

    gender = rng.choice(["Male", "Female"], size=n)
    age = np.clip(rng.normal(40, 15, n), 7, 85).round().astype(int)

    class_level = np.select(
        [travel_class == "Business", travel_class == "Eco Plus", travel_class == "Eco"], [4.2, 3.0, 2.3])
    digital_latent = np.clip(class_level + rng.normal(0, 0.8, n), 0, 5)
    physical_latent = np.clip(class_level + rng.normal(0, 0.8, n), 0, 5)

    wifi = to_rating(digital_latent + rng.normal(0, 0.5, n))
    online_boarding = to_rating(digital_latent + rng.normal(0, 0.5, n))
    ease_booking = to_rating(digital_latent + rng.normal(0, 0.5, n))
    dep_time_conv = to_rating(digital_latent * 0.6 + rng.normal(2, 1, n))
    gate_location = to_rating(rng.normal(2.9, 1.0, n))

    seat_comfort = to_rating(physical_latent + rng.normal(0, 0.5, n))
    entertainment = to_rating(physical_latent + rng.normal(0, 0.5, n))
    cleanliness = to_rating(physical_latent + rng.normal(0, 0.5, n))
    food = to_rating(physical_latent * 0.8 + rng.normal(0.5, 0.7, n))
    onboard_service = to_rating(physical_latent * 0.9 + rng.normal(0.3, 0.6, n))
    legroom = to_rating(physical_latent * 0.8 + rng.normal(0.3, 0.7, n))
    baggage = to_rating(rng.normal(3.2, 1.0, n))
    checkin = to_rating(rng.normal(3.2, 1.0, n))
    inflight_service = to_rating(physical_latent * 0.9 + rng.normal(0.3, 0.6, n))

    flight_distance = np.clip(rng.gamma(2.0, 500, n), 31, 5000).astype(int)
    no_delay_mask = rng.random(n) < 0.35
    dep_delay = np.where(no_delay_mask, 0, np.clip(rng.exponential(15, n), 1, 1600)).astype(int)
    arr_delay = np.clip(dep_delay + rng.normal(0, 10, n), 0, 1600).astype(int)
    arr_delay = np.where(no_delay_mask, 0, arr_delay)
    total_delay = dep_delay + arr_delay

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    logit = -0.55
    logit += np.select([travel_class == "Business", travel_class == "Eco Plus", travel_class == "Eco"],
                        [1.2, -0.5, -1.0])
    logit += np.where(customer_type == "Loyal Customer", 0.5, -0.5)
    logit += -np.abs(age - 45) / 100
    logit += 0.15 * (digital_latent - 2.5) + 0.15 * (physical_latent - 2.5)
    logit += -0.32 * np.log1p(total_delay) / 3
    p_satisfied = sigmoid(logit)
    satisfaction = np.where(rng.random(n) < p_satisfied, "satisfied", "neutral or dissatisfied")

    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "Gender": gender, "Customer Type": customer_type, "Age": age,
        "Type of Travel": type_of_travel, "Class": travel_class,
        "Flight Distance": flight_distance,
        "Inflight wifi service": wifi, "Departure/Arrival time convenient": dep_time_conv,
        "Ease of Online booking": ease_booking, "Gate location": gate_location,
        "Food and drink": food, "Online boarding": online_boarding, "Seat comfort": seat_comfort,
        "Inflight entertainment": entertainment, "On-board service": onboard_service,
        "Leg room service": legroom, "Baggage handling": baggage, "Checkin service": checkin,
        "Inflight service": inflight_service, "Cleanliness": cleanliness,
        "Departure Delay in Minutes": dep_delay, "Arrival Delay in Minutes": arr_delay.astype(float),
        "satisfaction": satisfaction,
    })

    # Reintroduce ~0.3% missingness in Arrival Delay in Minutes to match the real dataset (Report Section 2.1)
    missing_idx = rng.choice(df.index, size=int(0.003 * n), replace=False)
    df.loc[missing_idx, "Arrival Delay in Minutes"] = np.nan

    return df


if __name__ == "__main__":
    dataset = generate()
    dataset.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(dataset)} rows to {OUT_CSV}")
    print(dataset["satisfaction"].value_counts(normalize=True).round(3))
