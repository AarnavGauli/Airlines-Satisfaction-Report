"""
Preprocessing pipeline for the Airline Passenger Satisfaction dataset.
Corresponds to Report Section 2 (Data Pre-Processing & Feature Engineering).

Two stages are exposed, matching how the report uses the data in different sections:
  - load_and_clean():   imputation, outlier capping, feature engineering.
                        Categorical columns stay as human-readable labels and delay/
                        distance stay in original units -> used for EDA (Section 3).
  - encode_and_scale(): one-hot / ordinal encoding + RobustScaler on top of the clean
                        frame -> used for model training (Section 4).
"""
import os
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "..", "data")

SERVICE_COLS = [
    "Inflight wifi service", "Food and drink", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
]


def iqr_cap(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower=max(lower, 0), upper=upper)


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Sections 2.1, 2.2, 2.5 - imputation, outlier capping, feature engineering.
    Keeps Gender / Customer Type / Type of Travel / Class as readable strings so the
    output is directly plottable (Report Section 3)."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 0", "id"], errors="ignore")

    # 2.1 Missing value imputation (KNN, exploits departure/arrival delay correlation)
    impute_cols = ["Departure Delay in Minutes", "Arrival Delay in Minutes", "Flight Distance"]
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    df[impute_cols] = imputer.fit_transform(df[impute_cols])

    # 2.2 Outlier capping (IQR winsorization)
    for col in ["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
        df[col] = iqr_cap(df[col])

    # 2.5 Feature engineering
    df["Total_Delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["Overall_Service_Score"] = df[SERVICE_COLS].mean(axis=1)

    return df


def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Sections 2.3, 2.4 - categorical encoding + feature scaling, applied on top of
    the cleaned frame. Produces the model-ready dataset used in Report Section 4."""
    df = df.copy()

    nominal_cols = ["Gender", "Customer Type", "Type of Travel"]
    ohe = OneHotEncoder(drop="first", sparse_output=False)
    ohe_array = ohe.fit_transform(df[nominal_cols])
    ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(nominal_cols), index=df.index)

    class_order = [["Eco", "Eco Plus", "Business"]]
    oe = OrdinalEncoder(categories=class_order)
    df["Class_encoded"] = oe.fit_transform(df[["Class"]])

    df = pd.concat([df.drop(columns=nominal_cols + ["Class"]), ohe_df], axis=1)

    scale_cols = ["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    scaler = RobustScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """Convenience wrapper: clean + encode in one call (used by src/models.py)."""
    return encode_and_scale(load_and_clean(csv_path))


def check_class_balance(df: pd.DataFrame) -> pd.Series:
    """Report Section 2.6 - verifies whether stratified splitting is required."""
    return df["satisfaction"].value_counts(normalize=True)


if __name__ == "__main__":
    clean = load_and_clean(os.path.join(DATA_DIR, "airline_passenger_satisfaction.csv"))
    print(check_class_balance(clean))
    clean.to_csv(os.path.join(DATA_DIR, "airline_passenger_satisfaction_clean.csv"), index=False)

    encoded = encode_and_scale(clean)
    encoded.to_csv(os.path.join(DATA_DIR, "airline_passenger_satisfaction_processed.csv"), index=False)
