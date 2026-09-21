"""
Four-model classification comparison for the Airline Passenger Satisfaction dataset.
Corresponds to Report Section 4 (ML Model Implementation and Comparison).
"""
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

MODELS = {
    "Logistic Regression": LogisticRegression(penalty="l2", C=1.0, max_iter=1000, solver="lbfgs"),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                              eval_metric="logloss", n_jobs=-1, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1,
                                n_jobs=-1, random_state=42),
}


def run_comparison(df: pd.DataFrame) -> pd.DataFrame:
    y = (df["satisfaction"] == "satisfied").astype(int)
    X = df.drop(columns=["satisfaction"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

    results = []
    for name, model in MODELS.items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "Training Time (s)": round(elapsed, 2),
        })
    return pd.DataFrame(results).set_index("Model")


if __name__ == "__main__":
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _data_dir = os.path.join(_base_dir, "..", "data")
    data = pd.read_csv(os.path.join(_data_dir, "airline_passenger_satisfaction_processed.csv"))
    results_df = run_comparison(data)
    print(results_df.round(4))
    results_df.to_csv(os.path.join(_data_dir, "model_comparison_results.csv"))
