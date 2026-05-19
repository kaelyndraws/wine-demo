"""Build sklearn pipelines for each candidate model.

Each pipeline has the same shape: StandardScaler -> classifier except xgb.
This makes them interchangeable in training, evaluation, and inference.

Standardization is essential for Logistic Regression and KNN (both are
distance/scale sensitive). XGBoost is a tree-based model and does not require standardization, so we can skip it for that pipeline.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_pipelines() -> dict[str, Pipeline]:
    """Return a dict of {name: pipeline} for all candidate models."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15)),
        ]),
        "xgboost": Pipeline([
            ("clf", XGBClassifier(random_state=42)),
        ]),
    }
