"""Load and split the UCI Wine Quality (red) dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "winequality-red.csv"
)
DATA_URL_FALLBACK = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

FEATURE_NAMES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

CLASS_NAMES = ["low", "medium", "high"]


def bin_quality(q: int) -> int:
    """Collapse 3-8 quality scores into low(0) / medium(1) / high(2)."""
    if q <= 5:
        return 0
    if q == 6:
        return 1
    return 2


def load_dataset() -> pd.DataFrame:
    """Download the dataset. Tries the GitHub mirror first, then UCI."""
    try:
        df = pd.read_csv(DATA_URL, header=None, names=FEATURE_NAMES + ["quality"])
    except Exception as e:
        print(f"Primary mirror failed ({e}); trying UCI fallback.")
        df = pd.read_csv(DATA_URL_FALLBACK, sep=";")
        df.columns = FEATURE_NAMES + ["quality"]
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split into stratified train/test sets. Returns (X_train, X_test, y_train, y_test)."""
    X = df[FEATURE_NAMES]
    y = df["quality"].apply(bin_quality)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
