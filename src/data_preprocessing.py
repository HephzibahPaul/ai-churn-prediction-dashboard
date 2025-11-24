from pathlib import Path
from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(
    csv_path: str | Path = Path("data/raw_customers.csv"),
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. "
                                f"Run data/generate_synthetic_data.py first.")

    df = pd.read_csv(csv_path)

    # Separate features and target
    target_col = "Churn"
    y = df[target_col]
    X = df.drop(columns=[target_col, "CustomerID"], errors="ignore")

    # One-hot encode categorical columns
    categorical_cols = ["Gender", "Geography"]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_cols
