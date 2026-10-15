"""
preprocessing.py
------------------
Loads the raw credit dataset, does feature engineering, handles
scaling, and splits into train/test sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path="data/credit_data.csv"):
    df = pd.read_csv(path)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a couple of extra engineered features on top of the raw ones.
    (debt_to_income_ratio already exists from generation, but we add
    two more here so students can see how feature engineering works.)
    """
    df = df.copy()

    # Loan amount relative to income -> how "big" is the ask vs earning power
    df["loan_to_income_ratio"] = df["loan_amount_requested"] / (df["income"] * 12 + 1)

    # Simple risk flag: many late payments AND high utilization
    df["high_risk_flag"] = (
        (df["late_payments_last_year"] >= 2) & (df["credit_utilization"] > 0.6)
    ).astype(int)

    return df


def split_and_scale(df: pd.DataFrame, target_col="creditworthy", test_size=0.2, random_state=42):
    """
    Splits data into train/test and scales numeric features using StandardScaler.
    Returns X_train, X_test, y_train, y_test, scaler, feature_names
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def prepare_data(path="data/credit_data.csv"):
    """
    Convenience wrapper: load -> feature engineer -> split -> scale
    """
    df = load_data(path)
    df = feature_engineering(df)
    return split_and_scale(df)
