"""
train_model.py
----------------
Trains three classification models for credit scoring:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest

Each trained model is saved to the /models folder using joblib.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def get_models():
    """Returns a dict of model_name -> untrained sklearn model."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        ),
    }
    return models


def train_models(X_train, y_train, save_dir="models"):
    """
    Trains all models on the training set and saves them to disk.
    Returns dict of model_name -> trained model.
    """
    os.makedirs(save_dir, exist_ok=True)
    models = get_models()
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        model_path = os.path.join(save_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"  -> saved to {model_path}")

    return trained_models
