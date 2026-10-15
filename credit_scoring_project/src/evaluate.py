"""
evaluate.py
-------------
Evaluates trained models using:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
    - Confusion Matrix
    - ROC Curve plot (saved as PNG)

Prints a comparison table across all models at the end.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print(f"\n===== {name.upper()} =====")
    print(classification_report(y_test, y_pred, target_names=["Not Creditworthy", "Creditworthy"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    return metrics, y_pred, y_prob


def plot_confusion_matrix(name, y_test, y_pred, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Creditworthy", "Creditworthy"],
                yticklabels=["Not Creditworthy", "Creditworthy"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    path = os.path.join(save_dir, f"confusion_matrix_{name}.png")
    plt.savefig(path)
    plt.close()
    print(f"  -> confusion matrix saved to {path}")


def plot_roc_curves(results, y_test, save_dir="outputs"):
    """
    results: dict of model_name -> y_prob (predicted probabilities)
    Plots all ROC curves on one chart for easy comparison.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))

    for name, y_prob in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, "roc_curve_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"\n✅ ROC curve comparison saved to {path}")


def compare_models(all_metrics, save_dir="outputs"):
    """
    all_metrics: list of metric dicts (one per model)
    Prints and saves a comparison table.
    """
    df = pd.DataFrame(all_metrics).set_index("model").round(4)
    print("\n===== MODEL COMPARISON =====")
    print(df)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "model_comparison.csv")
    df.to_csv(path)
    print(f"\n✅ Comparison table saved to {path}")

    return df
