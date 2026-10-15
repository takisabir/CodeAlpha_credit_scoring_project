"""
main.py
---------
Entry point for the Credit Scoring Model project.

This script runs the FULL pipeline end-to-end:
    1. Load / generate data
    2. Feature engineering + train-test split + scaling
    3. Train Logistic Regression, Decision Tree, Random Forest
    4. Evaluate each model (Accuracy, Precision, Recall, F1, ROC-AUC)
    5. Save confusion matrices, ROC curve comparison, and results table

Run this from the project root:
    python main.py
"""

import os
import sys

# Make sure src/ and data/ are importable regardless of where script is run from
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "data"))

from preprocessing import prepare_data
from train_model import train_models
from evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curves, compare_models


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "credit_data.csv")


def main():
    # Step 0: generate synthetic data if it doesn't exist yet
    if not os.path.exists(DATA_PATH):
        print("No dataset found — generating synthetic credit data first...")
        from generate_data import generate_dataset
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset created at {DATA_PATH}\n")

    # Step 1: Preprocess
    print("Step 1: Loading & preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(DATA_PATH)
    print(f"  Train size: {X_train.shape}, Test size: {X_test.shape}\n")

    # Step 2: Train models
    print("Step 2: Training models...")
    trained_models = train_models(X_train, y_train)

    # Step 3: Evaluate models
    print("\nStep 3: Evaluating models...")
    all_metrics = []
    roc_data = {}

    for name, model in trained_models.items():
        metrics, y_pred, y_prob = evaluate_model(name, model, X_test, y_test)
        all_metrics.append(metrics)
        roc_data[name] = y_prob
        plot_confusion_matrix(name, y_test, y_pred)

    # Step 4: Compare models + plot ROC curves
    plot_roc_curves(roc_data, y_test)
    comparison_df = compare_models(all_metrics)

    # Step 5: Highlight the best model by ROC-AUC
    best_model = comparison_df["roc_auc"].idxmax()
    print(f"\n🏆 Best performing model (by ROC-AUC): {best_model}")
    print("\n✅ Pipeline complete! Check the 'outputs/' folder for plots and results,")
    print("   and the 'models/' folder for saved trained models.")


if __name__ == "__main__":
    main()
