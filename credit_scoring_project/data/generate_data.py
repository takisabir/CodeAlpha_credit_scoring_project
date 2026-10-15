"""
generate_data.py
-----------------
Synthetic financial dataset generator for the Credit Scoring project.

Since we don't have access to a real bank dataset (real credit data is
private/regulated), this script creates a REALISTIC synthetic dataset
with the kind of features a real credit scoring model would use:

    - income                  (monthly income in INR)
    - age
    - employment_years        (years at current job)
    - existing_debts          (total outstanding debt in INR)
    - debt_to_income_ratio    (engineered feature)
    - num_credit_accounts     (number of open credit lines)
    - credit_utilization      (% of available credit being used)
    - late_payments_last_year (count of late/missed payments)
    - loan_amount_requested
    - credit_history_years

Target:
    - creditworthy (1 = good credit risk / loan approved, 0 = bad risk / default likely)

Run:
    python data/generate_data.py
Produces:
    data/credit_data.csv
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_SAMPLES = 5000

def generate_dataset(n_samples=N_SAMPLES, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    # --- Base financial features ---
    age = rng.integers(21, 65, n_samples)
    income = rng.normal(55000, 20000, n_samples).clip(10000, 250000)          # monthly income
    employment_years = rng.integers(0, 35, n_samples)
    existing_debts = rng.normal(150000, 90000, n_samples).clip(0, 800000)
    num_credit_accounts = rng.integers(0, 12, n_samples)
    credit_utilization = rng.uniform(0, 1, n_samples)                          # 0 to 1 (fraction)
    late_payments_last_year = rng.poisson(0.8, n_samples).clip(0, 12)
    loan_amount_requested = rng.normal(300000, 150000, n_samples).clip(20000, 1500000)
    credit_history_years = rng.integers(0, 30, n_samples)

    # --- Engineered feature ---
    debt_to_income_ratio = existing_debts / (income * 12 + 1)

    # --- Build a "credit risk score" using domain logic + noise ---
    # Higher income, more employment years, longer credit history,
    # lower utilization, fewer late payments -> more creditworthy.
    score = (
        0.30 * (income / income.max())
        + 0.15 * (employment_years / employment_years.max())
        + 0.15 * (credit_history_years / credit_history_years.max())
        - 0.20 * credit_utilization
        - 0.20 * (late_payments_last_year / (late_payments_last_year.max() + 1))
        - 0.15 * debt_to_income_ratio
        - 0.10 * (loan_amount_requested / loan_amount_requested.max())
        + rng.normal(0, 0.08, n_samples)   # noise so it's not a trivial separable problem
    )

    # Convert score to probability via sigmoid, then sample binary label
    prob_creditworthy = 1 / (1 + np.exp(-10 * (score - score.mean())))
    creditworthy = rng.binomial(1, prob_creditworthy)

    df = pd.DataFrame({
        "age": age,
        "income": income.round(2),
        "employment_years": employment_years,
        "existing_debts": existing_debts.round(2),
        "debt_to_income_ratio": debt_to_income_ratio.round(4),
        "num_credit_accounts": num_credit_accounts,
        "credit_utilization": credit_utilization.round(4),
        "late_payments_last_year": late_payments_last_year,
        "loan_amount_requested": loan_amount_requested.round(2),
        "credit_history_years": credit_history_years,
        "creditworthy": creditworthy
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "credit_data.csv")
    df.to_csv(out_path, index=False)

    print(f"✅ Synthetic dataset generated: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Target balance:\n{df['creditworthy'].value_counts(normalize=True)}")
    print("\nFirst 5 rows:\n", df.head())
