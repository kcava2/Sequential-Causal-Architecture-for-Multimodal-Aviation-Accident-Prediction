import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.bayesian_net.train import (  # noqa: E402
    load_and_split, run_inference, fit_model, _oversample_df,
)

# --- 1. Scikit-Learn Wrapper for your BN Model ---
class BayesianNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, pseudo_counts=1.0):
        self.pseudo_counts = pseudo_counts
        self.model_ = None

    def fit(self, X, y=None):
        # We assume X is the dataframe containing both features and targets
        # as required by your fit_model function
        self.model_ = fit_model(X, pseudo_counts=self.pseudo_counts)
        return self

    def predict(self, X):
        # run_inference returns (true_A, true_B, true_C, pred_A, pred_B, pred_C)
        # We only need the predictions for the scorer
        _, _, _, pA, pB, pC = run_inference(self.model_, X)
        return np.column_stack([pA, pB, pC])

# --- 2. Custom Scorer for Multi-Output Balanced Accuracy ---
def custom_multioutput_scorer(y_true, y_pred):
    """
    y_true: Array-like of shape (n_samples, 3) 
    y_pred: Array-like of shape (n_samples, 3) from Wrapper.predict()
    """
    # If y_true is a DataFrame, convert to numpy
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
        
    bA = balanced_accuracy_score(y_true[:, 0], y_pred[:, 0])
    bB = balanced_accuracy_score(y_true[:, 1], y_pred[:, 1])
    bC = balanced_accuracy_score(y_true[:, 2], y_pred[:, 2])
    return (bA + bB + bC) / 3

def main():
    FILEPATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hfacs_bn.pkl")
    OUT_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "results", "bn_val_metrics.csv")

    df_train, df_val, df_test, emp_bins = load_and_split(FILEPATH)
    
    # Balance the training set
    df_train_balanced = _oversample_df(
        df_train, ["Supervisory", "Operator", "UnsafeActs"], random_state=42
    )

    # Prepare targets for the scorer (must match order in Wrapper.predict)
    target_cols = ["Supervisory", "Operator", "UnsafeActs"]
    y_train = df_train_balanced[target_cols]

    # --- 3. Grid Search Configuration ---
    param_grid = {'pseudo_counts': [0.5, 1, 2, 5, 10]}
    
    # We use cv=5 for Cross-Validation
    grid_search = GridSearchCV(
        estimator=BayesianNetWrapper(),
        param_grid=param_grid,
        scoring=make_scorer(custom_multioutput_scorer),
        cv=5, 
        verbose=2,
        n_jobs=-1 # Set to 1 if your BN library doesn't support multiprocessing
    )

    print("Starting GridSearchCV...")
    grid_search.fit(df_train_balanced, y_train)

    best_pc = grid_search.best_params_['pseudo_counts']
    best_avg = grid_search.best_score_

    print(f"\nBest pseudo_counts = {best_pc} (CV Avg Bal-Acc = {best_avg:.2%})")

    # --- 4. Final Evaluation on Holdout Validation Set ---
    # GridSearchCV automatically refits the model on the whole training set using best_pc
    best_model_wrapper = grid_search.best_estimator_
    best_model = best_model_wrapper.model_

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "emp_bins": emp_bins}, f)
    print(f"Best model saved to {MODEL_PATH}")

    # Final metrics using your original inference logic
    true_A, true_B, true_C, pred_A, pred_B, pred_C = run_inference(best_model, df_val)

    bal_A = balanced_accuracy_score(true_A, pred_A)
    bal_B = balanced_accuracy_score(true_B, pred_B)
    bal_C = balanced_accuracy_score(true_C, pred_C)
    avg = (bal_A + bal_B + bal_C) / 3

    print(f"\nFinal Validation Results (pseudo_counts={best_pc}):")
    print(f"Avg Balanced Acc: {avg:.2%}")
    print("\n── A: Supervisory ──\n", classification_report(true_A, pred_A, zero_division=0))
    print("\n── B: Operator ──\n", classification_report(true_B, pred_B, zero_division=0))
    print("\n── C: Unsafe Acts ──\n", classification_report(true_C, pred_C, zero_division=0))

    # Save results
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pd.DataFrame([{
        "best_pseudo_counts":  best_pc,
        "bal_acc_avg":         avg,
        "n_samples":           len(df_val),
    }]).to_csv(OUT_PATH, index=False)

if __name__ == "__main__":
    main()