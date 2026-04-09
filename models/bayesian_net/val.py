import os
import sys
import pickle

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.bayesian_net.train import (  # noqa: E402
    load_and_split, run_inference, fit_model, _oversample_df,
)

# Dirichlet pseudo-count candidates to search over.
# Lower values → weaker smoothing (sharper CPTs).
# Higher values → stronger smoothing (flatter CPTs, more regularized).
PSEUDO_COUNTS_GRID = [0.5, 1, 2, 5, 10]


def main():
    FILEPATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hfacs_bn.pkl")
    OUT_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "results", "bn_val_metrics.csv")

    df_train, df_val, df_test, emp_bins = load_and_split(FILEPATH)
    df_train_balanced = _oversample_df(
        df_train, ["Supervisory", "Operator", "UnsafeActs"], random_state=42
    )

    print(f"Train (balanced): {len(df_train_balanced)}  Val: {len(df_val)}\n")

    # ── Hyperparameter search over pseudo_counts ──────────────────────────────
    print("Searching pseudo_counts:", PSEUDO_COUNTS_GRID)
    print(f"{'pseudo_counts':>14}  {'Bal-Acc A':>10}  {'Bal-Acc B':>10}  {'Bal-Acc C':>10}  {'Avg':>8}")
    print("-" * 60)

    best_avg   = -1.0
    best_pc    = None
    search_rows = []

    for pc in PSEUDO_COUNTS_GRID:
        model = fit_model(df_train_balanced, pseudo_counts=pc)
        tA, tB, tC, pA, pB, pC = run_inference(model, df_val)

        bA  = balanced_accuracy_score(tA, pA)
        bB  = balanced_accuracy_score(tB, pB)
        bC  = balanced_accuracy_score(tC, pC)
        avg = (bA + bB + bC) / 3

        print(f"{pc:>14}  {bA:>10.2%}  {bB:>10.2%}  {bC:>10.2%}  {avg:>8.2%}")
        search_rows.append({"pseudo_counts": pc, "bal_acc_A": bA, "bal_acc_B": bB,
                            "bal_acc_C": bC, "bal_acc_avg": avg})

        if avg > best_avg:
            best_avg = avg
            best_pc  = pc

    print(f"\nBest pseudo_counts = {best_pc}  (val avg balanced_accuracy = {best_avg:.2%})")

    # ── Refit with best pseudo_counts and save ────────────────────────────────
    print(f"\nRefitting with pseudo_counts={best_pc}...")
    best_model = fit_model(df_train_balanced, pseudo_counts=best_pc)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "emp_bins": emp_bins}, f)
    print(f"Best model saved to {MODEL_PATH}")

    # ── Final val metrics with best model ─────────────────────────────────────
    true_A, true_B, true_C, pred_A, pred_B, pred_C = run_inference(best_model, df_val)

    bal_A = balanced_accuracy_score(true_A, pred_A)
    bal_B = balanced_accuracy_score(true_B, pred_B)
    bal_C = balanced_accuracy_score(true_C, pred_C)
    avg   = (bal_A + bal_B + bal_C) / 3

    print(f"\nValidation samples: {len(df_val)}")
    print(f"Balanced Acc A (Supervisory): {bal_A:.2%}")
    print(f"Balanced Acc B (Operator)   : {bal_B:.2%}")
    print(f"Balanced Acc C (Unsafe Acts): {bal_C:.2%}")
    print(f"Average                     : {avg:.2%}")
    print()
    print("── A: Supervisory Conditions ──")
    print(classification_report(true_A, pred_A, zero_division=0))
    print("── B: Operator Conditions ──")
    print(classification_report(true_B, pred_B, zero_division=0))
    print("── C: Unsafe Acts ──")
    print(classification_report(true_C, pred_C, zero_division=0))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pd.DataFrame([{
        "best_pseudo_counts":  best_pc,
        "bal_acc_supervisory": bal_A,
        "bal_acc_operator":    bal_B,
        "bal_acc_unsafe_acts": bal_C,
        "bal_acc_avg":         avg,
        "n_samples":           len(df_val),
    }]).to_csv(OUT_PATH, index=False)
    print(f"Val metrics saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
