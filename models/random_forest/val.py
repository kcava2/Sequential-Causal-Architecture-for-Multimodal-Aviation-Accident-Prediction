import os
import sys
import pickle

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_predict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.random_forest.train import (  # noqa: E402
    load_and_split, build_augmented_df, make_rf_pipeline, fit_cascade,
    TARGET_A, TARGET_B, TARGET_C, RAND_STATE,
)

# Hyperparameter search space 
PARAM_GRID = {
    "rf__n_estimators":      [50, 100, 200],
    "rf__criterion":         ["gini", "entropy"],
    "rf__max_features":      ["sqrt", "log2", None],
    "rf__min_samples_leaf":  [2, 4, 8],
    "rf__min_samples_split": [2, 8, 16],
}


def _predict_cascade(model_1, model_2, model_3, df):
    """Cascading predictions on a DataFrame; returns (pred_A, pred_B, pred_C)."""
    input_1 = df[["Organizational Climate", "Employment, Total Weighted Avg CY_QoQ_pct"]]
    pred_A  = model_1.predict(input_1)

    input_2 = df[["WeatherCondition", "TimeOfDay", "SkyCondNonceil", "Personnel Conditions"]].copy()
    input_2["Supervisory Conditions"] = pred_A
    pred_B  = model_2.predict(input_2)

    input_3 = pd.DataFrame({
        "Supervisory Conditions": pred_A,
        "Operator Conditions":    pred_B,
    })
    pred_C  = model_3.predict(input_3)

    return pred_A, pred_B, pred_C


def _val_avg(model_1, model_2, model_3, df_val):
    """Return (avg_bal_acc, bal_A, bal_B, bal_C) on the val split."""
    pred_A, pred_B, pred_C = _predict_cascade(model_1, model_2, model_3, df_val)
    bA = balanced_accuracy_score(df_val[TARGET_A], pred_A)
    bB = balanced_accuracy_score(df_val[TARGET_B], pred_B)
    bC = balanced_accuracy_score(df_val[TARGET_C], pred_C)
    return (bA + bB + bC) / 3, bA, bB, bC


def main():
    FILEPATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hfacs_rf.pkl")
    OUT_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "results", "rf_val_metrics.csv")

    df_train, df_val, _ = load_and_split(FILEPATH)
    df_aug = build_augmented_df(df_train)

    print(f"Train (augmented): {len(df_aug)}  Val: {len(df_val)}\n")

    # Build inputs for each stage (OOF preds feed cascading stages)
    input_1 = df_aug[["Organizational Climate", "Employment, Total Weighted Avg CY_QoQ_pct"]]
    y_A     = df_aug[TARGET_A]
    input_2_base = df_aug[["WeatherCondition", "TimeOfDay", "SkyCondNonceil",
                            "Personnel Conditions"]].copy()
    y_B     = df_aug[TARGET_B]
    y_C     = df_aug[TARGET_C]

    # ── GridSearchCV: Model 1 ─────────────────────────────────────────────── #
    print("Model 1 (Supervisory) — GridSearchCV...")
    gs_1 = GridSearchCV(
        make_rf_pipeline(input_1), PARAM_GRID,
        cv=5, scoring="balanced_accuracy", n_jobs=-1, refit=True, verbose=0,
    )
    gs_1.fit(input_1, y_A)
    model_1 = gs_1.best_estimator_
    print(f"  Best params : {gs_1.best_params_}")
    print(f"  CV bal-acc  : {gs_1.best_score_:.4f}")

    A_pred_oof = cross_val_predict(model_1, input_1, y_A, cv=5, method="predict")

    # ── GridSearchCV: Model 2 ─────────────────────────────────────────────── #
    input_2 = input_2_base.copy()
    input_2["Supervisory Conditions"] = A_pred_oof

    print("\nModel 2 (Operator) — GridSearchCV...")
    gs_2 = GridSearchCV(
        make_rf_pipeline(input_2), PARAM_GRID,
        cv=5, scoring="balanced_accuracy", n_jobs=-1, refit=True, verbose=0,
    )
    gs_2.fit(input_2, y_B)
    model_2 = gs_2.best_estimator_
    print(f"  Best params : {gs_2.best_params_}")
    print(f"  CV bal-acc  : {gs_2.best_score_:.4f}")

    B_pred_oof = cross_val_predict(model_2, input_2, y_B, cv=5, method="predict")

    # ── GridSearchCV: Model 3 ─────────────────────────────────────────────── #
    input_3 = pd.DataFrame({
        "Supervisory Conditions": A_pred_oof,
        "Operator Conditions":    B_pred_oof,
    })

    print("\nModel 3 (Unsafe Acts) — GridSearchCV...")
    gs_3 = GridSearchCV(
        make_rf_pipeline(input_3), PARAM_GRID,
        cv=5, scoring="balanced_accuracy", n_jobs=-1, refit=True, verbose=0,
    )
    gs_3.fit(input_3, y_C)
    model_3 = gs_3.best_estimator_
    print(f"  Best params : {gs_3.best_params_}")
    print(f"  CV bal-acc  : {gs_3.best_score_:.4f}")

    # ── Evaluate best models on val set ──────────────────────────────────── #
    avg, bal_A, bal_B, bal_C = _val_avg(model_1, model_2, model_3, df_val)
    print(f"\nVal set — Supervisory: {bal_A:.2%}  Operator: {bal_B:.2%}  "
          f"Unsafe Acts: {bal_C:.2%}  Avg: {avg:.2%}")

    # ── Save best models (overwrite pkl) ─────────────────────────────────── #
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model_1": model_1, "model_2": model_2, "model_3": model_3}, f)
    print(f"Best models saved to {MODEL_PATH}")

    # ── Full classification reports on val set ───────────────────────────── #
    pred_A, pred_B, pred_C = _predict_cascade(model_1, model_2, model_3, df_val)

    print()
    print("── A: Supervisory Conditions ──")
    print(classification_report(df_val[TARGET_A], pred_A, zero_division=0))
    print("── B: Operator Conditions ──")
    print(classification_report(df_val[TARGET_B], pred_B, zero_division=0))
    print("── C: Unsafe Acts ──")
    print(classification_report(df_val[TARGET_C], pred_C, zero_division=0))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pd.DataFrame([{
        "best_params_model_1":  str(gs_1.best_params_),
        "best_params_model_2":  str(gs_2.best_params_),
        "best_params_model_3":  str(gs_3.best_params_),
        "bal_acc_supervisory":  bal_A,
        "bal_acc_operator":     bal_B,
        "bal_acc_unsafe_acts":  bal_C,
        "bal_acc_avg":          avg,
        "n_samples":            len(df_val),
    }]).to_csv(OUT_PATH, index=False)
    print(f"Val metrics saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
