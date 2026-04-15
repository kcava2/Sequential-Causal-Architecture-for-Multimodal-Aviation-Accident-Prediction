import os
import sys
import pickle
import warnings

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score, f1_score, cohen_kappa_score,
)

# Add paths for local modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.random_forest.train import (  # noqa: E402
    load_and_split, FEAT_COLS_1, FEAT_COLS_2, FEAT_COLS_3,
    TARGET_A, TARGET_B, TARGET_C, build_augmented_df
)

# Adjust if your eval_utils is in a different directory
from models.eval_utils import (  # noqa: E402
    TASK_COLORS, clean_feature_name, apply_plot_style,
    CONFUSION_FIGSIZE, plot_roc_curves, plot_confusion_matrices,
)

def _predict_cascade(model_1, model_2, model_3, df):
    """Run cascading predictions using optimized hyperparameters (no thresholding)."""
    # Stage 1: Supervisory
    input_1 = df[FEAT_COLS_1]
    pred_A = model_1.predict(input_1)

    # Stage 2: Operator
    input_2 = df[["WeatherCondition", "TimeOfDay", "SkyCondNonceil", "Personnel Conditions"]].copy()
    input_2["Supervisory Conditions"] = pred_A
    pred_B = model_2.predict(input_2)

    # Stage 3: Unsafe Conditions
    input_3 = pd.DataFrame({
        "Supervisory Conditions": pred_A,
        "Operator Conditions":     pred_B,
    })
    pred_C = model_3.predict(input_3)

    return pred_A, pred_B, pred_C

def _get_feature_groups(pipeline):
    """
    Returns (original_names, group_indices) where group_indices[i] is the list
    of column indices in the encoded matrix that belong to original feature i.
    Numerical features map to a single index; OHE categorical features map to
    all the columns produced for that category.
    """
    ct = pipeline.named_steps["encoder"]
    original_names = []
    group_indices  = []
    col_idx = 0
    for name, transformer, cols in ct.transformers_:
        if name == "num":
            for c in cols:
                original_names.append(clean_feature_name(c))
                group_indices.append([col_idx])
                col_idx += 1
        elif name == "cat":
            for orig_col in cols:
                n_cats = len(transformer.categories_[list(cols).index(orig_col)])
                original_names.append(clean_feature_name(orig_col))
                group_indices.append(list(range(col_idx, col_idx + n_cats)))
                col_idx += n_cats
    return original_names, group_indices


def _aggregate_shap(shap_values, data, group_indices):
    """
    Sum SHAP values across OHE columns that belong to the same original feature.
    data values are averaged across the group (mean-encoded representation).
    Returns (aggregated_shap, aggregated_data) — shape (n_samples, n_original_features).
    """
    agg_shap = np.column_stack([shap_values[:, idx].sum(axis=1) for idx in group_indices])
    agg_data = np.column_stack([data[:, idx].mean(axis=1) for idx in group_indices])
    return agg_shap, agg_data

def main():
    FILEPATH    = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH  = os.path.join(os.path.dirname(__file__), "hfacs_rf.pkl")
    FIG_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load the optimized models saved by GridSearchCV
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    m1, m2, m3 = saved["model_1"], saved["model_2"], saved["model_3"]

    df_train, df_val, df_test = load_and_split(FILEPATH)
    df_train_aug = build_augmented_df(df_train)

    # ── 1. Generalization Gap (Verify Hyperparameter Fit) ───────────────────
    print("Evaluating Generalization Gap...")
    tr_A, tr_B, tr_C = _predict_cascade(m1, m2, m3, df_train_aug)
    ts_A, ts_B, ts_C = _predict_cascade(m1, m2, m3, df_test)

    metrics = []
    for lbl, tr_p, ts_p, tr_y, ts_y in [
        ("Supervisory", tr_A, ts_A, df_train_aug[TARGET_A], df_test[TARGET_A]),
        ("Operator",    tr_B, ts_B, df_train_aug[TARGET_B], df_test[TARGET_B]),
        ("Unsafe Cond", tr_C, ts_C, df_train_aug[TARGET_C], df_test[TARGET_C]),
    ]:
        tr_acc = balanced_accuracy_score(tr_y, tr_p)
        ts_acc = balanced_accuracy_score(ts_y, ts_p)
        metrics.append({"Task": lbl, "Train_Acc": tr_acc, "Test_Acc": ts_acc, "Gap": tr_acc - ts_acc})

    gap_df = pd.DataFrame(metrics)
    print("\n" + gap_df.to_string(index=False))
    gap_df.to_csv(os.path.join(RESULTS_DIR, "rf_generalization_gap.csv"), index=False)

    # ── 2. Classification Reports ───────────────────────────────────────────
    print("\nClassification Reports (Test Set):")
    for lbl, true, pred in [("Supervisory", df_test[TARGET_A], ts_A), 
                            ("Operator",    df_test[TARGET_B], ts_B), 
                            ("Unsafe Cond", df_test[TARGET_C], ts_C)]:
        print(f"\n── {lbl} ──")
        print(classification_report(true, pred, zero_division=0))

    # ── 3. Confusion Matrices ───────────────────────────────────────────────
    print("\nPlotting Confusion Matrices...")
    cm_data = [
        ("Supervisory", df_test[TARGET_A].tolist(), ts_A.tolist(), sorted(df_test[TARGET_A].unique())),
        ("Operator",    df_test[TARGET_B].tolist(), ts_B.tolist(), sorted(df_test[TARGET_B].unique())),
        ("Unsafe Acts", df_test[TARGET_C].tolist(), ts_C.tolist(), sorted(df_test[TARGET_C].unique())),
    ]
    plot_confusion_matrices(cm_data, "RF", os.path.join(FIG_DIR, "rf_confusion_matrices.png"))

    # ── 4. Fixed SHAP Analysis (All Models) ──────────────────────────────────
    print("\nComputing SHAP explanations (Fixing Dimensionality)...")
    
    shap_config = [
        ("Supervisory", m1, df_test[FEAT_COLS_1]),
        ("Operator",    m2, df_test[["WeatherCondition", "TimeOfDay", "SkyCondNonceil", "Personnel Conditions"]].assign(**{"Supervisory Conditions": ts_A})),
        ("Unsafe_Cond", m3, pd.DataFrame({"Supervisory Conditions": ts_A, "Operator Conditions": ts_B}))
    ]

    for label, pipeline, data in shap_config:
        rf      = pipeline.named_steps["rf"]
        encoder = pipeline.named_steps["encoder"]

        # Build feature groups: maps each original feature → its OHE column indices
        orig_names, group_indices = _get_feature_groups(pipeline)

        # Transform data to encoded space
        X_tx = encoder.transform(data)
        if hasattr(X_tx, "toarray"):
            X_tx = X_tx.toarray()

        explainer = shap.TreeExplainer(rf)
        exp = explainer(X_tx, check_additivity=False)

        # Slice to class 0 → shape (n_samples, n_encoded_features)
        if len(exp.shape) == 3:
            sv   = exp.values[:, :, 0]
            data_arr = exp.data[:, :]
        else:
            sv   = exp.values
            data_arr = exp.data

        # Aggregate OHE columns back to original features
        sv_agg, data_agg = _aggregate_shap(sv, data_arr, group_indices)

        base_val = float(np.mean(exp.base_values[:, 0]) if exp.base_values.ndim > 1
                         else np.mean(exp.base_values))
        exp_agg = shap.Explanation(
            values=sv_agg,
            base_values=base_val,
            data=data_agg,
            feature_names=orig_names,
        )

        plt.figure()
        shap.plots.beeswarm(exp_agg, max_display=len(orig_names), show=False)
        plt.title(f"SHAP Summary: {label}")
        plt.savefig(os.path.join(FIG_DIR, f"rf_shap_summary_{label}.png"), bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.plots.waterfall(exp_agg[0], max_display=len(orig_names), show=False)
        plt.title(f"SHAP Waterfall: {label} (Sample 0)")
        plt.savefig(os.path.join(FIG_DIR, f"rf_shap_waterfall_{label}.png"), bbox_inches="tight")
        plt.close()

    print(f"\nEvaluation finished. Saved results and plots to {RESULTS_DIR} and {FIG_DIR}")

if __name__ == "__main__":
    main()