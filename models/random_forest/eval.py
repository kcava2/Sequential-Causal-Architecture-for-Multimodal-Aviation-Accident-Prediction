import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score, f1_score, cohen_kappa_score,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.random_forest.train import (  # noqa: E402
    load_and_split, FEAT_COLS_1, FEAT_COLS_2, FEAT_COLS_3,
    TARGET_A, TARGET_B, TARGET_C,
)


def _cascade_probs(model_1, model_2, model_3, df):
    """Run cascade and return (A_probs, A_classes, B_probs, B_classes, C_probs, C_classes, A_pred, B_pred)."""
    input_1 = df[FEAT_COLS_1]
    A_probs  = model_1.predict_proba(input_1)
    A_classes = model_1.classes_
    A_pred   = model_1.predict(input_1)

    input_2 = df[["WeatherCondition", "TimeOfDay", "SkyCondNonceil", "Personnel Conditions"]].copy()
    input_2["Supervisory Conditions"] = A_pred
    B_probs  = model_2.predict_proba(input_2)
    B_classes = model_2.classes_
    B_pred   = model_2.predict(input_2)

    input_3 = pd.DataFrame({
        "Supervisory Conditions": A_pred,
        "Operator Conditions":    B_pred,
    })
    C_probs  = model_3.predict_proba(input_3)
    C_classes = model_3.classes_

    return A_probs, A_classes, B_probs, B_classes, C_probs, C_classes, A_pred, B_pred


def _optimize_thresholds(probs, true, classes, n_grid=20):
    """
    Find per-class thresholds maximizing balanced accuracy on the given set.
    Uses ratio form: predicted = classes[argmax(prob[c] / threshold[c])].
    Returns a 1-D numpy array of thresholds, one per class.
    """
    n_classes = probs.shape[1]
    thresholds = np.ones(n_classes)
    best_acc = balanced_accuracy_score(true, classes[probs.argmax(axis=1)])

    grid = np.linspace(0.1, 2.0, n_grid)
    for c in range(n_classes):
        best_t = 1.0
        for t in grid:
            thresholds[c] = t
            preds = classes[(probs / thresholds).argmax(axis=1)]
            acc = balanced_accuracy_score(true, preds)
            if acc > best_acc:
                best_acc = acc
                best_t = t
        thresholds[c] = best_t

    return thresholds


def _predict_cascade(model_1, model_2, model_3, df):
    """Run cascading predictions on a dataframe; returns (A_pred, B_pred, C_pred)."""
    input_1 = df[FEAT_COLS_1]
    A_pred  = model_1.predict(input_1)

    input_2 = df[["WeatherCondition", "TimeOfDay", "SkyCondNonceil", "Personnel Conditions"]].copy()
    input_2["Supervisory Conditions"] = A_pred
    B_pred  = model_2.predict(input_2)

    input_3 = pd.DataFrame({
        "Supervisory Conditions": A_pred,
        "Operator Conditions":    B_pred,
    })
    C_pred  = model_3.predict(input_3)

    return A_pred, B_pred, C_pred


def main():
    FILEPATH    = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH  = os.path.join(os.path.dirname(__file__), "hfacs_rf.pkl")
    FIG_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model_1 = saved["model_1"]
    model_2 = saved["model_2"]
    model_3 = saved["model_3"]

    _, df_val, df_test = load_and_split(FILEPATH)

    print(f"Test samples: {len(df_test)}\n")

    true_A = df_test[TARGET_A].values
    true_B = df_test[TARGET_B].values
    true_C = df_test[TARGET_C].values

    pred_A, pred_B, pred_C = _predict_cascade(model_1, model_2, model_3, df_test)

    # ── Metrics ──────────────────────────────────────────────────────────────
    bal_A = balanced_accuracy_score(true_A, pred_A)
    bal_B = balanced_accuracy_score(true_B, pred_B)
    bal_C = balanced_accuracy_score(true_C, pred_C)

    f1_A = f1_score(true_A, pred_A, average="macro", zero_division=0)
    f1_B = f1_score(true_B, pred_B, average="macro", zero_division=0)
    f1_C = f1_score(true_C, pred_C, average="macro", zero_division=0)

    kappa_A = cohen_kappa_score(true_A, pred_A)
    kappa_B = cohen_kappa_score(true_B, pred_B)
    kappa_C = cohen_kappa_score(true_C, pred_C)

    print(f"{'Metric':<22} {'A (Supervisory)':>16} {'B (Operator)':>14} {'C (Unsafe Acts)':>16}")
    print("-" * 72)
    print(f"{'Balanced Accuracy':<22} {bal_A:>16.2%} {bal_B:>14.2%} {bal_C:>16.2%}")
    print(f"{'Macro F1':<22} {f1_A:>16.4f} {f1_B:>14.4f} {f1_C:>16.4f}")
    print(f"{'Cohen Kappa':<22} {kappa_A:>16.4f} {kappa_B:>14.4f} {kappa_C:>16.4f}")
    print()

    print("── A: Supervisory Conditions ──")
    print(classification_report(true_A, pred_A, zero_division=0))
    print("── B: Operator Conditions ──")
    print(classification_report(true_B, pred_B, zero_division=0))
    print("── C: Unsafe Acts ──")
    print(classification_report(true_C, pred_C, zero_division=0))

    # ── Per-class metrics CSV ─────────────────────────────────────────────────
    rows = []
    for task_label, true, pred, bal, f1, kappa in [
        ("Supervisory (A)", true_A, pred_A, bal_A, f1_A, kappa_A),
        ("Operator (B)",    true_B, pred_B, bal_B, f1_B, kappa_B),
        ("Unsafe Acts (C)", true_C, pred_C, bal_C, f1_C, kappa_C),
    ]:
        report = classification_report(true, pred, zero_division=0, output_dict=True)
        for cls_name, metrics in report.items():
            if isinstance(metrics, dict):
                rows.append({
                    "task": task_label,
                    "class": cls_name,
                    "balanced_accuracy": bal if cls_name == "macro avg" else None,
                    "macro_f1": f1 if cls_name == "macro avg" else None,
                    "cohen_kappa": kappa if cls_name == "macro avg" else None,
                    **metrics,
                })

    eval_csv = os.path.join(RESULTS_DIR, "rf_eval_metrics.csv")
    pd.DataFrame(rows).to_csv(eval_csv, index=False)
    print(f"Eval metrics saved to {eval_csv}")

    # ── Confusion matrices ────────────────────────────────────────────────────
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (true, pred, title) in zip(axes, [
        (true_A, pred_A, "A: Supervisory Conditions"),
        (true_B, pred_B, "B: Operator Conditions"),
        (true_C, pred_C, "C: Unsafe Acts"),
    ]):
        labels = sorted(set(true))
        cm = confusion_matrix(true, pred, labels=labels)
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(title)

    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, "rf_confusion_matrices.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrices saved to {cm_path}")
    plt.close()

    # ── Threshold optimization (fit on val, apply to test) ────────────────────
    print("\nOptimizing per-class decision thresholds on validation set...")
    vA_probs, A_cls, vB_probs, B_cls, vC_probs, C_cls, _, _ = _cascade_probs(
        model_1, model_2, model_3, df_val
    )
    val_true_A = df_val[TARGET_A].values
    val_true_B = df_val[TARGET_B].values
    val_true_C = df_val[TARGET_C].values

    thresh_A = _optimize_thresholds(vA_probs, val_true_A, A_cls)
    thresh_B = _optimize_thresholds(vB_probs, val_true_B, B_cls)
    thresh_C = _optimize_thresholds(vC_probs, val_true_C, C_cls)

    tA_probs, _, tB_probs, _, tC_probs, _, _, _ = _cascade_probs(
        model_1, model_2, model_3, df_test
    )
    th_pred_A = A_cls[(tA_probs / thresh_A).argmax(axis=1)]
    th_pred_B = B_cls[(tB_probs / thresh_B).argmax(axis=1)]
    th_pred_C = C_cls[(tC_probs / thresh_C).argmax(axis=1)]

    th_bal_A = balanced_accuracy_score(true_A, th_pred_A)
    th_bal_B = balanced_accuracy_score(true_B, th_pred_B)
    th_bal_C = balanced_accuracy_score(true_C, th_pred_C)
    th_f1_A  = f1_score(true_A, th_pred_A, average="macro", zero_division=0)
    th_f1_B  = f1_score(true_B, th_pred_B, average="macro", zero_division=0)
    th_f1_C  = f1_score(true_C, th_pred_C, average="macro", zero_division=0)
    th_kap_A = cohen_kappa_score(true_A, th_pred_A)
    th_kap_B = cohen_kappa_score(true_B, th_pred_B)
    th_kap_C = cohen_kappa_score(true_C, th_pred_C)

    print("\nThresholded Test Metrics:")
    print(f"{'Metric':<22} {'A (Supervisory)':>16} {'B (Operator)':>14} {'C (Unsafe Acts)':>16}")
    print("-" * 72)
    print(f"{'Balanced Accuracy':<22} {th_bal_A:>16.2%} {th_bal_B:>14.2%} {th_bal_C:>16.2%}")
    print(f"{'Macro F1':<22} {th_f1_A:>16.4f} {th_f1_B:>14.4f} {th_f1_C:>16.4f}")
    print(f"{'Cohen Kappa':<22} {th_kap_A:>16.4f} {th_kap_B:>14.4f} {th_kap_C:>16.4f}")

    th_rows = []
    for task_label, true, pred, bal, f1, kappa in [
        ("Supervisory (A)", true_A, th_pred_A, th_bal_A, th_f1_A, th_kap_A),
        ("Operator (B)",    true_B, th_pred_B, th_bal_B, th_f1_B, th_kap_B),
        ("Unsafe Acts (C)", true_C, th_pred_C, th_bal_C, th_f1_C, th_kap_C),
    ]:
        report = classification_report(true, pred, zero_division=0, output_dict=True)
        for cls_name, metrics in report.items():
            if isinstance(metrics, dict):
                th_rows.append({
                    "task": task_label,
                    "class": cls_name,
                    "balanced_accuracy": bal if cls_name == "macro avg" else None,
                    "macro_f1": f1 if cls_name == "macro avg" else None,
                    "cohen_kappa": kappa if cls_name == "macro avg" else None,
                    **metrics,
                })

    th_csv = os.path.join(RESULTS_DIR, "rf_eval_metrics_thresholded.csv")
    pd.DataFrame(th_rows).to_csv(th_csv, index=False)
    print(f"Thresholded eval metrics saved to {th_csv}")

    # ── Feature importance analysis ───────────────────────────────────────────
    print("\nExtracting feature importances...")

    def _get_feature_names(pipeline, input_df):
        """Recover feature names after ColumnTransformer encoding."""
        ct = pipeline.named_steps["encoder"]
        names = []
        for name, transformer, cols in ct.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                names.extend(transformer.get_feature_names_out(cols))
        return names

    imp_rows = []
    for model_label, model, feat_cols in [
        ("A (Supervisory)", model_1, FEAT_COLS_1),
        ("B (Operator)",    model_2, FEAT_COLS_2),
        ("C (Unsafe Acts)", model_3, FEAT_COLS_3),
    ]:
        feat_names = _get_feature_names(model, feat_cols)
        importances = model.named_steps["rf"].feature_importances_
        for fname, imp in zip(feat_names, importances):
            imp_rows.append({"model": model_label, "feature": fname, "importance": imp})

    imp_df = pd.DataFrame(imp_rows)
    imp_csv = os.path.join(RESULTS_DIR, "rf_feature_importance.csv")
    imp_df.to_csv(imp_csv, index=False)
    print(f"Feature importances saved to {imp_csv}")

    # ── Feature importance plot ───────────────────────────────────────────────
    model_labels = ["A (Supervisory)", "B (Operator)", "C (Unsafe Acts)"]
    colors = ["steelblue", "darkorange", "green"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (model_label, color) in zip(axes, zip(model_labels, colors)):
        sub = imp_df[imp_df["model"] == model_label].sort_values("importance", ascending=False)
        ax.bar(range(len(sub)), sub["importance"].values, color=color)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["feature"].values, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Mean Decrease in Impurity")
        ax.set_title(f"Feature Importance — Model {model_label}")
        ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    imp_path = os.path.join(FIG_DIR, "rf_feature_importance.png")
    plt.savefig(imp_path, dpi=150)
    print(f"Feature importance plot saved to {imp_path}")
    plt.close()


if __name__ == "__main__":
    main()
