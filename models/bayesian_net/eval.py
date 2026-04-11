import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score, f1_score, cohen_kappa_score,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.bayesian_net.train import (  # noqa: E402
    load_and_split, run_inference, EVIDENCE_COLS, QUERY_COLS
)


def _run_inference_probs(model, df, drop_evidence=None):
    """
    Run probabilistic inference for each row.
    Returns (true_A, true_B, true_C, probs_A, probs_B, probs_C, classes_A, classes_B, classes_C)
    where probs_* are (N, n_classes) numpy arrays.
    """
    ve   = VariableElimination(model)
    drop = set(drop_evidence or [])

    cpd_A = model.get_cpds("Supervisory")
    cpd_B = model.get_cpds("Operator")
    cpd_C = model.get_cpds("UnsafeActs")
    classes_A = np.array(cpd_A.state_names["Supervisory"])
    classes_B = np.array(cpd_B.state_names["Operator"])
    classes_C = np.array(cpd_C.state_names["UnsafeActs"])

    true_A, true_B, true_C = [], [], []
    probs_A, probs_B, probs_C = [], [], []

    for _, row in df.iterrows():
        evidence = {col: str(row[col]) for col in EVIDENCE_COLS if col not in drop}
        # ve.query() returns a DiscreteFactor; .values is already ordered by state_names
        res_A = ve.query(["Supervisory"], evidence=evidence, show_progress=False)
        res_B = ve.query(["Operator"],    evidence=evidence, show_progress=False)
        res_C = ve.query(["UnsafeActs"],  evidence=evidence, show_progress=False)
        probs_A.append(res_A.values)
        probs_B.append(res_B.values)
        probs_C.append(res_C.values)
        true_A.append(str(row["Supervisory"]))
        true_B.append(str(row["Operator"]))
        true_C.append(str(row["UnsafeActs"]))

    return (
        true_A, true_B, true_C,
        np.array(probs_A), np.array(probs_B), np.array(probs_C),
        classes_A, classes_B, classes_C,
    )


def _optimize_thresholds(probs, true, classes, n_grid=20):
    """
    Find per-class thresholds maximizing balanced accuracy.
    Uses ratio form: predicted = classes[argmax(prob[c] / threshold[c])].
    Returns a 1-D numpy array of thresholds.
    """
    n_classes  = probs.shape[1]
    thresholds = np.ones(n_classes)
    best_acc   = balanced_accuracy_score(true, classes[probs.argmax(axis=1)])

    grid = np.linspace(0.1, 2.0, n_grid)
    for c in range(n_classes):
        best_t = 1.0
        for t in grid:
            thresholds[c] = t
            preds = classes[(probs / thresholds).argmax(axis=1)]
            acc   = balanced_accuracy_score(true, preds)
            if acc > best_acc:
                best_acc = acc
                best_t   = t
        thresholds[c] = best_t

    return thresholds


def main():
    FILEPATH    = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH  = os.path.join(os.path.dirname(__file__), "hfacs_bn.pkl")
    FIG_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]

    _, df_val, df_test, _ = load_and_split(FILEPATH)

    # ── Full classification reports ───────────────────────────────────────────
    print(f"Test samples: {len(df_test)}\n")
    true_A, true_B, true_C, pred_A, pred_B, pred_C = run_inference(model, df_test)

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

    eval_csv = os.path.join(RESULTS_DIR, "bn_eval_metrics.csv")
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
    cm_path = os.path.join(FIG_DIR, "bn_confusion_matrices.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrices saved to {cm_path}")
    plt.close()

    # ── Threshold optimization (fit on val, apply to test) ────────────────────
    print("\nOptimizing per-class decision thresholds on validation set...")
    (vA, vB, vC,
     vp_A, vp_B, vp_C,
     cls_A, cls_B, cls_C) = _run_inference_probs(model, df_val)

    thresh_A = _optimize_thresholds(vp_A, vA, cls_A)
    thresh_B = _optimize_thresholds(vp_B, vB, cls_B)
    thresh_C = _optimize_thresholds(vp_C, vC, cls_C)

    (tA, tB, tC,
     tp_A, tp_B, tp_C,
     _, _, _) = _run_inference_probs(model, df_test)

    th_pred_A = cls_A[(tp_A / thresh_A).argmax(axis=1)]
    th_pred_B = cls_B[(tp_B / thresh_B).argmax(axis=1)]
    th_pred_C = cls_C[(tp_C / thresh_C).argmax(axis=1)]

    th_bal_A = balanced_accuracy_score(tA, th_pred_A)
    th_bal_B = balanced_accuracy_score(tB, th_pred_B)
    th_bal_C = balanced_accuracy_score(tC, th_pred_C)
    th_f1_A  = f1_score(tA, th_pred_A, average="macro", zero_division=0)
    th_f1_B  = f1_score(tB, th_pred_B, average="macro", zero_division=0)
    th_f1_C  = f1_score(tC, th_pred_C, average="macro", zero_division=0)
    th_kap_A = cohen_kappa_score(tA, th_pred_A)
    th_kap_B = cohen_kappa_score(tB, th_pred_B)
    th_kap_C = cohen_kappa_score(tC, th_pred_C)

    print("\nThresholded Test Metrics:")
    print(f"{'Metric':<22} {'A (Supervisory)':>16} {'B (Operator)':>14} {'C (Unsafe Acts)':>16}")
    print("-" * 72)
    print(f"{'Balanced Accuracy':<22} {th_bal_A:>16.2%} {th_bal_B:>14.2%} {th_bal_C:>16.2%}")
    print(f"{'Macro F1':<22} {th_f1_A:>16.4f} {th_f1_B:>14.4f} {th_f1_C:>16.4f}")
    print(f"{'Cohen Kappa':<22} {th_kap_A:>16.4f} {th_kap_B:>14.4f} {th_kap_C:>16.4f}")

    th_rows = []
    for task_label, true, pred, bal, f1, kappa in [
        ("Supervisory (A)", tA, th_pred_A, th_bal_A, th_f1_A, th_kap_A),
        ("Operator (B)",    tB, th_pred_B, th_bal_B, th_f1_B, th_kap_B),
        ("Unsafe Acts (C)", tC, th_pred_C, th_bal_C, th_f1_C, th_kap_C),
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

    th_csv = os.path.join(RESULTS_DIR, "bn_eval_metrics_thresholded.csv")
    pd.DataFrame(th_rows).to_csv(th_csv, index=False)
    print(f"Thresholded eval metrics saved to {th_csv}")

    # ── Feature-ablation sensitivity ─────────────────────────────────────────
    # For a BN, noise injection is not natural. Instead, we systematically
    # drop one evidence variable at a time (marginalizing it out) and measure
    # how much balanced accuracy degrades — telling us which features matter most.
    print("\nRunning feature-ablation sensitivity analysis...")

    abl_rows = []

    # Baseline (no ablation)
    for task_key, task_label, true, pred in [
        ("A", "Supervisory", true_A, pred_A),
        ("B", "Operator",    true_B, pred_B),
        ("C", "UnsafeActs",  true_C, pred_C),
    ]:
        abl_rows.append({
            "ablated":   "None (baseline)",
            "task":      task_label,
            "bal_acc":   balanced_accuracy_score(true, pred),
        })

    # Single-feature ablations
    for drop_col in EVIDENCE_COLS:
        tA, tB, tC, pA, pB, pC = run_inference(model, df_test, drop_evidence={drop_col})
        for task_label, true, pred in [
            ("Supervisory", tA, pA),
            ("Operator",    tB, pB),
            ("UnsafeActs",  tC, pC),
        ]:
            abl_rows.append({
                "ablated": drop_col,
                "task":    task_label,
                "bal_acc": balanced_accuracy_score(true, pred),
            })

    abl_df = pd.DataFrame(abl_rows)

    # Compute accuracy drop relative to baseline
    baseline_map = {
        row["task"]: row["bal_acc"]
        for _, row in abl_df[abl_df["ablated"] == "None (baseline)"].iterrows()
    }
    abl_df["baseline_acc"] = abl_df["task"].map(baseline_map)
    abl_df["acc_drop"]     = abl_df["baseline_acc"] - abl_df["bal_acc"]

    abl_csv = os.path.join(RESULTS_DIR, "bn_ablation_sensitivity.csv")
    abl_df.to_csv(abl_csv, index=False)
    print(f"Ablation results saved to {abl_csv}")

    print("\nFeature Ablation (balanced accuracy drop vs. baseline):")
    print(f"{'Dropped Feature':<20}  {'Task':<14}  {'Bal Acc':>8}  {'Drop':>7}")
    print("-" * 56)
    for _, row in abl_df[abl_df["ablated"] != "None (baseline)"].sort_values(
            ["task", "acc_drop"], ascending=[True, False]).iterrows():
        print(f"{row['ablated']:<20}  {row['task']:<14}  "
              f"{row['bal_acc']:>8.2%}  {row['acc_drop']:>+7.2%}")

    # ── Ablation bar chart ────────────────────────────────────────────────────
    plot_df = abl_df[abl_df["ablated"] != "None (baseline)"].copy()
    tasks   = ["Supervisory", "Operator", "UnsafeActs"]
    x       = np.arange(len(EVIDENCE_COLS))
    width   = 0.25
    colors  = ["steelblue", "darkorange", "green"]

    _, ax = plt.subplots(figsize=(11, 5))
    for i, (task, color) in enumerate(zip(tasks, colors)):
        task_df = plot_df[plot_df["task"] == task].set_index("ablated")
        drops   = [task_df.loc[col, "acc_drop"] if col in task_df.index else 0
                   for col in EVIDENCE_COLS]
        ax.bar(x + i * width, drops, width, label=task, color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x + width)
    ax.set_xticklabels(EVIDENCE_COLS, rotation=30, ha="right")
    ax.set_ylabel("Balanced Accuracy Drop")
    ax.set_title("Feature Ablation: Accuracy Drop when Evidence is Removed")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    abl_path = os.path.join(FIG_DIR, "bn_ablation_sensitivity.png")
    plt.savefig(abl_path, dpi=150)
    print(f"Ablation plot saved to {abl_path}")
    plt.close()


if __name__ == "__main__":
    main()