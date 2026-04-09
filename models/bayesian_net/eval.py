import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score, f1_score, cohen_kappa_score,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.bayesian_net.train import (  # noqa: E402
    load_and_split, run_inference, EVIDENCE_COLS
)


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

    _, _, df_test, _ = load_and_split(FILEPATH)

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