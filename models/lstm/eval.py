import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score, f1_score, cohen_kappa_score,
)
from scipy import stats

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.real_dataloader import get_dataloaders          # noqa: E402
from models.lstm.train import HFACSCausalLSTM             # noqa: E402


def _collect_probs(model, loader, device):
    """Return softmax probability arrays (N, n_classes) for each task on the given loader."""
    model.eval()
    probs_A, probs_B, probs_C = [], [], []
    true_A, true_B, true_C    = [], [], []
    with torch.no_grad():
        for s_a, s0, y_A, y_B, y_C in loader:
            lA, lB, lC = model(s_a.to(device), s0.to(device))
            probs_A.append(torch.softmax(lA, dim=1).cpu())
            probs_B.append(torch.softmax(lB, dim=1).cpu())
            probs_C.append(torch.softmax(lC, dim=1).cpu())
            true_A.extend(y_A.tolist())
            true_B.extend(y_B.tolist())
            true_C.extend(y_C.tolist())
    return (
        torch.cat(probs_A).numpy(), np.array(true_A),
        torch.cat(probs_B).numpy(), np.array(true_B),
        torch.cat(probs_C).numpy(), np.array(true_C),
    )


def optimize_thresholds(probs, true, n_grid=20):
    """
    Find per-class thresholds that maximize balanced accuracy on the given set.
    Uses the ratio form: predicted = argmax(prob[c] / threshold[c]).
    Thresholds are searched independently per class in [0.1, 2.0].
    Returns a 1-D numpy array of thresholds, one per class.
    """
    n_classes = probs.shape[1]
    best_thresholds = np.ones(n_classes)
    best_acc = balanced_accuracy_score(true, probs.argmax(axis=1))

    grid = np.linspace(0.1, 2.0, n_grid)
    thresholds = np.ones(n_classes)
    for c in range(n_classes):
        best_t = 1.0
        for t in grid:
            thresholds[c] = t
            preds = (probs / thresholds).argmax(axis=1)
            acc = balanced_accuracy_score(true, preds)
            if acc > best_acc:
                best_acc = acc
                best_t = t
        thresholds[c] = best_t
        best_thresholds[c] = best_t

    return best_thresholds


def apply_thresholds(probs, thresholds):
    return (probs / thresholds).argmax(axis=1)


def run_inference(model, loader, device, noise_std=0.0):
    """Return (true_A, true_B, true_C, pred_A, pred_B, pred_C)."""
    model.eval()
    true_A, true_B, true_C = [], [], []
    pred_A, pred_B, pred_C = [], [], []
    with torch.no_grad():
        for s_a, s0, y_A, y_B, y_C in loader:
            s_a = s_a.to(device)
            s0  = s0.to(device)
            if noise_std > 0:
                s_a = s_a + torch.randn_like(s_a) * noise_std
                s0  = s0  + torch.randn_like(s0)  * noise_std
            lA, lB, lC = model(s_a, s0)
            pred_A.extend(lA.argmax(1).cpu().tolist())
            pred_B.extend(lB.argmax(1).cpu().tolist())
            pred_C.extend(lC.argmax(1).cpu().tolist())
            true_A.extend(y_A.tolist())
            true_B.extend(y_B.tolist())
            true_C.extend(y_C.tolist())
    return true_A, true_B, true_C, pred_A, pred_B, pred_C


def main():
    FILEPATH    = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH  = os.path.join(os.path.dirname(__file__), "hfacs_lstm.pt")
    FIG_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    BATCH_SIZE   = 32
    N_TRIALS     = 30
    NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _, val_loader, test_loader, encoders = get_dataloaders(FILEPATH, batch_size=BATCH_SIZE)

    n_A = len(encoders.enc_supervisory.classes_)
    n_B = len(encoders.enc_operator.classes_)
    n_C = len(encoders.enc_unsafe.classes_)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    hidden_size = ckpt["cell_a.weight_hh"].shape[1]
    model = HFACSCausalLSTM(hidden_size=hidden_size, n_A=n_A, n_B=n_B, n_C=n_C).to(DEVICE)
    model.load_state_dict(ckpt)
    model.eval()

    # ── Classification reports ────────────────────────────────────────────────
    true_A, true_B, true_C, pred_A, pred_B, pred_C = run_inference(model, test_loader, DEVICE)

    bal_A = balanced_accuracy_score(true_A, pred_A)
    bal_B = balanced_accuracy_score(true_B, pred_B)
    bal_C = balanced_accuracy_score(true_C, pred_C)

    f1_A = f1_score(true_A, pred_A, average="macro", zero_division=0)
    f1_B = f1_score(true_B, pred_B, average="macro", zero_division=0)
    f1_C = f1_score(true_C, pred_C, average="macro", zero_division=0)

    kappa_A = cohen_kappa_score(true_A, pred_A)
    kappa_B = cohen_kappa_score(true_B, pred_B)
    kappa_C = cohen_kappa_score(true_C, pred_C)

    print(f"Test samples: {len(true_A)}\n")
    print(f"{'Metric':<22} {'A (Supervisory)':>16} {'B (Operator)':>14} {'C (Unsafe Acts)':>16}")
    print("-" * 72)
    print(f"{'Balanced Accuracy':<22} {bal_A:>16.2%} {bal_B:>14.2%} {bal_C:>16.2%}")
    print(f"{'Macro F1':<22} {f1_A:>16.4f} {f1_B:>14.4f} {f1_C:>16.4f}")
    print(f"{'Cohen Kappa':<22} {kappa_A:>16.4f} {kappa_B:>14.4f} {kappa_C:>16.4f}")
    print()
    print()
    print("── A: Supervisory Conditions ──")
    print(classification_report(true_A, pred_A,
                                target_names=encoders.enc_supervisory.classes_,
                                zero_division=0))
    print("── B: Operator Conditions ──")
    print(classification_report(true_B, pred_B,
                                target_names=encoders.enc_operator.classes_,
                                zero_division=0))
    print("── C: Unsafe Acts ──")
    print(classification_report(true_C, pred_C,
                                target_names=encoders.enc_unsafe.classes_,
                                zero_division=0))

    # ── Save per-class metrics CSV ────────────────────────────────────────────
    rows = []
    for task_label, true, pred, enc, bal, f1, kappa in [
        ("Supervisory (A)", true_A, pred_A, encoders.enc_supervisory, bal_A, f1_A, kappa_A),
        ("Operator (B)",    true_B, pred_B, encoders.enc_operator,    bal_B, f1_B, kappa_B),
        ("Unsafe Acts (C)", true_C, pred_C, encoders.enc_unsafe,      bal_C, f1_C, kappa_C),
    ]:
        report = classification_report(true, pred, target_names=enc.classes_,
                                       zero_division=0, output_dict=True)
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

    eval_csv = os.path.join(RESULTS_DIR, "lstm_eval_metrics.csv")
    pd.DataFrame(rows).to_csv(eval_csv, index=False)
    print(f"Eval metrics saved to {eval_csv}")

    # ── Confusion matrices ────────────────────────────────────────────────────
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (true, pred, enc, title) in zip(axes, [
        (true_A, pred_A, encoders.enc_supervisory, "A: Supervisory Conditions"),
        (true_B, pred_B, encoders.enc_operator,    "B: Operator Conditions"),
        (true_C, pred_C, encoders.enc_unsafe,      "C: Unsafe Acts"),
    ]):
        cm = confusion_matrix(true, pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=enc.classes_)
        disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(title)

    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, "lstm_confusion_matrices.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrices saved to {cm_path}")
    plt.close()

    # ── Threshold optimization (fit on val, apply to test) ────────────────────
    print("\nOptimizing per-class decision thresholds on validation set...")
    pA_v, tA_v, pB_v, tB_v, pC_v, tC_v = _collect_probs(model, val_loader, DEVICE)
    thresh_A = optimize_thresholds(pA_v, tA_v)
    thresh_B = optimize_thresholds(pB_v, tB_v)
    thresh_C = optimize_thresholds(pC_v, tC_v)

    pA_t, tA_t, pB_t, tB_t, pC_t, tC_t = _collect_probs(model, test_loader, DEVICE)
    th_pred_A = apply_thresholds(pA_t, thresh_A)
    th_pred_B = apply_thresholds(pB_t, thresh_B)
    th_pred_C = apply_thresholds(pC_t, thresh_C)

    th_bal_A = balanced_accuracy_score(tA_t, th_pred_A)
    th_bal_B = balanced_accuracy_score(tB_t, th_pred_B)
    th_bal_C = balanced_accuracy_score(tC_t, th_pred_C)
    th_f1_A  = f1_score(tA_t, th_pred_A, average="macro", zero_division=0)
    th_f1_B  = f1_score(tB_t, th_pred_B, average="macro", zero_division=0)
    th_f1_C  = f1_score(tC_t, th_pred_C, average="macro", zero_division=0)
    th_kap_A = cohen_kappa_score(tA_t, th_pred_A)
    th_kap_B = cohen_kappa_score(tB_t, th_pred_B)
    th_kap_C = cohen_kappa_score(tC_t, th_pred_C)

    print("\nThresholded Test Metrics:")
    print(f"{'Metric':<22} {'A (Supervisory)':>16} {'B (Operator)':>14} {'C (Unsafe Acts)':>16}")
    print("-" * 72)
    print(f"{'Balanced Accuracy':<22} {th_bal_A:>16.2%} {th_bal_B:>14.2%} {th_bal_C:>16.2%}")
    print(f"{'Macro F1':<22} {th_f1_A:>16.4f} {th_f1_B:>14.4f} {th_f1_C:>16.4f}")
    print(f"{'Cohen Kappa':<22} {th_kap_A:>16.4f} {th_kap_B:>14.4f} {th_kap_C:>16.4f}")

    th_rows = []
    for task_label, true, pred, enc, bal, f1, kappa in [
        ("Supervisory (A)", tA_t, th_pred_A, encoders.enc_supervisory, th_bal_A, th_f1_A, th_kap_A),
        ("Operator (B)",    tB_t, th_pred_B, encoders.enc_operator,    th_bal_B, th_f1_B, th_kap_B),
        ("Unsafe Acts (C)", tC_t, th_pred_C, encoders.enc_unsafe,      th_bal_C, th_f1_C, th_kap_C),
    ]:
        report = classification_report(true, pred, target_names=enc.classes_,
                                       zero_division=0, output_dict=True)
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

    th_csv = os.path.join(RESULTS_DIR, "lstm_eval_metrics_thresholded.csv")
    pd.DataFrame(th_rows).to_csv(th_csv, index=False)
    print(f"Thresholded eval metrics saved to {th_csv}")

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    print("\nRunning sensitivity analysis...")

    noise_results = {std: {"A": [], "B": [], "C": []} for std in NOISE_LEVELS}

    for noise_std in NOISE_LEVELS:
        for _ in range(N_TRIALS):
            tA, tB, tC, pA, pB, pC = run_inference(model, test_loader, DEVICE, noise_std)
            noise_results[noise_std]["A"].append(balanced_accuracy_score(tA, pA))
            noise_results[noise_std]["B"].append(balanced_accuracy_score(tB, pB))
            noise_results[noise_std]["C"].append(balanced_accuracy_score(tC, pC))

    baseline = noise_results[0.0]

    sens_rows = []
    for noise_std in NOISE_LEVELS:
        for task_key, task_label in [("A", "Supervisory"), ("B", "Operator"), ("C", "Unsafe Acts")]:
            trial_accs = noise_results[noise_std][task_key]
            base_accs  = baseline[task_key]
            mean_acc   = float(np.mean(trial_accs))
            std_acc    = float(np.std(trial_accs))

            if noise_std == 0.0:
                p_value     = 1.0
                significant = False
            else:
                _, p_value  = stats.ttest_rel(base_accs, trial_accs)
                p_value     = float(p_value)
                significant = p_value < 0.05

            sens_rows.append({
                "noise_std": noise_std,
                "task": task_label,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "p_value": p_value,
                "significant": significant,
            })

    sens_df = pd.DataFrame(sens_rows)
    sens_csv = os.path.join(RESULTS_DIR, "lstm_sensitivity.csv")
    sens_df.to_csv(sens_csv, index=False)
    print(f"Sensitivity results saved to {sens_csv}")

    print("\nSensitivity Analysis (balanced acc ± std, p-value vs. baseline):")
    print(f"{'Noise':>6}  {'Task':<14}  {'Mean Acc':>9}  {'Std':>6}  {'p-value':>9}  {'Sig?':>5}")
    print("-" * 60)
    for _, row in sens_df.iterrows():
        sig = "*" if row["significant"] else ""
        print(f"{row['noise_std']:>6.2f}  {row['task']:<14}  "
              f"{row['mean_acc']:>9.2%}  {row['std_acc']:>6.4f}  "
              f"{row['p_value']:>9.4f}  {sig:>5}")

    # ── Sensitivity plot ──────────────────────────────────────────────────────
    _, ax = plt.subplots(figsize=(9, 5))
    markers = {"Supervisory": "o", "Operator": "s", "Unsafe Acts": "^"}
    colors  = {"Supervisory": "steelblue", "Operator": "darkorange", "Unsafe Acts": "green"}

    for task_label in ["Supervisory", "Operator", "Unsafe Acts"]:
        task_df = sens_df[sens_df["task"] == task_label]
        ax.errorbar(
            task_df["noise_std"], task_df["mean_acc"], yerr=task_df["std_acc"],
            marker=markers[task_label], color=colors[task_label],
            label=task_label, capsize=4, linewidth=1.5,
        )
        for _, row in task_df[task_df["significant"]].iterrows():
            ax.annotate("*", (row["noise_std"], row["mean_acc"] - row["std_acc"] - 0.01),
                        ha="center", color=colors[task_label], fontsize=12)

    ax.set_xlabel("Noise Std Dev")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Sensitivity Analysis: Balanced Accuracy vs. Input Noise")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    sens_path = os.path.join(FIG_DIR, "lstm_sensitivity.png")
    plt.savefig(sens_path, dpi=150)
    print(f"Sensitivity plot saved to {sens_path}")
    plt.close()


if __name__ == "__main__":
    main()
