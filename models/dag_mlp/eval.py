import os
import sys
import warnings

import shap
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score, f1_score, cohen_kappa_score,
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.real_dataloader import get_dataloaders               # noqa: E402
from models.dag_mlp.train import HFACSCausalDAGMLP, evaluate   # noqa: E402
from models.eval_utils import (                                # noqa: E402
    TASK_COLORS, clean_feature_name, apply_plot_style,
    plot_roc_curves, plot_sensitivity_bars, plot_confusion_matrices,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_metrics(true_A, true_B, true_C, pred_A, pred_B, pred_C):
    tasks = [
        ("Supervisory", true_A, pred_A),
        ("Operator",    true_B, pred_B),
        ("Unsafe Acts", true_C, pred_C),
    ]
    results = []
    for name, true, pred in tasks:
        results.append({
            "Task":     name,
            "Bal-Acc":  balanced_accuracy_score(true, pred),
            "Macro-F1": f1_score(true, pred, average="macro", zero_division=0),
            "Kappa":    cohen_kappa_score(true, pred),
        })
    return results


def _collect_probs(model, loader, device):
    """Return softmax probability arrays for ROC analysis."""
    model.eval()
    probs_A, probs_B, probs_C = [], [], []
    true_A,  true_B,  true_C  = [], [], []
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


def _batch_infer(model, s_a_arr, s0_arr, dev, batch_size=64):
    """Batch inference for bootstrapped ablation samples."""
    model.eval()
    pA_all, pB_all, pC_all = [], [], []
    with torch.no_grad():
        for i in range(0, len(s_a_arr), batch_size):
            lA, lB, lC = model(
                s_a_arr[i:i + batch_size].to(dev),
                s0_arr[i:i + batch_size].to(dev),
            )
            pA_all.extend(lA.argmax(1).cpu().tolist())
            pB_all.extend(lB.argmax(1).cpu().tolist())
            pC_all.extend(lC.argmax(1).cpu().tolist())
    return np.array(pA_all), np.array(pB_all), np.array(pC_all)


class _DAGMLPOutputWrapper(nn.Module):
    """Concatenates [s_a | s0] → (B, 7) for SHAP GradientExplainer compatibility."""
    def __init__(self, dag_model, output_idx):
        super().__init__()
        self.model      = dag_model
        self.output_idx = output_idx

    def forward(self, x):
        return self.model(x[:, :2], x[:, 2:])[self.output_idx]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    FILEPATH    = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH  = os.path.join(os.path.dirname(__file__), "hfacs_dag_mlp.pt")
    FIG_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    BATCH_SIZE, N_BOOTSTRAP = 32, 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    apply_plot_style()

    train_loader, _, test_loader, encoders = get_dataloaders(FILEPATH, batch_size=BATCH_SIZE)

    # ── Load model ─────────────────────────────────────────────────────────────
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd          = ckpt["state_dict"]
        hidden_size = ckpt.get("config", {}).get("hidden_size", None)
    else:
        sd          = ckpt
        hidden_size = None
    if hidden_size is None:
        hidden_size = sd["mlp_a.0.weight"].shape[0]

    n_A = len(encoders.enc_supervisory.classes_)
    n_B = len(encoders.enc_operator.classes_)
    n_C = len(encoders.enc_unsafe.classes_)

    model = HFACSCausalDAGMLP(hidden_size=hidden_size, n_A=n_A, n_B=n_B, n_C=n_C).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()

    # ── 1. Generalization Gap ──────────────────────────────────────────────────
    print("Computing Generalization Gap...")
    tr_tA, tr_tB, tr_tC, tr_pA, tr_pB, tr_pC = evaluate(model, train_loader, DEVICE)
    ts_tA, ts_tB, ts_tC, ts_pA, ts_pB, ts_pC = evaluate(model, test_loader,  DEVICE)

    tr_mets = _run_metrics(tr_tA, tr_tB, tr_tC, tr_pA, tr_pB, tr_pC)
    ts_mets = _run_metrics(ts_tA, ts_tB, ts_tC, ts_pA, ts_pB, ts_pC)

    gap_rows = []
    header = f"{'Task':<15} | {'Metric':<12} | {'Train':>10} | {'Test':>10} | {'Gap':>10}"
    print("\n" + header + "\n" + "-" * 65)
    for tr, ts in zip(tr_mets, ts_mets):
        for m in ["Bal-Acc", "Macro-F1", "Kappa"]:
            gap = tr[m] - ts[m]
            fmt = ".2%" if "Acc" in m else ".4f"
            print(f"{tr['Task']:<15} | {m:<12} | {tr[m]:>{10}{fmt}} | {ts[m]:>{10}{fmt}} | {gap:>{10}{fmt}}")
            gap_rows.append({"Task": tr["Task"], "Metric": m, "Train": tr[m], "Test": ts[m], "Gap": gap})
        print("-" * 65)
    pd.DataFrame(gap_rows).to_csv(
        os.path.join(RESULTS_DIR, "dag_mlp_generalization_gap.csv"), index=False
    )

    # ── 2. Bootstrapped Feature Ablation ──────────────────────────────────────
    print("\nRunning Bootstrapped Feature Ablation (30 resamples)...")
    all_s_a, all_s0 = [], []
    with torch.no_grad():
        for s_a, s0, _, _, _ in test_loader:
            all_s_a.append(s_a)
            all_s0.append(s0)
    s_a_ts, s0_ts = torch.cat(all_s_a), torch.cat(all_s0)

    ABL_MAP = {
        "Org. Climate": ("s_a", 0),
        "Employment":   ("s_a", 1),
        "Weather":      ("s0",  0),
        "Time of Day":  ("s0",  1),
        "Sky Cond.":    ("s0",  2),
        "Personnel":    ("s0",  3),
    }
    rng            = np.random.default_rng(42)
    final_abl_rows = []

    for feat in [None] + list(ABL_MAP.keys()):
        drops = {"Supervisory": [], "Operator": [], "Unsafe Acts": []}
        for _ in range(N_BOOTSTRAP):
            idx   = rng.choice(len(s_a_ts), size=len(s_a_ts), replace=True)
            s_a_b = s_a_ts[idx]
            s0_b  = s0_ts[idx]
            tA_b  = np.array(ts_tA)[idx]
            tB_b  = np.array(ts_tB)[idx]
            tC_b  = np.array(ts_tC)[idx]

            pA_base, pB_base, pC_base = _batch_infer(model, s_a_b, s0_b, DEVICE)
            base_accs = [
                balanced_accuracy_score(tA_b, pA_base),
                balanced_accuracy_score(tB_b, pB_base),
                balanced_accuracy_score(tC_b, pC_base),
            ]

            if feat is None:
                for k in drops:
                    drops[k].append(0.0)
            else:
                s_a_a, s0_a = s_a_b.clone(), s0_b.clone()
                t_name, col = ABL_MAP[feat]
                if t_name == "s_a":
                    s_a_a[:, col] = 0.0
                else:
                    s0_a[:, col] = 0.0
                pA_a, pB_a, pC_a = _batch_infer(model, s_a_a, s0_a, DEVICE)
                abl_accs = [
                    balanced_accuracy_score(tA_b, pA_a),
                    balanced_accuracy_score(tB_b, pB_a),
                    balanced_accuracy_score(tC_b, pC_a),
                ]
                for i, k in enumerate(drops):
                    drops[k].append(base_accs[i] - abl_accs[i])

        for k in drops:
            final_abl_rows.append({
                "ablated":   feat or "Baseline",
                "task":      k,
                "mean_drop": np.mean(drops[k]),
                "std_drop":  np.std(drops[k]),
            })

    abl_df = pd.DataFrame(final_abl_rows)
    abl_df.to_csv(os.path.join(RESULTS_DIR, "dag_mlp_sensitivity.csv"), index=False)

    plot_sensitivity_bars(
        abl_df[abl_df["ablated"] != "Baseline"].copy(),
        feature_col="ablated",
        feature_order=list(ABL_MAP.keys()),
        title="DAG-MLP Feature Ablation: Accuracy Drop with 95% Bootstrap CI",
        save_path=os.path.join(FIG_DIR, "dag_mlp_sensitivity.png"),
        task_col="task",
        mean_col="mean_drop",
        err_col="std_drop",
        tasks=["Supervisory", "Operator", "Unsafe Acts"],
    )

    # ── 3. ROC Curves & Classification Reports ─────────────────────────────────
    print("\nComputing ROC Curves and Classification Reports...")
    pA, tA, pB, tB, pC, tC = _collect_probs(model, test_loader, DEVICE)
    roc_data = [
        ("Supervisory", pA, tA, np.arange(n_A)),
        ("Operator",    pB, tB, np.arange(n_B)),
        ("Unsafe Acts", pC, tC, np.arange(n_C)),
    ]
    plot_roc_curves(
        roc_data, "DAG-MLP",
        os.path.join(FIG_DIR, "dag_mlp_roc_curves.png"),
        csv_path=os.path.join(RESULTS_DIR, "dag_mlp_roc_auc.csv"),
    )

    for lbl, t, p, enc in [
        ("Supervisory", ts_tA, ts_pA, encoders.enc_supervisory),
        ("Operator",    ts_tB, ts_pB, encoders.enc_operator),
        ("Unsafe Acts", ts_tC, ts_pC, encoders.enc_unsafe),
    ]:
        print(f"\n── {lbl} ──\n",
              classification_report(t, p, target_names=enc.classes_, zero_division=0))

    # ── 4. Confusion Matrices ──────────────────────────────────────────────────
    print("\nPlotting Confusion Matrices...")
    cm_data = [
        ("Supervisory", ts_tA, ts_pA, encoders.enc_supervisory.classes_),
        ("Operator",    ts_tB, ts_pB, encoders.enc_operator.classes_),
        ("Unsafe Acts", ts_tC, ts_pC, encoders.enc_unsafe.classes_),
    ]
    plot_confusion_matrices(
        cm_data, "DAG-MLP",
        os.path.join(FIG_DIR, "dag_mlp_confusion_matrices.png"),
    )

    # ── 5. SHAP Analysis ───────────────────────────────────────────────────────
    print("\nComputing DAG-MLP SHAP (GradientExplainer)...")
    X_cat = torch.cat([s_a_ts, s0_ts], dim=1).to(DEVICE)
    feat_names = [
        clean_feature_name(f) for f in [
            "Org. Climate", "Employment", "Weather",
            "Time of Day",  "Sky Cond.",  "Personnel", "Supervisory",
        ]
    ]

    for label, idx in [("Supervisory", 0), ("Operator", 1), ("Unsafe_Acts", 2)]:
        wrapper  = _DAGMLPOutputWrapper(model, idx).to(DEVICE)
        explainer = shap.GradientExplainer(wrapper, X_cat[:50])
        sv        = explainer.shap_values(X_cat[50:100])

        if isinstance(sv, list):
            actual_sv = np.array(sv[0])
        else:
            actual_sv = np.array(sv)
            if actual_sv.ndim == 3:
                actual_sv = actual_sv[:, :, 0]

        with torch.no_grad():
            bg_p    = wrapper(X_cat[:50]).cpu().numpy()
            base_v  = float(np.mean(bg_p[:, 0])) if bg_p.ndim > 1 else float(np.mean(bg_p))

        exp = shap.Explanation(
            values=actual_sv.astype(float),
            base_values=base_v,
            data=X_cat[50:100].cpu().numpy(),
            feature_names=feat_names,
        )

        plt.figure()
        shap.plots.beeswarm(exp, max_display=10, show=False)
        plt.title(f"DAG-MLP SHAP Summary: {label}")
        plt.savefig(os.path.join(FIG_DIR, f"dag_mlp_shap_summary_{label}.png"), bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.plots.waterfall(exp[0], max_display=10, show=False)
        plt.title(f"DAG-MLP SHAP Waterfall: {label} (Sample 0)")
        plt.savefig(os.path.join(FIG_DIR, f"dag_mlp_shap_waterfall_{label}.png"), bbox_inches="tight")
        plt.close()

    print(f"\nEvaluation complete. Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
