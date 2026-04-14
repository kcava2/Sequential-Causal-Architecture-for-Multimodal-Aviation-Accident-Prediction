import os
import sys
import pickle
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder as LE
from pgmpy.inference import VariableElimination
from sklearn.metrics import (
    classification_report, 
    balanced_accuracy_score, 
    f1_score, 
    cohen_kappa_score
)

# Add paths for local modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.bayesian_net.train import (  # noqa: E402
    load_and_split, run_inference, EVIDENCE_COLS
)

from models.eval_utils import (  # noqa: E402
    clean_feature_name, apply_plot_style,
    CONFUSION_FIGSIZE, plot_roc_curves, plot_sensitivity_bars
)

def _run_inference_probs(model, df):
    ve = VariableElimination(model)
    nodes = ["Supervisory", "Operator", "UnsafeActs"]
    cpds = {node: model.get_cpds(node) for node in nodes}
    classes = {node: np.array(cpds[node].state_names[node]) for node in nodes}
    true_vals = {node: [] for node in nodes}
    probs = {node: [] for node in nodes}

    print("Extracting probabilities...")
    for _, row in df.iterrows():
        evidence = {col: str(row[col]) for col in EVIDENCE_COLS}
        for node in nodes:
            res = ve.query([node], evidence=evidence, show_progress=False)
            probs[node].append(res.values)
            true_vals[node].append(str(row[node]))
    return true_vals, {k: np.array(v) for k, v in probs.items()}, classes

def main():
    # --- Paths and Setup ---
    FILEPATH    = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH  = os.path.join(os.path.dirname(__file__), "hfacs_bn.pkl")
    FIG_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

    os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]

    df_train, _, df_test, _ = load_and_split(FILEPATH)

    # ── 1. Generalization Gap ────────────────────────────────────────────────
    print("\nComputing Generalization Gap...")
    tr_A, tr_B, tr_C, pr_tr_A, pr_tr_B, pr_tr_C = run_inference(model, df_train)
    ts_A, ts_B, ts_C, pr_ts_A, pr_ts_B, pr_ts_C = run_inference(model, df_test)

    gap_metrics = []
    tasks = [("Supervisory", tr_A, pr_tr_A, ts_A, pr_ts_A),
             ("Operator",    tr_B, pr_tr_B, ts_B, pr_ts_B),
             ("Unsafe Acts", tr_C, pr_tr_C, ts_C, pr_ts_C)]

    header = f"{'Task':<15} | {'Metric':<12} | {'Train':>10} | {'Test':>10} | {'Gap':>10}"
    print("\n" + header + "\n" + "-" * len(header))

    for name, tr_y, tr_p, ts_y, ts_p in tasks:
        for m_name, m_func in [("Bal-Acc", balanced_accuracy_score),
                               ("Macro-F1", lambda y, p: f1_score(y, p, average='macro', zero_division=0)),
                               ("Kappa", cohen_kappa_score)]:
            tr_v, ts_v = m_func(tr_y, tr_p), m_func(ts_y, ts_p)
            gap = tr_v - ts_v
            fmt = ".2%" if "Acc" in m_name else ".4f"
            print(f"{name:<15} | {m_name:<12} | {tr_v:>{10}{fmt}} | {ts_v:>{10}{fmt}} | {gap:>{10}{fmt}}")
            gap_metrics.append({"Task": name, "Metric": m_name, "Train": tr_v, "Test": ts_v, "Gap": gap})
        print("-" * len(header))
    pd.DataFrame(gap_metrics).to_csv(os.path.join(RESULTS_DIR, "bn_generalization_gap.csv"), index=False)

    # ── 2. Feature Ablation Sensitivity (With Error Bars) ──────────────────
    print("\nRunning Bootstrapped BN Feature Ablation (30 resamples)...")
    N_BOOTSTRAP = 30
    rng = np.random.default_rng(42)
    abl_results = []

    # Evidence columns to test
    for drop_col in [None] + EVIDENCE_COLS:
        col_label = drop_col if drop_col else "None (baseline)"
        drops_per_task = {"Supervisory": [], "Operator": [], "Unsafe Acts": []}

        for _ in range(N_BOOTSTRAP):
            # Bootstrap resample
            idx = rng.choice(len(df_test), size=len(df_test), replace=True)
            df_boot = df_test.iloc[idx].reset_index(drop=True)
            
            # 1. Baseline (always full evidence for this bootstrap sample)
            tA, tB, tC, pA, pB, pC = run_inference(model, df_boot)
            base_accs = {
                "Supervisory": balanced_accuracy_score(tA, pA),
                "Operator":    balanced_accuracy_score(tB, pB),
                "Unsafe Acts": balanced_accuracy_score(tC, pC)
            }

            if drop_col is None:
                # Store baseline as 0 drop for error bars
                for task in drops_per_task: drops_per_task[task].append(0.0)
            else:
                # 2. Ablated (drop the specific evidence column)
                tA_a, tB_a, tC_a, pA_a, pB_a, pC_a = run_inference(model, df_boot, drop_evidence={drop_col})
                abl_accs = {
                    "Supervisory": balanced_accuracy_score(tA_a, pA_a),
                    "Operator":    balanced_accuracy_score(tB_a, pB_a),
                    "Unsafe Acts": balanced_accuracy_score(tC_a, pC_a)
                }
                for task in drops_per_task:
                    drops_per_task[task].append(base_accs[task] - abl_accs[task])

        # Aggregate stats
        for task, drop_list in drops_per_task.items():
            abl_results.append({
                "ablated": col_label,
                "task": task,
                "mean_drop": np.mean(drop_list),
                "std_drop": np.std(drop_list)
            })

    abl_df = pd.DataFrame(abl_results)
    abl_df.to_csv(os.path.join(RESULTS_DIR, "bn_sensitivity.csv"), index=False)

    # Plotting with error bars
    plot_df = abl_df[abl_df["ablated"] != "None (baseline)"].copy()
    plot_sensitivity_bars(
        plot_df,
        feature_col="ablated",
        feature_order=EVIDENCE_COLS,
        title="BN Feature Ablation: Accuracy Drop with 95% Bootstrap CI",
        save_path=os.path.join(FIG_DIR, "bn_sensitivity.png"),
        task_col="task",
        mean_col="mean_drop",
        err_col="std_drop",
        tasks=["Supervisory", "Operator", "Unsafe Acts"]
    )

    # ── 3. SHAP Analysis (Strict Dimension Fix) ──────────────────────────
    print("\nComputing BN SHAP (KernelExplainer)...")
    bn_enc = {col: LE().fit(df_test[col].astype(str)) for col in EVIDENCE_COLS}
    X_numeric = np.column_stack([bn_enc[col].transform(df_test[col].astype(str)) for col in EVIDENCE_COLS]).astype(float)
    bg_data, eval_data = X_numeric[:20], X_numeric[20:70] 
    ve_shap = VariableElimination(model)
    feature_names = [clean_feature_name(c) for c in EVIDENCE_COLS]

    for node in ["Supervisory", "Operator", "UnsafeActs"]:
        def wrapper_f(X_in):
            out = []
            for row in X_in:
                ev = {col: bn_enc[col].inverse_transform([int(round(row[i]))])[0] for i, col in enumerate(EVIDENCE_COLS)}
                res = ve_shap.query([node], evidence=ev, show_progress=False); out.append(res.values)
            return np.array(out)

        explainer = shap.KernelExplainer(wrapper_f, bg_data)
        shap_values = explainer.shap_values(eval_data, nsamples=100)
        actual_sv = np.array(shap_values[0]) if isinstance(shap_values, list) else np.array(shap_values)
        if actual_sv.ndim == 3: actual_sv = actual_sv[:, :, 0]
        
        base_val = float(explainer.expected_value[0]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)

        exp = shap.Explanation(values=actual_sv.astype(float), base_values=base_val, data=eval_data.astype(float), feature_names=feature_names)
        
        plt.figure(); shap.plots.beeswarm(exp, max_display=10, show=False)
        plt.title(f"BN SHAP Summary: {node}"); plt.savefig(os.path.join(FIG_DIR, f"bn_shap_summary_{node}.png"), bbox_inches='tight'); plt.close()

    print(f"\nEvaluation Complete. All figures saved to {FIG_DIR}")

if __name__ == "__main__":
    main()