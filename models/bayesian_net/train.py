import os
import sys
import pickle
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# ── DAG ───────────────────────────────────────────────────────────────────────
#
#  Step A:  OrgClimate, Employment  ->  Supervisory
#  Step B:  Supervisory, Weather, TimeOfDay, SkyCondNonceil, Personnel  ->  Operator
#  Step C:  Supervisory, Operator  ->  UnsafeActs
#
EDGES = [
    ("OrgClimate",     "Supervisory"),
    ("Employment",     "Supervisory"),
    ("Supervisory",    "Operator"),
    ("Weather",        "Operator"),
    ("TimeOfDay",      "Operator"),
    ("SkyCondNonceil", "Operator"),
    ("Personnel",      "Operator"),
    ("Supervisory",    "UnsafeActs"),
    ("Operator",       "UnsafeActs"),
]

EVIDENCE_COLS = ["OrgClimate", "Employment", "Weather", "TimeOfDay", "SkyCondNonceil", "Personnel"]
QUERY_COLS    = ["Supervisory", "Operator", "UnsafeActs"]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_raw(filepath):
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        "merged_air_crash_data_weather.sky_cond_nonceil": "SkyCondNonceil"
    })
    df["WeatherCondition"]    = df["WeatherCondition"].fillna("Unknown")
    df["SkyCondNonceil"]      = df["SkyCondNonceil"].fillna("Unknown")
    df["Personnel Conditions"] = df["Personnel Conditions"].fillna("Unknown")
    df = df.dropna(subset=["Supervisory Conditions", "Operator Conditions", "Unsafe Conditions"])
    return df.reset_index(drop=True)


def prepare_df(df, emp_bins=None):
    """
    Rename columns to short BN node names and discretize Employment.

    emp_bins : pd.IntervalIndex from training data (pass None to fit from df).
    Returns (prepared_df, emp_bins).
    """
    out = pd.DataFrame()
    out["OrgClimate"]     = df["Organizational Climate"].astype(str).values
    out["Weather"]        = df["WeatherCondition"].astype(str).values
    out["TimeOfDay"]      = df["TimeOfDay"].astype(str).values
    out["SkyCondNonceil"] = df["SkyCondNonceil"].astype(str).values
    out["Personnel"]      = df["Personnel Conditions"].astype(str).values
    out["Supervisory"]    = df["Supervisory Conditions"].astype(str).values
    out["Operator"]       = df["Operator Conditions"].astype(str).values
    out["UnsafeActs"]     = df["Unsafe Conditions"].astype(str).values

    emp = df["Employment, Total Weighted Avg CY_QoQ_pct"].values
    if emp_bins is None:
        _, emp_bins = pd.qcut(emp, q=3, labels=["Low", "Mid", "High"], retbins=True)
        emp_bins[0]  = -np.inf
        emp_bins[-1] =  np.inf
    out["Employment"] = pd.cut(emp, bins=emp_bins, labels=["Low", "Mid", "High"]).astype(str)

    return out, emp_bins


def load_and_split(filepath, test_split=0.2, val_split=0.1, seed=42):
    """
    Returns (df_train, df_val, df_test, emp_bins) with BN-ready column names.
    Uses the same deterministic split as real_dataloader.py.
    """
    df = _load_raw(filepath)
    n  = len(df)

    rng  = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()

    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_test - n_val

    raw_train = df.iloc[perm[:n_train]].reset_index(drop=True)
    raw_val   = df.iloc[perm[n_train : n_train + n_val]].reset_index(drop=True)
    raw_test  = df.iloc[perm[n_train + n_val :]].reset_index(drop=True)

    df_train, emp_bins = prepare_df(raw_train)
    df_val,   _        = prepare_df(raw_val,  emp_bins)
    df_test,  _        = prepare_df(raw_test, emp_bins)

    return df_train, df_val, df_test, emp_bins


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, df, drop_evidence=None):
    """
    Run MAP inference for each row in df.

    drop_evidence : set of evidence column names to omit (ablation).
    Returns (true_A, true_B, true_C, pred_A, pred_B, pred_C).
    """
    ve = VariableElimination(model)
    drop = set(drop_evidence or [])

    true_A, true_B, true_C = [], [], []
    pred_A, pred_B, pred_C = [], [], []

    for _, row in df.iterrows():
        evidence = {col: str(row[col]) for col in EVIDENCE_COLS if col not in drop}
        result   = ve.map_query(QUERY_COLS, evidence=evidence, show_progress=False)
        pred_A.append(result["Supervisory"])
        pred_B.append(result["Operator"])
        pred_C.append(result["UnsafeActs"])
        true_A.append(row["Supervisory"])
        true_B.append(row["Operator"])
        true_C.append(row["UnsafeActs"])

    return true_A, true_B, true_C, pred_A, pred_B, pred_C


# ── Helpers ───────────────────────────────────────────────────────────────────

def _oversample_df(df, target_cols, random_state=42):
    """
    Random oversampling for multi-target discrete data.
    For each target column, duplicates minority-class rows up to the majority-class count.
    All per-target synthetic rows are pooled once and concatenated with the original df.
    """
    rng = np.random.default_rng(random_state)
    extra_rows = []
    for col in target_cols:
        counts = df[col].value_counts()
        majority_count = counts.iloc[0]
        for cls, cnt in counts.items():
            # Partial oversampling: bring minority classes to 50% of majority count
            # (softer than 100% to avoid uniform CPTs that over-predict minorities)
            target_count = int(majority_count * 0.5)
            if cnt < target_count:
                minority_rows = df[df[col] == cls]
                n_needed = max(0, target_count - cnt)
                sampled = minority_rows.sample(
                    n=n_needed, replace=True,
                    random_state=int(rng.integers(0, 2**31))
                )
                extra_rows.append(sampled)
    if extra_rows:
        df_augmented = pd.concat([df] + extra_rows, ignore_index=True)
        return df_augmented.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def fit_model(df_train_balanced, pseudo_counts=2):
    """
    Fit a DiscreteBayesianNetwork on a pre-balanced training DataFrame.
    Returns the fitted model.
    """
    model = DiscreteBayesianNetwork(EDGES)
    model.fit(
        df_train_balanced,
        estimator=BayesianEstimator,
        prior_type="dirichlet",
        pseudo_counts=pseudo_counts,
    )
    return model


def main():
    FILEPATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hfacs_bn.pkl")

    df_train, df_val, df_test, emp_bins = load_and_split(FILEPATH)

    print("=" * 60)
    print("Bayesian Network - Causal DAG Structure")
    print("=" * 60)
    print("Step A -> predict: Supervisory Conditions")
    print("  Parents: OrgClimate, Employment (discretized)")
    print()
    print("Step B -> predict: Operator Conditions")
    print("  Parents: Supervisory, Weather, TimeOfDay, SkyCondNonceil, Personnel")
    print()
    print("Step C -> predict: Unsafe Acts")
    print("  Parents: Supervisory, Operator")
    print("=" * 60)
    print(f"Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")
    print()

    # ── Fit ───────────────────────────────────────────────────────────────────
    df_train_balanced = _oversample_df(df_train, ["Supervisory", "Operator", "UnsafeActs"], random_state=42)
    print(f"Training rows after oversampling: {len(df_train_balanced)}  (original: {len(df_train)})")

    model = fit_model(df_train_balanced, pseudo_counts=5)

    print("Model fitted. CPT entry counts per node:")
    for node in ["Supervisory", "Operator", "UnsafeActs"]:
        cpd = model.get_cpds(node)
        n_parent_combos = int(np.prod([len(v) for v in cpd.state_names.values() if v != cpd.state_names[node]]) or 1)
        n_classes = len(cpd.state_names[node])
        print(f"  {node:14s}: {n_parent_combos} parent combos x {n_classes} classes")
    print()

    # ── Training accuracy ─────────────────────────────────────────────────────
    print("Computing training metrics...")
    true_A, true_B, true_C, pred_A, pred_B, pred_C = run_inference(model, df_train)
    bal_A  = balanced_accuracy_score(true_A, pred_A)
    bal_B  = balanced_accuracy_score(true_B, pred_B)
    bal_C  = balanced_accuracy_score(true_C, pred_C)
    f1_A   = f1_score(true_A, pred_A, average="macro", zero_division=0)
    f1_B   = f1_score(true_B, pred_B, average="macro", zero_division=0)
    f1_C   = f1_score(true_C, pred_C, average="macro", zero_division=0)
    kap_A  = cohen_kappa_score(true_A, pred_A)
    kap_B  = cohen_kappa_score(true_B, pred_B)
    kap_C  = cohen_kappa_score(true_C, pred_C)

    print(f"\n{'─' * 72}")
    print("Final Training Metrics")
    print(f"{'─' * 72}")
    print(f"{'Metric':<22} {'A (Supervisory)':>16} {'B (Operator)':>14} {'C (Unsafe Acts)':>16}")
    print(f"{'─' * 72}")
    print(f"{'Balanced Accuracy':<22} {bal_A:>16.2%} {bal_B:>14.2%} {bal_C:>16.2%}")
    print(f"{'Macro F1':<22} {f1_A:>16.4f} {f1_B:>14.4f} {f1_C:>16.4f}")
    print(f"{'Cohen Kappa':<22} {kap_A:>16.4f} {kap_B:>14.4f} {kap_C:>16.4f}")
    print(f"{'─' * 72}")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "emp_bins": emp_bins}, f)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
