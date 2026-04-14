import os
import sys
import pickle

import numpy as np
import pandas as pd
import torch
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.real_dataloader import load_and_clean  # noqa: E402

# --------------------------------------------------------------------------- #
# Column constants (real dataset)
# --------------------------------------------------------------------------- #
FEAT_COLS_1 = ["Organizational Climate", "Employment, Total Weighted Avg CY_QoQ_pct"]
FEAT_COLS_2 = ["WeatherCondition", "TimeOfDay", "SkyCondNonceil", "Personnel Conditions",
               "Supervisory Conditions"]
FEAT_COLS_3 = ["Supervisory Conditions", "Operator Conditions"]

TARGET_A = "Supervisory Conditions"
TARGET_B = "Operator Conditions"
TARGET_C = "Unsafe Conditions"

RAND_STATE = 42

# Default hyperparameters (used by train.py; val.py searches over these)
DEFAULT_PARAMS_1 = dict(rf__n_estimators=100, rf__criterion="gini",
                        rf__max_features="sqrt", rf__min_samples_leaf=2,
                        rf__min_samples_split=2)
DEFAULT_PARAMS_2 = dict(rf__n_estimators=100, rf__criterion="gini",
                        rf__max_features="sqrt", rf__min_samples_leaf=2,
                        rf__min_samples_split=2)
DEFAULT_PARAMS_3 = dict(rf__n_estimators=100, rf__criterion="gini",
                        rf__max_features="sqrt", rf__min_samples_leaf=2,
                        rf__min_samples_split=2)


# --------------------------------------------------------------------------- #
# Data splitting (mirrors real_dataloader.get_dataloaders seed/logic)
# --------------------------------------------------------------------------- #
def load_and_split(filepath, test_split=0.2, val_split=0.1, seed=42):
    """Return (df_train, df_val, df_test) using same permutation as real_dataloader."""
    df = load_and_clean(filepath)
    n = len(df)

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()

    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_test - n_val

    df_train = df.iloc[perm[:n_train]].reset_index(drop=True)
    df_val   = df.iloc[perm[n_train : n_train + n_val]].reset_index(drop=True)
    df_test  = df.iloc[perm[n_train + n_val :]].reset_index(drop=True)

    return df_train, df_val, df_test


# --------------------------------------------------------------------------- #
# Pipeline factory — standard sklearn Pipeline (fully cloneable for GridSearchCV)
# --------------------------------------------------------------------------- #
def make_rf_pipeline(input_df):
    """Build a ColumnTransformer + RandomForestClassifier pipeline for input_df."""
    num_cols = input_df.select_dtypes(exclude=["object"]).columns.tolist()
    cat_cols = input_df.select_dtypes(include=["object"]).columns.tolist()

    encoder = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("encoder", encoder),
            ("rf", RandomForestClassifier(random_state=RAND_STATE, bootstrap=True)),
        ]
    )


# --------------------------------------------------------------------------- #
# SMOTE helper
# --------------------------------------------------------------------------- #
def _safe_smote(X, y, seed):
    """Run SMOTE; returns (X_synthetic, y_synthetic) — only the new rows."""
    counts = np.bincount(y)
    min_count = counts[counts > 0].min()
    if min_count < 2:
        return None, None
    k = min(5, min_count - 1)
    sm = SMOTETomek(smote=SMOTE(random_state=seed, k_neighbors=k), random_state=seed)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res[len(X):], y_res[len(y):]


# --------------------------------------------------------------------------- #
# Shared augmentation — exported for reuse in val.py
# --------------------------------------------------------------------------- #
def build_augmented_df(df_train):
    """
    Apply SMOTE to the training split and return the augmented DataFrame
    along with the fitted label encoders.

    Returns (df_aug, encoders_dict)
    """
    le_org  = LabelEncoder().fit(df_train["Organizational Climate"])
    le_wea  = LabelEncoder().fit(df_train["WeatherCondition"])
    le_tod  = LabelEncoder().fit(df_train["TimeOfDay"])
    le_sky  = LabelEncoder().fit(df_train["SkyCondNonceil"])
    le_pers = LabelEncoder().fit(df_train["Personnel Conditions"])
    le_sup  = LabelEncoder().fit(df_train[TARGET_A])
    le_op   = LabelEncoder().fit(df_train[TARGET_B])
    le_un   = LabelEncoder().fit(df_train[TARGET_C])

    X_enc = np.column_stack([
        le_org.transform(df_train["Organizational Climate"]),
        df_train["Employment, Total Weighted Avg CY_QoQ_pct"].values,
        le_wea.transform(df_train["WeatherCondition"]),
        le_tod.transform(df_train["TimeOfDay"]),
        le_sky.transform(df_train["SkyCondNonceil"]),
        le_pers.transform(df_train["Personnel Conditions"]),
    ])

    y_A_enc = le_sup.transform(df_train[TARGET_A])
    y_B_enc = le_op.transform(df_train[TARGET_B])
    y_C_enc = le_un.transform(df_train[TARGET_C])

    nn_lookup = NearestNeighbors(n_neighbors=1).fit(X_enc)

    def _inherit(X_synth, arrays):
        _, indices = nn_lookup.kneighbors(X_synth)
        idx = indices[:, 0]
        return [arr[idx] for arr in arrays]

    def _empty(n_cols, dtype):
        return np.empty((0, n_cols), dtype=dtype)

    X_synth_A, yA_synth = _safe_smote(X_enc, y_A_enc, RAND_STATE)
    X_synth_B, yB_synth = _safe_smote(X_enc, y_B_enc, RAND_STATE)
    X_synth_C, yC_synth = _safe_smote(X_enc, y_C_enc, RAND_STATE)

    if X_synth_A is None:
        X_synth_A = _empty(X_enc.shape[1], X_enc.dtype)
        yA_synth  = np.empty(0, dtype=y_A_enc.dtype)
        yB_from_A = yC_from_A = np.empty(0, dtype=y_B_enc.dtype)
    else:
        yB_from_A, yC_from_A = _inherit(X_synth_A, [y_B_enc, y_C_enc])

    if X_synth_B is None:
        X_synth_B = _empty(X_enc.shape[1], X_enc.dtype)
        yB_synth  = np.empty(0, dtype=y_B_enc.dtype)
        yA_from_B = yC_from_B = np.empty(0, dtype=y_A_enc.dtype)
    else:
        yA_from_B, yC_from_B = _inherit(X_synth_B, [y_A_enc, y_C_enc])

    if X_synth_C is None:
        X_synth_C = _empty(X_enc.shape[1], X_enc.dtype)
        yC_synth  = np.empty(0, dtype=y_C_enc.dtype)
        yA_from_C = yB_from_C = np.empty(0, dtype=y_A_enc.dtype)
    else:
        yA_from_C, yB_from_C = _inherit(X_synth_C, [y_A_enc, y_B_enc])

    X_aug  = np.vstack([X_enc,   X_synth_A,  X_synth_B,  X_synth_C])
    yA_aug = np.hstack([y_A_enc, yA_synth,   yA_from_B,  yA_from_C])
    yB_aug = np.hstack([y_B_enc, yB_from_A,  yB_synth,   yB_from_C])
    yC_aug = np.hstack([y_C_enc, yC_from_A,  yC_from_B,  yC_synth])

    df_aug = pd.DataFrame({
        "Organizational Climate":                    le_org.inverse_transform(X_aug[:, 0].astype(int)),
        "Employment, Total Weighted Avg CY_QoQ_pct": X_aug[:, 1],
        "WeatherCondition":                          le_wea.inverse_transform(X_aug[:, 2].astype(int)),
        "TimeOfDay":                                 le_tod.inverse_transform(X_aug[:, 3].astype(int)),
        "SkyCondNonceil":                            le_sky.inverse_transform(X_aug[:, 4].astype(int)),
        "Personnel Conditions":                      le_pers.inverse_transform(X_aug[:, 5].astype(int)),
        TARGET_A:                                    le_sup.inverse_transform(yA_aug),
        TARGET_B:                                    le_op.inverse_transform(yB_aug),
        TARGET_C:                                    le_un.inverse_transform(yC_aug),
    })

    return df_aug


# --------------------------------------------------------------------------- #
# Fit cascade — exported for reuse in val.py
# --------------------------------------------------------------------------- #
def fit_cascade(df_aug, params_1=None, params_2=None, params_3=None):
    """
    Fit the three cascading RF models on df_aug using the given param dicts.
    Uses DEFAULT_PARAMS_* when params are None.
    Returns (model_1, model_2, model_3).
    """
    p1 = params_1 or DEFAULT_PARAMS_1
    p2 = params_2 or DEFAULT_PARAMS_2
    p3 = params_3 or DEFAULT_PARAMS_3

    input_1 = df_aug[FEAT_COLS_1]
    y_A     = df_aug[TARGET_A]
    pipe_1  = make_rf_pipeline(input_1)
    pipe_1.set_params(**p1)
    model_1 = pipe_1.fit(input_1, y_A)
    A_pred_oof = cross_val_predict(model_1, input_1, y_A, cv=5, method="predict")

    input_2 = df_aug[["WeatherCondition", "TimeOfDay", "SkyCondNonceil",
                       "Personnel Conditions"]].copy()
    input_2["Supervisory Conditions"] = A_pred_oof
    y_B     = df_aug[TARGET_B]
    pipe_2  = make_rf_pipeline(input_2)
    pipe_2.set_params(**p2)
    model_2 = pipe_2.fit(input_2, y_B)
    B_pred_oof = cross_val_predict(model_2, input_2, y_B, cv=5, method="predict")

    input_3 = pd.DataFrame({
        "Supervisory Conditions": A_pred_oof,
        "Operator Conditions":    B_pred_oof,
    })
    y_C     = df_aug[TARGET_C]
    pipe_3  = make_rf_pipeline(input_3)
    pipe_3.set_params(**p3)
    model_3 = pipe_3.fit(input_3, y_C)

    return model_1, model_2, model_3, A_pred_oof, B_pred_oof, y_A, y_B, y_C


# --------------------------------------------------------------------------- #
# Training pipeline (default params, no search)
# --------------------------------------------------------------------------- #
def trainPipeline(filepath=None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hfacs_rf.pkl")

    datasplit = load_and_split(filepath)
    df_train, df_val, df_test = datasplit
    print(f"Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")

    df_aug = build_augmented_df(df_train)
    print(f"Training rows after SMOTE: {len(df_aug)}")

    model_1, model_2, model_3, A_oof, B_oof, y_A, y_B, y_C = fit_cascade(df_aug)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model_1": model_1, "model_2": model_2, "model_3": model_3}, f)
    print(f"Models saved to {MODEL_PATH}")

    # ── OOF metrics summary ───────────────────────────────────────────────── #
    input_3  = pd.DataFrame({"Supervisory Conditions": A_oof, "Operator Conditions": B_oof})
    C_oof    = model_3.predict(input_3)

    bal_A = balanced_accuracy_score(y_A, A_oof)
    bal_B = balanced_accuracy_score(y_B, B_oof)
    bal_C = balanced_accuracy_score(y_C, C_oof)

    f1_A = f1_score(y_A, A_oof, average="macro", zero_division=0)
    f1_B = f1_score(y_B, B_oof, average="macro", zero_division=0)
    f1_C = f1_score(y_C, C_oof, average="macro", zero_division=0)

    kA = cohen_kappa_score(y_A, A_oof)
    kB = cohen_kappa_score(y_B, B_oof)
    kC = cohen_kappa_score(y_C, C_oof)

    print(f"\n{'Metric':<22} {'A (Supervisory)':>16} {'B (Operator)':>14} {'C (Unsafe Acts)':>16}")
    print("-" * 72)
    print(f"{'Balanced Accuracy':<22} {bal_A:>16.2%} {bal_B:>14.2%} {bal_C:>16.2%}")
    print(f"{'Macro F1':<22} {f1_A:>16.4f} {f1_B:>14.4f} {f1_C:>16.4f}")
    print(f"{'Cohen Kappa':<22} {kA:>16.4f} {kB:>14.4f} {kC:>16.4f}")

    return model_1, model_2, model_3, datasplit


if __name__ == "__main__":
    trainPipeline()
