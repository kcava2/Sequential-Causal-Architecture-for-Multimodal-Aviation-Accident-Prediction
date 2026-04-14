"""
eval_utils.py — Shared utilities for evaluation across all three models.

Usage in each eval.py:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.eval_utils import (
        TASK_COLORS, LABEL_MAP, clean_feature_name, apply_plot_style,
        CONFUSION_FIGSIZE, SINGLE_FIGSIZE, ROC_FIGSIZE,
        compute_roc_auc, plot_roc_curves, plot_sensitivity_bars,
    )
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ── Color scheme (consistent across all model eval scripts) ──────────────────
TASK_COLORS = {
    "Supervisory": "steelblue",
    "Operator":    "darkorange",
    "Unsafe Acts": "green",
}

# ── Figure sizes ──────────────────────────────────────────────────────────────
CONFUSION_FIGSIZE = (16, 5)
SINGLE_FIGSIZE    = (10, 5)
ROC_FIGSIZE       = (16, 5)

# ── Feature label cleaning ────────────────────────────────────────────────────
LABEL_MAP = {
    # RF raw column names (long)
    "Organizational Climate":                     "Org. Climate",
    "Employment, Total Weighted Avg CY_QoQ_pct":  "Employment QoQ%",
    "WeatherCondition":                           "Weather",
    "TimeOfDay":                                  "Time of Day",
    "SkyCondNonceil":                             "Sky Cond.",
    "Personnel Conditions":                       "Personnel",
    "Supervisory Conditions":                     "Supervisory",
    "Operator Conditions":                        "Operator",
    "Unsafe Conditions":                          "Unsafe Acts",
    # BN short node names
    "OrgClimate":  "Org. Climate",
    "Employment":  "Employment",
    "Weather":     "Weather",
    "Personnel":   "Personnel",
    "Supervisory": "Supervisory",
    "Operator":    "Operator",
    "UnsafeActs":  "Unsafe Acts",
}


def clean_feature_name(raw: str, max_len: int = 25) -> str:
    """
    Produce a display-friendly feature name.

    Steps:
    1. Strip ColumnTransformer prefixes ("num__" / "cat__").
    2. For one-hot encoded names ("ColName_ClassValue"), split on the
       FIRST space-or-underscore boundary that separates a known column
       name from a class value, clean the column portion, then rejoin.
    3. Fall back to a direct LABEL_MAP lookup.
    4. Truncate to max_len characters.

    Examples
    --------
    "cat__WeatherCondition_VMC"                         → "Weather_VMC"
    "num__Employment, Total Weighted Avg CY_QoQ_pct"   → "Employment QoQ%"
    "cat__Organizational Climate_High Risk"             → "Org. Climate_High Risk"
    "OrgClimate"                                        → "Org. Climate"
    """
    # 1. Strip sklearn ColumnTransformer prefix
    stripped_cat = False
    for prefix in ("num__", "cat__"):
        if raw.startswith(prefix):
            stripped_cat = (prefix == "cat__")
            raw = raw[len(prefix):]
            break

    # 2. For one-hot encoded cat features: find the column name by checking
    #    progressively longer prefixes of `raw` against LABEL_MAP.
    if stripped_cat and "_" in raw:
        # Try matching the longest LABEL_MAP key that is a prefix of `raw`
        best_key = None
        for key in LABEL_MAP:
            # The separator between ColName and ClassValue is either "_" or " "
            if raw.startswith(key + "_") or raw.startswith(key + " "):
                if best_key is None or len(key) > len(best_key):
                    best_key = key
        if best_key is not None:
            sep_char = raw[len(best_key)]          # "_" or " "
            suffix   = raw[len(best_key) + 1:]     # the class-value portion
            raw      = f"{LABEL_MAP[best_key]}{sep_char}{suffix}"

    # 3. Direct lookup (numeric features, BN short names, etc.)
    if raw in LABEL_MAP:
        raw = LABEL_MAP[raw]

    # 4. Truncate
    if len(raw) > max_len:
        raw = raw[:max_len - 1] + "…"

    return raw


def apply_plot_style() -> None:
    """Apply project-wide matplotlib rcParams for consistent figures."""
    matplotlib.rcParams.update({
        "figure.dpi":      150,
        "font.size":       10,
        "axes.titlesize":  12,
        "axes.labelsize":  10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


# ── ROC / AUC helpers ─────────────────────────────────────────────────────────

def compute_roc_auc(probs, true_labels, classes):
    """
    Compute per-class OvR ROC curves and macro-averaged AUC for one task.

    Parameters
    ----------
    probs       : (N, n_classes) float array of predicted probabilities
    true_labels : (N,) array of true labels (same dtype/encoding as classes)
    classes     : ordered array of class labels (same order as probs columns)

    Returns
    -------
    fpr_dict  : {class_label: fpr array}
    tpr_dict  : {class_label: tpr array}
    auc_dict  : {class_label: float AUC}
    macro_auc : float — mean of per-class AUCs
    """
    y_bin = label_binarize(true_labels, classes=classes)
    if y_bin.ndim == 1:
        # Binary edge case: label_binarize returns (N,1) for 2 classes
        y_bin = np.column_stack([1 - y_bin, y_bin])

    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        fpr_dict[cls] = fpr
        tpr_dict[cls] = tpr
        auc_dict[cls] = float(auc(fpr, tpr))

    macro_auc = float(np.mean(list(auc_dict.values())))
    return fpr_dict, tpr_dict, auc_dict, macro_auc


# Map internal task keys (BN uses "UnsafeActs") to display names
_TASK_DISPLAY = {
    "Supervisory": "Supervisory",
    "Operator":    "Operator",
    "Unsafe Acts": "Unsafe Acts",
    "UnsafeActs":  "Unsafe Acts",
}


def plot_roc_curves(task_data, title_prefix, save_path, csv_path=None):
    """
    Draw a 1×3 subplot of macro-average ROC curves, one per task.

    Parameters
    ----------
    task_data    : list of (task_label, probs, true_labels, classes)
    title_prefix : string prefix for the figure suptitle
    save_path    : full .png path
    csv_path     : optional .csv path for per-class AUC values
    """
    apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=ROC_FIGSIZE)
    csv_rows  = []

    for ax, (task_label, probs, true_labels, classes) in zip(axes, task_data):
        fpr_dict, tpr_dict, auc_dict, macro_auc = compute_roc_auc(
            probs, true_labels, classes
        )
        display = _TASK_DISPLAY.get(task_label, task_label)
        color   = TASK_COLORS.get(display, "gray")

        # Per-class curves in light gray
        for cls in classes:
            ax.plot(fpr_dict[cls], tpr_dict[cls],
                    color="lightgray", linewidth=0.8, alpha=0.7)
            csv_rows.append({
                "task": display, "class": str(cls),
                "auc":  round(auc_dict[cls], 4),
            })

        # Macro-average curve on a common FPR grid
        mean_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(mean_fpr)
        for cls in classes:
            mean_tpr += np.interp(mean_fpr, fpr_dict[cls], tpr_dict[cls])
        mean_tpr /= len(classes)

        ax.plot(mean_fpr, mean_tpr, color=color, linewidth=2,
                label=f"Macro AUC = {macro_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{display} Conditions")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

        csv_rows.append({
            "task": display, "class": "macro_avg",
            "auc":  round(macro_auc, 4),
        })

    fig.suptitle(f"{title_prefix} — ROC Curves (One-vs-Rest)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"ROC curves saved to {save_path}")

    if csv_path:
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"ROC AUC CSV saved to {csv_path}")


# ── Sensitivity grouped bar-chart helper ─────────────────────────────────────

def plot_sensitivity_bars(
    abl_df,
    feature_col,
    feature_order,
    title,
    save_path,
    task_col="task",
    metric_col="acc_drop",
    mean_col=None,
    err_col=None,
    tasks=("Supervisory", "Operator", "Unsafe Acts"),
):
    """
    Grouped bar chart of balanced accuracy drop per ablated feature.

    Parameters
    ----------
    abl_df        : DataFrame with at least [feature_col, task_col, metric_col or mean_col]
    feature_col   : column name for the ablated feature (x-axis groups)
    feature_order : ordered list of feature values to plot
    title         : axes title string
    save_path     : full .png path
    task_col      : column name for the task label
    metric_col    : column used as bar height when mean_col is None
    mean_col      : column used as bar height (mutually exclusive with metric_col)
    err_col       : column used for error bar half-widths (optional)
    tasks         : ordered tuple of task label strings
    """
    apply_plot_style()
    x     = np.arange(len(feature_order))
    width = 0.25

    _, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    for i, task in enumerate(tasks):
        color    = TASK_COLORS.get(task, "gray")
        task_sub = abl_df[abl_df[task_col] == task].set_index(feature_col)

        heights, errors = [], []
        for feat in feature_order:
            if feat in task_sub.index:
                row = task_sub.loc[feat]
                # .loc can return a Series if the feature appears multiple times;
                # take the first value in that case.
                h = float(row[mean_col].iloc[0] if hasattr(row[mean_col], "iloc")
                          else row[mean_col]) if mean_col else \
                    float(row[metric_col].iloc[0] if hasattr(row[metric_col], "iloc")
                          else row[metric_col])
                e = float(row[err_col].iloc[0] if hasattr(row[err_col], "iloc")
                          else row[err_col]) if err_col else 0.0
            else:
                h, e = 0.0, 0.0
            heights.append(h)
            errors.append(e)

        ax.bar(x + i * width, heights, width, label=task, color=color)
        if err_col:
            ax.errorbar(
                x + i * width, heights, yerr=errors,
                fmt="none", color="black", capsize=3, linewidth=1,
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    clean_labels = [clean_feature_name(f) for f in feature_order]
    ax.set_xticks(x + width)
    ax.set_xticklabels(clean_labels, rotation=30, ha="right")
    ax.set_ylabel("Balanced Accuracy Drop")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close("all")
    print(f"Sensitivity plot saved to {save_path}")
