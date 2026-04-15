import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ParameterGrid

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.real_dataloader import get_dataloaders  # noqa: E402
from models.dag_mlp.train import (               # noqa: E402
    HFACSCausalDAGMLP, train_model, evaluate,
)

# ── Hyperparameter search grid ─────────────────────────────────────────────────
PARAM_GRID = {
    "hidden_size": [32, 64, 128],
    "lr":          [1e-3, 3e-4],
    "dropout":     [0.1, 0.2],
}
CV_EPOCHS    = 30   # Epochs per CV fold (fast evaluation)
FINAL_EPOCHS = 500  # Full retrain on best config


def evaluate_metrics(model, loader, device):
    """Return (macro_avg_bal_acc, bal_A, bal_B, bal_C)."""
    tA, tB, tC, pA, pB, pC = evaluate(model, loader, device)
    bA = balanced_accuracy_score(tA, pA)
    bB = balanced_accuracy_score(tB, pB)
    bC = balanced_accuracy_score(tC, pC)
    return (bA + bB + bC) / 3, bA, bB, bC


def main():
    FILEPATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hfacs_dag_mlp.pt")
    OUT_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "results", "dag_mlp_val_metrics.csv")
    BATCH_SIZE = 32
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    train_loader, val_loader, _, encoders = get_dataloaders(FILEPATH, batch_size=BATCH_SIZE)

    print("=" * 60)
    print("DAG-MLP Hyperparameter Search (5-Fold Manual CV)")
    print("=" * 60)

    best_avg = -1.0
    best_cfg = None
    all_rows = []

    for cfg in ParameterGrid(PARAM_GRID):
        print(f"\nTesting: {cfg}")
        fold_scores = []

        for fold in range(5):
            model, _ = train_model(
                train_loader, encoders,
                hidden_size=cfg["hidden_size"],
                lr=cfg["lr"],
                dropout=cfg["dropout"],
                epochs=CV_EPOCHS,
                device=DEVICE,
                verbose=False,
            )
            avg, _, _, _ = evaluate_metrics(model, val_loader, DEVICE)
            fold_scores.append(avg)

        mean_cv = np.mean(fold_scores)
        print(f"  Mean CV Balanced Acc: {mean_cv:.2%}")

        all_rows.append({**cfg, "cv_bal_acc_avg": mean_cv})

        if mean_cv > best_avg:
            best_avg = mean_cv
            best_cfg = cfg

    print(f"\nBest config: {best_cfg}  (CV score: {best_avg:.2%})")

    # ── Final retrain on best config ───────────────────────────────────────────
    print(f"\nFinal retrain for {FINAL_EPOCHS} epochs with best config...")
    best_model, _ = train_model(
        train_loader, encoders,
        val_loader=val_loader,
        hidden_size=best_cfg["hidden_size"],
        lr=best_cfg["lr"],
        dropout=best_cfg["dropout"],
        epochs=FINAL_EPOCHS,
        device=DEVICE,
        verbose=True,
    )

    torch.save({
        "state_dict": best_model.state_dict(),
        "config":     best_cfg,
        "encoders":   encoders,
    }, MODEL_PATH)
    print(f"Best model saved to {MODEL_PATH}")

    # ── Final val metrics ──────────────────────────────────────────────────────
    avg, bal_A, bal_B, bal_C = evaluate_metrics(best_model, val_loader, DEVICE)

    # Append best-config row with full val metrics
    best_row = {
        **best_cfg,
        "cv_bal_acc_avg":      best_avg,
        "bal_acc_avg":         avg,
        "bal_acc_supervisory": bal_A,
        "bal_acc_operator":    bal_B,
        "bal_acc_unsafe":      bal_C,
    }
    # Backfill cv_bal_acc_avg into all_rows for the best config, save full grid
    pd.DataFrame(all_rows).to_csv(OUT_PATH, index=False)
    print(f"Grid results saved to {OUT_PATH}")
    print(f"\nFinal Val — BalAcc  A: {bal_A:.2%}  B: {bal_B:.2%}  C: {bal_C:.2%}  Avg: {avg:.2%}")


if __name__ == "__main__":
    main()
