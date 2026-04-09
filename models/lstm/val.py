import os
import sys
import itertools

import torch
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.real_dataloader import get_dataloaders          # noqa: E402
from models.lstm.train import (                           # noqa: E402
    HFACSCausalLSTM, train_model, evaluate,
)

# ── Hyperparameter search grid ────────────────────────────────────────────────
# Each config is trained for N_SEARCH_EPOCHS; the best is then fully retrained.
SEARCH_GRID = {
    "hidden_size": [32, 64, 128],
    "lr":          [1e-3, 3e-4, 1e-4],
    "dropout":     [0.1, 0.2],
}
N_SEARCH_EPOCHS = 30   # short runs for candidate ranking
N_FULL_EPOCHS   = 500  # full run for the winning config


def _val_avg(model, val_loader, device):
    """Return mean balanced accuracy across A/B/C on the val set."""
    all_A, all_B, all_C, pred_A, pred_B, pred_C = evaluate(model, val_loader, device)
    bA = balanced_accuracy_score(all_A, pred_A)
    bB = balanced_accuracy_score(all_B, pred_B)
    bC = balanced_accuracy_score(all_C, pred_C)
    return (bA + bB + bC) / 3, bA, bB, bC


def main():
    FILEPATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hfacs_lstm.pt")
    OUT_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lstm_val_metrics.csv")
    BATCH_SIZE = 32
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, encoders = get_dataloaders(FILEPATH, batch_size=BATCH_SIZE)

    # ── Hyperparameter search ─────────────────────────────────────────────────
    keys   = list(SEARCH_GRID.keys())
    combos = list(itertools.product(*[SEARCH_GRID[k] for k in keys]))
    n_total = len(combos)

    print(f"Searching {n_total} configs × {N_SEARCH_EPOCHS} epochs each "
          f"({n_total * N_SEARCH_EPOCHS} total epochs for search)\n")
    print(f"{'#':>3}  {'hidden':>6}  {'lr':>8}  {'dropout':>7}  {'val avg':>8}")
    print("-" * 42)

    best_avg   = -1.0
    best_cfg   = None
    search_rows = []

    for i, values in enumerate(combos, 1):
        cfg = dict(zip(keys, values))

        model, _ = train_model(
            train_loader, encoders,
            hidden_size=cfg["hidden_size"],
            lr=cfg["lr"],
            dropout=cfg["dropout"],
            epochs=N_SEARCH_EPOCHS,
            device=DEVICE,
            verbose=False,
        )

        avg, bA, bB, bC = _val_avg(model, val_loader, DEVICE)
        marker = " ◄" if avg > best_avg else ""
        print(f"{i:>3}  {cfg['hidden_size']:>6}  {cfg['lr']:>8.0e}  "
              f"{cfg['dropout']:>7}  {avg:>8.2%}{marker}")

        search_rows.append({**cfg, "val_bal_acc_avg": avg,
                            "val_bal_acc_A": bA, "val_bal_acc_B": bB, "val_bal_acc_C": bC})

        if avg > best_avg:
            best_avg = avg
            best_cfg = cfg

    print(f"\nBest config: {best_cfg}  (val avg = {best_avg:.2%})")

    # ── Full retrain with best hyperparams ────────────────────────────────────
    print(f"\nFull retrain: {N_FULL_EPOCHS} epochs with best config...")
    best_model, _ = train_model(
        train_loader, encoders,
        hidden_size=best_cfg["hidden_size"],
        lr=best_cfg["lr"],
        dropout=best_cfg["dropout"],
        epochs=N_FULL_EPOCHS,
        device=DEVICE,
        verbose=True,
    )

    torch.save(best_model.state_dict(), MODEL_PATH)
    print(f"Best model saved to {MODEL_PATH}")

    # ── Final val metrics ─────────────────────────────────────────────────────
    all_A, all_B, all_C, pred_A, pred_B, pred_C = evaluate(best_model, val_loader, DEVICE)

    bal_A = balanced_accuracy_score(all_A, pred_A)
    bal_B = balanced_accuracy_score(all_B, pred_B)
    bal_C = balanced_accuracy_score(all_C, pred_C)
    avg   = (bal_A + bal_B + bal_C) / 3

    print(f"\nValidation samples : {len(all_B)}")
    print(f"Balanced Acc A (Supervisory): {bal_A:.2%}")
    print(f"Balanced Acc B (Operator)   : {bal_B:.2%}")
    print(f"Balanced Acc C (Unsafe Acts): {bal_C:.2%}")
    print(f"Average                     : {avg:.2%}")
    print()
    print("── A: Supervisory Conditions ──")
    print(classification_report(all_A, pred_A,
                                target_names=encoders.enc_supervisory.classes_, zero_division=0))
    print("── B: Operator Conditions ──")
    print(classification_report(all_B, pred_B,
                                target_names=encoders.enc_operator.classes_, zero_division=0))
    print("── C: Unsafe Acts ──")
    print(classification_report(all_C, pred_C,
                                target_names=encoders.enc_unsafe.classes_, zero_division=0))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pd.DataFrame([{
        "best_hidden_size":    best_cfg["hidden_size"],
        "best_lr":             best_cfg["lr"],
        "best_dropout":        best_cfg["dropout"],
        "bal_acc_supervisory": bal_A,
        "bal_acc_operator":    bal_B,
        "bal_acc_unsafe_acts": bal_C,
        "bal_acc_avg":         avg,
        "n_samples":           len(all_B),
    }]).to_csv(OUT_PATH, index=False)
    print(f"Val metrics saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
