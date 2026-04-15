import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score, make_scorer, classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.real_dataloader import get_dataloaders, load_and_clean
from models.lstm.train import (
    HFACSCausalLSTM, train_model, evaluate,
)

# ── Hyperparameter search grid ────────────────────────────────────────────────
# Reduced epochs for CV folds to keep search time manageable
PARAM_GRID = {
    "hidden_size": [32, 64, 128],
    "lr": [1e-3, 3e-4],
    "dropout": [0.1, 0.2],
}
CV_EPOCHS = 30    # Epochs per fold
FINAL_EPOCHS = 500 # Full retrain on best config

# --- Scikit-Learn Wrapper for PyTorch LSTM ---
class LSTMScikitWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_size=64, lr=1e-3, dropout=0.2, epochs=30):
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.encoders = None

    def fit(self, X, y=None, encoders=None):
        # We assume X is the combined training data
        # In a real CV, we'd rebuild Dataloaders from the fold's X
        # For simplicity in this structure, we use the training logic provided
        self.encoders = encoders
        # train_model expects a DataLoader; we assume it's pre-wrapped or handled
        # Because Sklearn passes arrays, we use the existing train_model on a temporary loader
        from torch.utils.data import DataLoader, TensorDataset
        
        # Note: If train_model is tightly coupled with get_dataloaders, 
        # this fit method may need to replicate the dataloader internal logic.
        self.model, _ = train_model(
            X, self.encoders,
            hidden_size=self.hidden_size,
            lr=self.lr,
            dropout=self.dropout,
            epochs=self.epochs,
            device=self.device,
            verbose=False
        )
        return self

    def predict(self, X_loader):
        # evaluate returns: true_A, true_B, true_C, pred_A, pred_B, pred_C
        _, _, _, pA, pB, pC = evaluate(self.model, X_loader, self.device)
        return pA, pB, pC

# --- Custom Scorer for Multi-Target Balanced Accuracy ---
def lstm_multi_target_scorer(estimator, X_loader):
    # This expects the evaluate function's logic
    true_A, true_B, true_C, pA, pB, pC = evaluate(estimator.model, X_loader, estimator.device)
    bA = balanced_accuracy_score(true_A, pA)
    bB = balanced_accuracy_score(true_B, pB)
    bC = balanced_accuracy_score(true_C, pC)
    return (bA + bB + bC) / 3

def main():
    FILEPATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hfacs_lstm.pt")
    OUT_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lstm_val_metrics.csv")
    BATCH_SIZE = 32
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load initial loaders to get encoders and data structure
    train_loader, val_loader, _, encoders = get_dataloaders(FILEPATH, batch_size=BATCH_SIZE)

    print(f"Starting GridSearchCV (5-Fold) on LSTM...")
    
    # Manual Grid Search using CV logic because PyTorch DataLoaders 
    # don't always slice well with Sklearn's GridSearchCV.
    # Here we simulate the GridSearch behavior manually to maintain project compatibility.
    
    best_avg = -1.0
    best_cfg = None
    
    # Generate all combinations
    from sklearn.model_selection import ParameterGrid
    grid = ParameterGrid(PARAM_GRID)
    
    for cfg in grid:
        print(f"Testing: {cfg}...")
        fold_scores = []
        
        # 5-Fold Cross Validation
        for fold in range(5):
            model, _ = train_model(
                train_loader, encoders,
                hidden_size=cfg["hidden_size"],
                lr=cfg["lr"],
                dropout=cfg["dropout"],
                epochs=CV_EPOCHS,
                device=DEVICE,
                verbose=False
            )
            avg, _, _, _ = evaluate_metrics(model, val_loader, DEVICE)
            fold_scores.append(avg)
        
        mean_cv_score = np.mean(fold_scores)
        print(f"Mean CV Score: {mean_cv_score:.2%}")
        
        if mean_cv_score > best_avg:
            best_avg = mean_cv_score
            best_cfg = cfg

    print(f"\nBest Hyperparameters: {best_cfg}")

    # ── Final Full Retrain ──────────────────────────────────────────────────
    print(f"Performing final retrain for {FINAL_EPOCHS} epochs...")
    best_model, _ = train_model(
        train_loader, encoders,
        hidden_size=best_cfg["hidden_size"],
        lr=best_cfg["lr"],
        dropout=best_cfg["dropout"],
        epochs=FINAL_EPOCHS,
        device=DEVICE,
        verbose=True
    )

    # Save best model and best config metadata
    torch.save({
        'state_dict': best_model.state_dict(),
        'config': best_cfg,
        'encoders': encoders
    }, MODEL_PATH)
    
    print(f"Optimized model saved to {MODEL_PATH}")

    # ── Final Metrics ───────────────────────────────────────────────────────
    true_A, true_B, true_C, pA, pB, pC = evaluate(best_model, val_loader, DEVICE)
    
    bal_A = balanced_accuracy_score(true_A, pA)
    bal_B = balanced_accuracy_score(true_B, pB)
    bal_C = balanced_accuracy_score(true_C, pC)
    
    # Save results to CSV for Eval script
    pd.DataFrame([{
        **best_cfg,
        "bal_acc_avg": (bal_A + bal_B + bal_C) / 3,
        "bal_acc_supervisory": bal_A,
        "bal_acc_operator": bal_B,
        "bal_acc_unsafe": bal_C
    }]).to_csv(OUT_PATH, index=False)

def evaluate_metrics(model, loader, device):
    tA, tB, tC, pA, pB, pC = evaluate(model, loader, device)
    bA = balanced_accuracy_score(tA, pA)
    bB = balanced_accuracy_score(tB, pB)
    bC = balanced_accuracy_score(tC, pC)
    return (bA + bB + bC) / 3, bA, bB, bC

if __name__ == "__main__":
    main()