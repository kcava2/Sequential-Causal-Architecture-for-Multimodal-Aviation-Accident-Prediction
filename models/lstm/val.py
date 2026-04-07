import os
import sys
import torch
import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.real_dataloader import get_dataloaders          # noqa: E402
from models.lstm.train import HFACSCausalLSTM, evaluate   # noqa: E402


def main():
    FILEPATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hfacs_lstm.pt")
    OUT_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lstm_val_metrics.csv")
    HIDDEN_SIZE = 64
    BATCH_SIZE  = 32
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, _, encoders = get_dataloaders(FILEPATH, batch_size=BATCH_SIZE)

    n_A = len(encoders.enc_supervisory.classes_)
    n_B = len(encoders.enc_operator.classes_)
    n_C = len(encoders.enc_unsafe.classes_)

    model = HFACSCausalLSTM(hidden_size=HIDDEN_SIZE, n_A=n_A, n_B=n_B, n_C=n_C).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

    all_A, all_B, all_C, pred_A, pred_B, pred_C = evaluate(model, val_loader, DEVICE)

    bal_A = balanced_accuracy_score(all_A, pred_A)
    bal_B = balanced_accuracy_score(all_B, pred_B)
    bal_C = balanced_accuracy_score(all_C, pred_C)

    print(f"Validation samples : {len(all_B)}")
    print(f"Balanced Acc A (Supervisory): {bal_A:.2%}")
    print(f"Balanced Acc B (Operator)   : {bal_B:.2%}")
    print(f"Balanced Acc C (Unsafe Acts): {bal_C:.2%}")
    print(f"Average                     : {(bal_A + bal_B + bal_C) / 3:.2%}")
    print()
    print("── A: Supervisory Conditions ──")
    print(classification_report(all_A, pred_A, target_names=encoders.enc_supervisory.classes_, zero_division=0))
    print("── B: Operator Conditions ──")
    print(classification_report(all_B, pred_B, target_names=encoders.enc_operator.classes_, zero_division=0))
    print("── C: Unsafe Acts ──")
    print(classification_report(all_C, pred_C, target_names=encoders.enc_unsafe.classes_, zero_division=0))

    metrics = pd.DataFrame([{
        "bal_acc_supervisory":  bal_A,
        "bal_acc_operator":     bal_B,
        "bal_acc_unsafe_acts":  bal_C,
        "bal_acc_avg":          (bal_A + bal_B + bal_C) / 3,
        "n_samples":            len(all_B),
    }])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    metrics.to_csv(OUT_PATH, index=False)
    print(f"Metrics saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
