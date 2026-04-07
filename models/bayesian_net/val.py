import os
import sys
import pickle
import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.bayesian_net.train import load_and_split, run_inference  # noqa: E402


def main():
    FILEPATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hfacs_bn.pkl")
    OUT_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "results", "bn_val_metrics.csv")

    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]

    _, df_val, _, _ = load_and_split(FILEPATH)

    print(f"Validation samples: {len(df_val)}")
    true_A, true_B, true_C, pred_A, pred_B, pred_C = run_inference(model, df_val)

    bal_A = balanced_accuracy_score(true_A, pred_A)
    bal_B = balanced_accuracy_score(true_B, pred_B)
    bal_C = balanced_accuracy_score(true_C, pred_C)

    print(f"Balanced Acc A (Supervisory): {bal_A:.2%}")
    print(f"Balanced Acc B (Operator)   : {bal_B:.2%}")
    print(f"Balanced Acc C (Unsafe Acts): {bal_C:.2%}")
    print(f"Average                     : {(bal_A + bal_B + bal_C) / 3:.2%}")
    print()

    print("── A: Supervisory Conditions ──")
    print(classification_report(true_A, pred_A, zero_division=0))
    print("── B: Operator Conditions ──")
    print(classification_report(true_B, pred_B, zero_division=0))
    print("── C: Unsafe Acts ──")
    print(classification_report(true_C, pred_C, zero_division=0))

    metrics = pd.DataFrame([{
        "bal_acc_supervisory": bal_A,
        "bal_acc_operator":    bal_B,
        "bal_acc_unsafe_acts": bal_C,
        "bal_acc_avg":         (bal_A + bal_B + bal_C) / 3,
        "n_samples":           len(df_val),
    }])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    metrics.to_csv(OUT_PATH, index=False)
    print(f"Metrics saved to {OUT_PATH}")


if __name__ == "__main__":
    main()