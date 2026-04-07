import os
import sys
import pickle
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.bayesian_net.train import load_and_split, run_inference  # noqa: E402


def main():
    FILEPATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hfacs_bn.pkl")
    OUT_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "results", "bn_test_predictions.csv")

    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]

    _, _, df_test, _ = load_and_split(FILEPATH)

    print(f"Test samples: {len(df_test)}")
    true_A, true_B, true_C, pred_A, pred_B, pred_C = run_inference(model, df_test)

    results = pd.DataFrame({
        "true_supervisory": true_A,
        "pred_supervisory": pred_A,
        "true_operator":    true_B,
        "pred_operator":    pred_B,
        "true_unsafe_acts": true_C,
        "pred_unsafe_acts": pred_C,
    })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    results.to_csv(OUT_PATH, index=False)
    print(f"Predictions saved to {OUT_PATH}")


if __name__ == "__main__":
    main()