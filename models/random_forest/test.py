import os
import sys
import pickle

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.random_forest.train import (  # noqa: E402
    load_and_split, TARGET_A, TARGET_B, TARGET_C,
)


def main():
    FILEPATH    = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH  = os.path.join(os.path.dirname(__file__), "hfacs_rf.pkl")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model_1 = saved["model_1"]
    model_2 = saved["model_2"]
    model_3 = saved["model_3"]

    _, _, df_test = load_and_split(FILEPATH)

    true_A = df_test[TARGET_A].values
    true_B = df_test[TARGET_B].values
    true_C = df_test[TARGET_C].values

    input_1 = df_test[["Organizational Climate", "Employment, Total Weighted Avg CY_QoQ_pct"]]
    pred_A  = model_1.predict(input_1)

    input_2 = df_test[["WeatherCondition", "TimeOfDay", "SkyCondNonceil", "Personnel Conditions"]].copy()
    input_2["Supervisory Conditions"] = pred_A
    pred_B  = model_2.predict(input_2)

    input_3 = pd.DataFrame({
        "Supervisory Conditions": pred_A,
        "Operator Conditions":    pred_B,
    })
    pred_C  = model_3.predict(input_3)

    out = pd.DataFrame({
        "true_supervisory":  true_A,
        "pred_supervisory":  pred_A,
        "true_operator":     true_B,
        "pred_operator":     pred_B,
        "true_unsafe_acts":  true_C,
        "pred_unsafe_acts":  pred_C,
    })

    out_path = os.path.join(RESULTS_DIR, "rf_test_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"Test predictions saved to {out_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
