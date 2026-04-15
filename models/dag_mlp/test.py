import os
import sys
import torch
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.real_dataloader import get_dataloaders              # noqa: E402
from models.dag_mlp.train import HFACSCausalDAGMLP, evaluate  # noqa: E402


def main():
    FILEPATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hfacs_dag_mlp.pt")
    OUT_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "results", "dag_mlp_test_predictions.csv")
    BATCH_SIZE = 32
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, encoders = get_dataloaders(FILEPATH, batch_size=BATCH_SIZE)

    n_A = len(encoders.enc_supervisory.classes_)
    n_B = len(encoders.enc_operator.classes_)
    n_C = len(encoders.enc_unsafe.classes_)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # Support both raw state_dict and val.py checkpoint dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd          = ckpt["state_dict"]
        hidden_size = ckpt.get("config", {}).get("hidden_size", None)
    else:
        sd          = ckpt
        hidden_size = None

    # Infer hidden_size from checkpoint weights if not stored in config
    if hidden_size is None:
        hidden_size = sd["mlp_a.0.weight"].shape[0]

    model = HFACSCausalDAGMLP(hidden_size=hidden_size, n_A=n_A, n_B=n_B, n_C=n_C).to(DEVICE)
    model.load_state_dict(sd)

    all_A, all_B, all_C, pred_A, pred_B, pred_C = evaluate(model, test_loader, DEVICE)

    results = pd.DataFrame({
        "true_supervisory": encoders.enc_supervisory.inverse_transform(all_A),
        "pred_supervisory": encoders.enc_supervisory.inverse_transform(pred_A),
        "true_operator":    encoders.enc_operator.inverse_transform(all_B),
        "pred_operator":    encoders.enc_operator.inverse_transform(pred_B),
        "true_unsafe_acts": encoders.enc_unsafe.inverse_transform(all_C),
        "pred_unsafe_acts": encoders.enc_unsafe.inverse_transform(pred_C),
    })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    results.to_csv(OUT_PATH, index=False)
    print(f"Test samples: {len(results)}")
    print(f"Predictions saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
