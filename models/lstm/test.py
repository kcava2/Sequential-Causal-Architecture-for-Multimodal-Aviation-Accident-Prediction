import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Encoders ─────────────────────────────────────────────────────────────────

class HFACSEncoders:
    def __init__(self, df):
        self.scaler = StandardScaler()
        self.enc_supervisory  = LabelEncoder()
        self.enc_operator     = LabelEncoder()
        self.enc_unsafe       = LabelEncoder()
        self.enc_light        = LabelEncoder()
        self.enc_met          = LabelEncoder()
        self.enc_personnel    = LabelEncoder()

        self.scaler.fit(df[["Employment Change vs Prior Period (%)",
                             "Wind Conditions (kt)", "Temperature (C)"]])
        self.enc_light.fit(df["Light Conditions"])
        self.enc_met.fit(df["Basic Meteorological Conditions"])
        self.enc_personnel.fit(df["Personnel Conditions"])
        self.enc_supervisory.fit(df["Supervisory Conditions"])
        self.enc_operator.fit(df["Operator Conditions"])
        self.enc_unsafe.fit(df["Unsafe Conditions"])

# ── Dataset ───────────────────────────────────────────────────────────────────

class HFACSSequenceDataset(Dataset):
    def __init__(self, df, encoders: HFACSEncoders):
        e = encoders

        num = e.scaler.transform(df[["Employment Change vs Prior Period (%)",
                                     "Wind Conditions (kt)", "Temperature (C)"]]).astype("float32")

        light      = e.enc_light.transform(df["Light Conditions"])
        met        = e.enc_met.transform(df["Basic Meteorological Conditions"])
        personnel  = e.enc_personnel.transform(df["Personnel Conditions"])

        env = torch.tensor(
            list(zip(light, met, num[:, 1], num[:, 2])), dtype=torch.float32)

        employment  = torch.tensor(num[:, 0], dtype=torch.float32).unsqueeze(1)
        personnel_t = torch.tensor(personnel, dtype=torch.float32).unsqueeze(1)

        self.step0 = torch.cat([employment, env], dim=1)   # (N, 5)
        self.step1 = torch.cat([env, personnel_t], dim=1)  # (N, 5)

        self.y_A = torch.tensor(e.enc_supervisory.transform(df["Supervisory Conditions"]), dtype=torch.long)
        self.y_B = torch.tensor(e.enc_operator.transform(df["Operator Conditions"]),       dtype=torch.long)
        self.y_C = torch.tensor(e.enc_unsafe.transform(df["Unsafe Conditions"]),           dtype=torch.long)

    def __len__(self):
        return len(self.y_A)

    def __getitem__(self, idx):
        seq = torch.stack([self.step0[idx], self.step1[idx]])
        return seq, self.y_A[idx], self.y_B[idx], self.y_C[idx]

# ── Model ─────────────────────────────────────────────────────────────────────

class HFACSCausalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_A, n_B, n_C, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell   = nn.LSTMCell(input_size, hidden_size)
        self.drop        = nn.Dropout(dropout)
        self.embed_proj  = nn.Linear(n_A + n_B, input_size)

        self.head_A = nn.Linear(hidden_size, n_A)
        self.head_B = nn.Linear(hidden_size, n_B)
        self.head_C = nn.Linear(hidden_size, n_C)

    def forward(self, x):
        batch = x.size(0)
        h = torch.zeros(batch, self.hidden_size, device=x.device)
        c = torch.zeros(batch, self.hidden_size, device=x.device)

        h, c    = self.lstm_cell(x[:, 0, :], (h, c))
        logits_A = self.head_A(self.drop(h))
        soft_A   = torch.softmax(logits_A, dim=1)

        h, c    = self.lstm_cell(x[:, 1, :], (h, c))
        logits_B = self.head_B(self.drop(h))
        soft_B   = torch.softmax(logits_B, dim=1)

        step2_input = self.embed_proj(torch.cat([soft_A, soft_B], dim=1))
        h, c    = self.lstm_cell(step2_input, (h, c))
        logits_C = self.head_C(self.drop(h))

        return logits_A, logits_B, logits_C

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    FILEPATH    = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Simulated_Dataset.xlsx")
    MODEL_PATH  = os.path.join(os.path.dirname(__file__), "hfacs_lstm.pt")
    OUT_PATH    = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lstm_test_predictions.csv")
    HIDDEN_SIZE = 64
    BATCH_SIZE  = 32
    TEST_SPLIT  = 0.2
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_excel(FILEPATH)

    n_test   = int(len(df) * TEST_SPLIT)
    df_train = df.iloc[:-n_test].reset_index(drop=True)
    df_test  = df.iloc[-n_test:].reset_index(drop=True)

    # Fit encoders on training data only
    encoders = HFACSEncoders(df_train)
    test_set = HFACSSequenceDataset(df_test, encoders)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    n_A = len(encoders.enc_supervisory.classes_)
    n_B = len(encoders.enc_operator.classes_)
    n_C = len(encoders.enc_unsafe.classes_)

    model = HFACSCausalLSTM(input_size=5, hidden_size=HIDDEN_SIZE,
                             n_A=n_A, n_B=n_B, n_C=n_C).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    true_A, true_B, true_C = [], [], []
    pred_A, pred_B, pred_C = [], [], []

    with torch.no_grad():
        for seq, y_A, y_B, y_C in test_loader:
            seq = seq.to(DEVICE)
            lA, lB, lC = model(seq)
            pred_A.extend(lA.argmax(1).cpu().tolist())
            pred_B.extend(lB.argmax(1).cpu().tolist())
            pred_C.extend(lC.argmax(1).cpu().tolist())
            true_A.extend(y_A.tolist())
            true_B.extend(y_B.tolist())
            true_C.extend(y_C.tolist())

    # Decode integer labels back to class names
    results = pd.DataFrame({
        "true_supervisory": encoders.enc_supervisory.inverse_transform(true_A),
        "pred_supervisory": encoders.enc_supervisory.inverse_transform(pred_A),
        "true_operator":    encoders.enc_operator.inverse_transform(true_B),
        "pred_operator":    encoders.enc_operator.inverse_transform(pred_B),
        "true_unsafe_acts": encoders.enc_unsafe.inverse_transform(true_C),
        "pred_unsafe_acts": encoders.enc_unsafe.inverse_transform(pred_C),
    })

    results.to_csv(OUT_PATH, index=False)
    print(f"Predictions saved to {OUT_PATH}")
    print(f"Test samples: {len(results)}")

if __name__ == "__main__":
    main()
