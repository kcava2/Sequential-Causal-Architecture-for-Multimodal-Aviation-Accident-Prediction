import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# ── Encoders ────────────────────────────────────────────────────────────────

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

# ── Dataset ──────────────────────────────────────────────────────────────────

class HFACSSequenceDataset(Dataset):
    """
    Each sample is a sequence of 3 LSTM steps: A → B → C
    following the Direct Parent Dependency rule.

    Step 0 input: [d: employment, e: environmental]          → predict A: Supervisory
    Step 1 input: [A_pred_embed, e: environmental, f: personnel] → predict B: Operator
    Step 2 input: [B_pred_embed, A_pred_embed]                → predict C: Unsafe Acts
    """
    def __init__(self, df, encoders: HFACSEncoders):
        e = encoders

        num = e.scaler.transform(df[["Employment Change vs Prior Period (%)",
                                     "Wind Conditions (kt)", "Temperature (C)"]]).astype("float32")

        light      = e.enc_light.transform(df["Light Conditions"])
        met        = e.enc_met.transform(df["Basic Meteorological Conditions"])
        personnel  = e.enc_personnel.transform(df["Personnel Conditions"])

        # Environmental = light + met + wind + temp (indices into num: wind=1, temp=2)
        env = torch.tensor(
            list(zip(light, met, num[:, 1], num[:, 2])), dtype=torch.float32)  # (N, 4)

        employment = torch.tensor(num[:, 0], dtype=torch.float32).unsqueeze(1)  # (N, 1)
        personnel_t = torch.tensor(personnel, dtype=torch.float32).unsqueeze(1)  # (N, 1)

        # Step 0: d + e  →  [employment(1) | env(4)]  dim=5
        self.step0 = torch.cat([employment, env], dim=1)  # (N, 5)

        # Step 1: e + f  →  [env(4) | personnel(1)]  dim=5
        self.step1 = torch.cat([env, personnel_t], dim=1)  # (N, 5)

        # Step 2: no extra inputs — LSTM hidden state carries A & B context
        self.step2 = torch.zeros(len(df), 5)              # (N, 5) placeholder

        # Targets
        self.y_A = torch.tensor(e.enc_supervisory.transform(df["Supervisory Conditions"]), dtype=torch.long)
        self.y_B = torch.tensor(e.enc_operator.transform(df["Operator Conditions"]),       dtype=torch.long)
        self.y_C = torch.tensor(e.enc_unsafe.transform(df["Unsafe Conditions"]),           dtype=torch.long)

    def __len__(self):
        return len(self.y_A)

    def __getitem__(self, idx):
        # sequence shape: (3, input_size)
        seq = torch.stack([self.step0[idx], self.step1[idx], self.step2[idx]])
        return seq, self.y_A[idx], self.y_B[idx], self.y_C[idx]

# ── Model ────────────────────────────────────────────────────────────────────

class HFACSCausalLSTM(nn.Module):
    """
    Single LSTM over 3 steps. Output at each step is passed to a
    classification head respecting the causal order A → B → C.
    """
    def __init__(self, input_size, hidden_size, n_A, n_B, n_C, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.head_A = nn.Linear(hidden_size, n_A)  # step 0 → Supervisory
        self.head_B = nn.Linear(hidden_size, n_B)  # step 1 → Operator
        self.head_C = nn.Linear(hidden_size, n_C)  # step 2 → Unsafe Acts

    def forward(self, x):
        # x: (batch, 3, input_size)
        out, _ = self.lstm(x)           # out: (batch, 3, hidden)
        out = self.drop(out)
        logits_A = self.head_A(out[:, 0, :])
        logits_B = self.head_B(out[:, 1, :])
        logits_C = self.head_C(out[:, 2, :])
        return logits_A, logits_B, logits_C

# ── Training loop ────────────────────────────────────────────────────────────

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for seq, y_A, y_B, y_C in loader:
        seq, y_A, y_B, y_C = seq.to(device), y_A.to(device), y_B.to(device), y_C.to(device)
        optimizer.zero_grad()
        logits_A, logits_B, logits_C = model(seq)
        loss = criterion(logits_A, y_A) + criterion(logits_B, y_B) + criterion(logits_C, y_C)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_A, all_B, all_C = [], [], []
    pred_A, pred_B, pred_C = [], [], []
    with torch.no_grad():
        for seq, y_A, y_B, y_C in loader:
            seq = seq.to(device)
            lA, lB, lC = model(seq)
            pred_A.extend(lA.argmax(1).cpu().tolist())
            pred_B.extend(lB.argmax(1).cpu().tolist())
            pred_C.extend(lC.argmax(1).cpu().tolist())
            all_A.extend(y_A.tolist())
            all_B.extend(y_B.tolist())
            all_C.extend(y_C.tolist())
    return all_A, all_B, all_C, pred_A, pred_B, pred_C

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    FILEPATH    = "HFACS_Simulated_Dataset.xlsx"
    HIDDEN_SIZE = 64
    BATCH_SIZE  = 32
    EPOCHS      = 20
    LR          = 1e-3
    TEST_SPLIT  = 0.2
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_excel(FILEPATH)

    # Train/test split
    n_test  = int(len(df) * TEST_SPLIT)
    df_train, df_test = df.iloc[:-n_test].reset_index(drop=True), df.iloc[-n_test:].reset_index(drop=True)

    encoders   = HFACSEncoders(df_train)
    train_set  = HFACSSequenceDataset(df_train, encoders)
    test_set   = HFACSSequenceDataset(df_test,  encoders)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

    n_A = len(encoders.enc_supervisory.classes_)   # 4
    n_B = len(encoders.enc_operator.classes_)       # 3
    n_C = len(encoders.enc_unsafe.classes_)         # 5

    model     = HFACSCausalLSTM(input_size=5, hidden_size=HIDDEN_SIZE,
                                 n_A=n_A, n_B=n_B, n_C=n_C).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_loader, optimizer, criterion, DEVICE)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {loss:.4f}")

    # Evaluation
    all_A, all_B, all_C, pred_A, pred_B, pred_C = evaluate(model, test_loader, DEVICE)

    print("\n── A: Supervisory Conditions ──")
    print(classification_report(all_A, pred_A, target_names=encoders.enc_supervisory.classes_))
    print("── B: Operator Conditions ──")
    print(classification_report(all_B, pred_B, target_names=encoders.enc_operator.classes_))
    print("── C: Unsafe Acts ──")
    print(classification_report(all_C, pred_C, target_names=encoders.enc_unsafe.classes_))

    torch.save(model.state_dict(), "hfacs_lstm.pt")
    print("\nModel saved to hfacs_lstm.pt")

if __name__ == "__main__":
    main()