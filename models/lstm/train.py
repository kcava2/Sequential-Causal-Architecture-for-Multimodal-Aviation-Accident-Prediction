import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
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

        # Targets
        self.y_A = torch.tensor(e.enc_supervisory.transform(df["Supervisory Conditions"]), dtype=torch.long)
        self.y_B = torch.tensor(e.enc_operator.transform(df["Operator Conditions"]),       dtype=torch.long)
        self.y_C = torch.tensor(e.enc_unsafe.transform(df["Unsafe Conditions"]),           dtype=torch.long)

    def __len__(self):
        return len(self.y_A)

    def __getitem__(self, idx):
        # sequence shape: (2, input_size) — step2 is computed dynamically in the model
        seq = torch.stack([self.step0[idx], self.step1[idx]])
        return seq, self.y_A[idx], self.y_B[idx], self.y_C[idx]

# ── Model ────────────────────────────────────────────────────────────────────

class HFACSCausalLSTM(nn.Module):
    """
    Step-by-step LSTM respecting the causal order A → B → C.

    Step 0: features → predict A (Supervisory)
    Step 1: features → predict B (Operator)
    Step 2: softmax(A) + softmax(B) projected to input_size → predict C (Unsafe Acts)

    The soft predictions of A and B are fed as real inputs at step 2,
    encoding the causal dependency instead of using placeholder zeros.
    """
    def __init__(self, input_size, hidden_size, n_A, n_B, n_C, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.embed_proj = nn.Linear(n_A + n_B, input_size)  # project A+B soft preds → input_size

        self.head_A = nn.Linear(hidden_size, n_A)  # step 0 → Supervisory
        self.head_B = nn.Linear(hidden_size, n_B)  # step 1 → Operator
        self.head_C = nn.Linear(hidden_size, n_C)  # step 2 → Unsafe Acts

    def forward(self, x):
        # x: (batch, 2, input_size)
        batch = x.size(0)
        h = torch.zeros(batch, self.hidden_size, device=x.device)
        c = torch.zeros(batch, self.hidden_size, device=x.device)

        # Step 0 → predict A (Supervisory)
        h, c = self.lstm_cell(x[:, 0, :], (h, c))
        logits_A = self.head_A(self.drop(h))
        soft_A = torch.softmax(logits_A, dim=1)

        # Step 1 → predict B (Operator)
        h, c = self.lstm_cell(x[:, 1, :], (h, c))
        logits_B = self.head_B(self.drop(h))
        soft_B = torch.softmax(logits_B, dim=1)

        # Step 2: soft A + soft B as input → predict C (Unsafe Acts)
        step2_input = self.embed_proj(torch.cat([soft_A, soft_B], dim=1))
        h, c = self.lstm_cell(step2_input, (h, c))
        logits_C = self.head_C(self.drop(h))

        return logits_A, logits_B, logits_C

# ── Training loop ────────────────────────────────────────────────────────────

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_A = correct_B = correct_C = total = 0
    for seq, y_A, y_B, y_C in loader:
        seq, y_A, y_B, y_C = seq.to(device), y_A.to(device), y_B.to(device), y_C.to(device)
        optimizer.zero_grad()
        logits_A, logits_B, logits_C = model(seq)
        loss = criterion(logits_A, y_A) + criterion(logits_B, y_B) + criterion(logits_C, y_C)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_A += (logits_A.argmax(1) == y_A).sum().item()
        correct_B += (logits_B.argmax(1) == y_B).sum().item()
        correct_C += (logits_C.argmax(1) == y_C).sum().item()
        total += len(y_A)
    avg_acc = (correct_A + correct_B + correct_C) / (3 * total)
    return total_loss / len(loader), correct_A / total, correct_B / total, correct_C / total, avg_acc

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
    FILEPATH    = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Simulated_Dataset.xlsx")
    HIDDEN_SIZE = 64
    BATCH_SIZE  = 32
    EPOCHS      = 1000
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

    history = {"loss": [], "acc_A": [], "acc_B": [], "acc_C": [], "acc_avg": []}

    for epoch in range(1, EPOCHS + 1):
        loss, acc_A, acc_B, acc_C, acc_avg = train(model, train_loader, optimizer, criterion, DEVICE)
        history["loss"].append(loss)
        history["acc_A"].append(acc_A)
        history["acc_B"].append(acc_B)
        history["acc_C"].append(acc_C)
        history["acc_avg"].append(acc_avg)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {loss:.4f} | Acc A: {acc_A:.2%}  B: {acc_B:.2%}  C: {acc_C:.2%}  Avg: {acc_avg:.2%}")

    # ── Plots ────────────────────────────────────────────────────────────────
    epochs = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["loss"], marker="o", color="steelblue")
    ax1.set_title("Training Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    ax2.plot(epochs, history["acc_A"], marker="o", label="Supervisory (A)")
    ax2.plot(epochs, history["acc_B"], marker="s", label="Operator (B)")
    ax2.plot(epochs, history["acc_C"], marker="^", label="Unsafe Acts (C)")
    ax2.plot(epochs, history["acc_avg"], marker="D", linestyle="--", color="black", label="Average")
    ax2.set_title("Training Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "lstm_training_curves.png")
    plt.savefig(plot_path)
    print(f"\nPlots saved to {plot_path}")
    plt.show()

    # ── Evaluation ───────────────────────────────────────────────────────────
    all_A, all_B, all_C, pred_A, pred_B, pred_C = evaluate(model, test_loader, DEVICE)

    print("\n── A: Supervisory Conditions ──")
    print(classification_report(all_A, pred_A, target_names=encoders.enc_supervisory.classes_, zero_division=0))
    print("── B: Operator Conditions ──")
    print(classification_report(all_B, pred_B, target_names=encoders.enc_operator.classes_, zero_division=0))
    print("── C: Unsafe Acts ──")
    print(classification_report(all_C, pred_C, target_names=encoders.enc_unsafe.classes_, zero_division=0))

    model_path = os.path.join(os.path.dirname(__file__), "hfacs_lstm.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()