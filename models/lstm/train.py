import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Allow importing from data/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from data.real_dataloader import get_dataloaders  # noqa: E402


def class_weights(labels_tensor, n_classes, device):
    """Inverse-frequency class weights."""
    labels = labels_tensor.numpy()
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=labels)
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    """
    Weighted focal loss — combines class weights with a (1-pt)^gamma modulator
    that down-weights easy majority-class examples and forces the model to focus
    on hard minority-class examples.
    gamma=2 is the standard value from the original paper.
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, targets):
        ce  = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Model ────────────────────────────────────────────────────────────────────

class HFACSCausalLSTM(nn.Module):
    """
    Three-step causal LSTM following the Direct Parent Dependency rule.

    Step A: [Organizational Climate | Employment]  dim=2
              → predict A: Supervisory Conditions  (research purposes)

    Step 0: [WeatherCondition | TimeOfDay | SkyCondNonceil | Personnel | Supervisory]  dim=5
              → predict B: Operator Conditions

    Step 1: embed_proj([soft_B | Supervisory Conditions])  →  predict C: Unsafe Acts
            Supervisory Conditions substitutes for A_pred_embed per the original DAG.
    """

    STEP_A_SIZE = 2  # Organizational Climate (encoded), Employment (numerical)
    STEP0_SIZE  = 5  # 5 label-encoded categorical inputs

    def __init__(self, hidden_size, n_A, n_B, n_C, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size

        self.cell_a     = nn.LSTMCell(self.STEP_A_SIZE, hidden_size)
        self.cell_0     = nn.LSTMCell(self.STEP0_SIZE,  hidden_size)
        self.cell_1     = nn.LSTMCell(self.STEP0_SIZE,  hidden_size)
        self.drop       = nn.Dropout(dropout)

        # Project [soft_B (n_B) | Supervisory Conditions (1)] → STEP0_SIZE
        self.embed_proj = nn.Linear(n_B + 1, self.STEP0_SIZE)

        self.head_A = nn.Linear(hidden_size, n_A)
        self.head_B = nn.Linear(hidden_size, n_B)
        self.head_C = nn.Linear(hidden_size, n_C)

    def forward(self, step_a, step0):
        """
        step_a : (batch, 2) — Organizational Climate, Employment
        step0  : (batch, 5) — WeatherCondition, TimeOfDay, SkyCondNonceil,
                               Personnel Conditions, Supervisory Conditions
        """
        batch = step0.size(0)
        zeros = lambda: torch.zeros(batch, self.hidden_size, device=step0.device)

        # ── Step A → predict A (Supervisory Conditions) ───────────────────────
        hA, _        = self.cell_a(step_a, (zeros(), zeros()))
        logits_A     = self.head_A(self.drop(hA))

        # ── Step 0 → predict B (Operator Conditions) ─────────────────────────
        h0, c0       = self.cell_0(step0, (zeros(), zeros()))
        logits_B     = self.head_B(self.drop(h0))
        soft_B       = torch.softmax(logits_B, dim=1)

        # ── Step 1 → predict C (Unsafe Acts) ─────────────────────────────────
        # Supervisory Conditions (step0[:, 4]) substitutes for A_pred_embed
        supervisory  = step0[:, 4:5]
        step1_in     = self.embed_proj(torch.cat([soft_B.detach(), supervisory], dim=1))
        h1, _        = self.cell_1(step1_in, (h0.detach(), c0.detach()))
        logits_C     = self.head_C(self.drop(h1))

        return logits_A, logits_B, logits_C


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, crit_A, crit_B, crit_C, device):
    model.train()
    total_loss = 0
    all_A, all_B, all_C = [], [], []
    pred_A, pred_B, pred_C = [], [], []

    for s_a, s0, y_A, y_B, y_C in loader:
        s_a              = s_a.to(device)
        s0               = s0.to(device)
        y_A, y_B, y_C   = y_A.to(device), y_B.to(device), y_C.to(device)

        optimizer.zero_grad()
        lA, lB, lC = model(s_a, s0)
        loss        = crit_A(lA, y_A) + crit_B(lB, y_B) + crit_C(lC, y_C)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_A.extend(y_A.cpu().tolist());  pred_A.extend(lA.argmax(1).cpu().tolist())
        all_B.extend(y_B.cpu().tolist());  pred_B.extend(lB.argmax(1).cpu().tolist())
        all_C.extend(y_C.cpu().tolist());  pred_C.extend(lC.argmax(1).cpu().tolist())

    bal_A   = balanced_accuracy_score(all_A, pred_A)
    bal_B   = balanced_accuracy_score(all_B, pred_B)
    bal_C   = balanced_accuracy_score(all_C, pred_C)
    bal_avg = (bal_A + bal_B + bal_C) / 3
    return total_loss / len(loader), bal_A, bal_B, bal_C, bal_avg


def evaluate(model, loader, device):
    model.eval()
    all_A,  all_B,  all_C  = [], [], []
    pred_A, pred_B, pred_C = [], [], []

    with torch.no_grad():
        for s_a, s0, y_A, y_B, y_C in loader:
            s_a = s_a.to(device)
            s0  = s0.to(device)
            lA, lB, lC = model(s_a, s0)
            pred_A.extend(lA.argmax(1).cpu().tolist())
            pred_B.extend(lB.argmax(1).cpu().tolist())
            pred_C.extend(lC.argmax(1).cpu().tolist())
            all_A.extend(y_A.tolist())
            all_B.extend(y_B.tolist())
            all_C.extend(y_C.tolist())

    return all_A, all_B, all_C, pred_A, pred_B, pred_C


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    FILEPATH    = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scamaap dataset.csv")
    HIDDEN_SIZE = 64
    BATCH_SIZE  = 32
    EPOCHS      = 500
    LR          = 3e-4
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, _, encoders = get_dataloaders(
        FILEPATH, batch_size=BATCH_SIZE
    )

    n_A = len(encoders.enc_supervisory.classes_)
    n_B = len(encoders.enc_operator.classes_)
    n_C = len(encoders.enc_unsafe.classes_)

    print("=" * 60)
    print("LSTM Causal Architecture — Input/Output per Step")
    print("=" * 60)
    print("Step A → predict: Supervisory Conditions  (research)")
    print("  Inputs : Organizational Climate")
    print("           Employment (QoQ %)")
    print(f"  Classes: {list(encoders.enc_supervisory.classes_)}")
    print()
    print("Step 0 → predict: Operator Conditions")
    print("  Inputs : Weather Condition")
    print("           Time of Day")
    print("           Sky Condition (Non-ceiling)")
    print("           Personnel Conditions")
    print("           Supervisory Conditions")
    print(f"  Classes: {list(encoders.enc_operator.classes_)}")
    print()
    print("Step 1 → predict: Unsafe Acts")
    print("  Inputs : [soft predictions from Step 0]")
    print("           Supervisory Conditions  (substitutes A_pred_embed per DAG)")
    print(f"  Classes: {list(encoders.enc_unsafe.classes_)}")
    print("=" * 60)
    print(f"Classes — A (Supervisory): {n_A}  B (Operator): {n_B}  C (Unsafe Acts): {n_C}")
    print()

    model = HFACSCausalLSTM(
        hidden_size=HIDDEN_SIZE, n_A=n_A, n_B=n_B, n_C=n_C,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    train_set = train_loader.dataset
    crit_A = nn.CrossEntropyLoss(weight=class_weights(train_set.y_A, n_A, DEVICE))
    crit_B = nn.CrossEntropyLoss(weight=class_weights(train_set.y_B, n_B, DEVICE))
    crit_C = FocalLoss(weight=class_weights(train_set.y_C, n_C, DEVICE), gamma=2.0)

    history = {"loss": [], "acc_A": [], "acc_B": [], "acc_C": [], "acc_avg": []}

    for epoch in range(1, EPOCHS + 1):
        loss, acc_A, acc_B, acc_C, acc_avg = train_epoch(
            model, train_loader, optimizer, crit_A, crit_B, crit_C, DEVICE
        )
        scheduler.step(loss)
        history["loss"].append(loss)
        history["acc_A"].append(acc_A)
        history["acc_B"].append(acc_B)
        history["acc_C"].append(acc_C)
        history["acc_avg"].append(acc_avg)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | Loss: {loss:.4f} | "
            f"BalAcc A: {acc_A:.2%}  B: {acc_B:.2%}  C: {acc_C:.2%}  Avg: {acc_avg:.2%} | "
            f"LR: {current_lr:.2e}"
        )

    # ── Plots ─────────────────────────────────────────────────────────────────
    epochs = range(1, EPOCHS + 1)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["loss"], color="steelblue")
    ax1.set_title("Training Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    ax2.plot(epochs, history["acc_A"], label="Supervisory (A)")
    ax2.plot(epochs, history["acc_B"], label="Operator (B)")
    ax2.plot(epochs, history["acc_C"], label="Unsafe Acts (C)")
    ax2.plot(epochs, history["acc_avg"], linestyle="--", color="black", label="Average")
    ax2.set_title("Training Balanced Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Balanced Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "lstm_training_curves.png")
    plt.savefig(plot_path)
    print(f"\nPlots saved to {plot_path}")
    plt.show()

    model_path = os.path.join(os.path.dirname(__file__), "hfacs_lstm.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
