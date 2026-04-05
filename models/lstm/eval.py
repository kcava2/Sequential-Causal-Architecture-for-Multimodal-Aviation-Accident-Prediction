import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy import stats

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

        h, c     = self.lstm_cell(x[:, 0, :], (h, c))
        logits_A = self.head_A(self.drop(h))
        soft_A   = torch.softmax(logits_A, dim=1)

        h, c     = self.lstm_cell(x[:, 1, :], (h, c))
        logits_B = self.head_B(self.drop(h))
        soft_B   = torch.softmax(logits_B, dim=1)

        step2_input = self.embed_proj(torch.cat([soft_A, soft_B], dim=1))
        h, c     = self.lstm_cell(step2_input, (h, c))
        logits_C = self.head_C(self.drop(h))

        return logits_A, logits_B, logits_C

# ── Inference helpers ─────────────────────────────────────────────────────────

def run_inference(model, loader, device, noise_std=0.0):
    """Return (true_A, true_B, true_C, pred_A, pred_B, pred_C)."""
    model.eval()
    true_A, true_B, true_C = [], [], []
    pred_A, pred_B, pred_C = [], [], []
    with torch.no_grad():
        for seq, y_A, y_B, y_C in loader:
            seq = seq.to(device)
            if noise_std > 0:
                seq = seq + torch.randn_like(seq) * noise_std
            lA, lB, lC = model(seq)
            pred_A.extend(lA.argmax(1).cpu().tolist())
            pred_B.extend(lB.argmax(1).cpu().tolist())
            pred_C.extend(lC.argmax(1).cpu().tolist())
            true_A.extend(y_A.tolist())
            true_B.extend(y_B.tolist())
            true_C.extend(y_C.tolist())
    return true_A, true_B, true_C, pred_A, pred_B, pred_C


def task_accuracy(true, pred):
    return sum(t == p for t, p in zip(true, pred)) / len(true)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    FILEPATH     = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Simulated_Dataset.xlsx")
    MODEL_PATH   = os.path.join(os.path.dirname(__file__), "hfacs_lstm.pt")
    FIG_DIR      = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    HIDDEN_SIZE  = 64
    BATCH_SIZE   = 32
    TEST_SPLIT   = 0.2
    VAL_SPLIT    = 0.2
    N_TRIALS     = 30
    NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_excel(FILEPATH)

    n_test = int(len(df) * TEST_SPLIT)
    n_val  = int(len(df) * VAL_SPLIT)

    df_train = df.iloc[:-(n_test + n_val)].reset_index(drop=True)
    df_test  = df.iloc[-n_test:].reset_index(drop=True)

    encoders    = HFACSEncoders(df_train)
    test_set    = HFACSSequenceDataset(df_test, encoders)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    n_A = len(encoders.enc_supervisory.classes_)
    n_B = len(encoders.enc_operator.classes_)
    n_C = len(encoders.enc_unsafe.classes_)

    model = HFACSCausalLSTM(input_size=5, hidden_size=HIDDEN_SIZE,
                             n_A=n_A, n_B=n_B, n_C=n_C).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # ── Classification reports ────────────────────────────────────────────────
    true_A, true_B, true_C, pred_A, pred_B, pred_C = run_inference(model, test_loader, DEVICE)

    print(f"Test samples: {len(true_A)}\n")
    print("── A: Supervisory Conditions ──")
    print(classification_report(true_A, pred_A,
                                target_names=encoders.enc_supervisory.classes_,
                                zero_division=0))
    print("── B: Operator Conditions ──")
    print(classification_report(true_B, pred_B,
                                target_names=encoders.enc_operator.classes_,
                                zero_division=0))
    print("── C: Unsafe Acts ──")
    print(classification_report(true_C, pred_C,
                                target_names=encoders.enc_unsafe.classes_,
                                zero_division=0))

    # ── Save per-class metrics CSV ────────────────────────────────────────────
    rows = []
    for task_label, true, pred, enc in [
        ("Supervisory (A)", true_A, pred_A, encoders.enc_supervisory),
        ("Operator (B)",    true_B, pred_B, encoders.enc_operator),
        ("Unsafe Acts (C)", true_C, pred_C, encoders.enc_unsafe),
    ]:
        report = classification_report(true, pred, target_names=enc.classes_,
                                       zero_division=0, output_dict=True)
        for cls_name, metrics in report.items():
            if isinstance(metrics, dict):
                rows.append({"task": task_label, "class": cls_name, **metrics})

    eval_csv = os.path.join(RESULTS_DIR, "lstm_eval_metrics.csv")
    pd.DataFrame(rows).to_csv(eval_csv, index=False)
    print(f"Eval metrics saved to {eval_csv}")

    # ── Confusion matrices ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (true, pred, enc, title) in zip(axes, [
        (true_A, pred_A, encoders.enc_supervisory, "A: Supervisory Conditions"),
        (true_B, pred_B, encoders.enc_operator,    "B: Operator Conditions"),
        (true_C, pred_C, encoders.enc_unsafe,      "C: Unsafe Acts"),
    ]):
        cm = confusion_matrix(true, pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=enc.classes_)
        disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(title)

    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, "lstm_confusion_matrices.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrices saved to {cm_path}")
    plt.close()

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    print("\nRunning sensitivity analysis...")

    # For each noise level, run N_TRIALS inferences and record per-trial accuracy
    noise_results = {std: {"A": [], "B": [], "C": []} for std in NOISE_LEVELS}

    for noise_std in NOISE_LEVELS:
        for _ in range(N_TRIALS):
            tA, tB, tC, pA, pB, pC = run_inference(model, test_loader, DEVICE, noise_std)
            noise_results[noise_std]["A"].append(task_accuracy(tA, pA))
            noise_results[noise_std]["B"].append(task_accuracy(tB, pB))
            noise_results[noise_std]["C"].append(task_accuracy(tC, pC))

    baseline = noise_results[0.0]

    # Statistical tests and summary
    sens_rows = []
    for noise_std in NOISE_LEVELS:
        for task_key, task_label in [("A", "Supervisory"), ("B", "Operator"), ("C", "Unsafe Acts")]:
            trial_accs  = noise_results[noise_std][task_key]
            base_accs   = baseline[task_key]
            mean_acc    = float(np.mean(trial_accs))
            std_acc     = float(np.std(trial_accs))

            if noise_std == 0.0:
                p_value     = 1.0
                significant = False
            else:
                t_stat, p_value = stats.ttest_rel(base_accs, trial_accs)
                p_value     = float(p_value)
                significant = p_value < 0.05

            sens_rows.append({
                "noise_std": noise_std,
                "task": task_label,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "p_value": p_value,
                "significant": significant,
            })

    sens_df = pd.DataFrame(sens_rows)
    sens_csv = os.path.join(RESULTS_DIR, "lstm_sensitivity.csv")
    sens_df.to_csv(sens_csv, index=False)
    print(f"Sensitivity results saved to {sens_csv}")

    # Print sensitivity table
    print("\nSensitivity Analysis (mean acc ± std, p-value vs. baseline):")
    print(f"{'Noise':>6}  {'Task':<14}  {'Mean Acc':>9}  {'Std':>6}  {'p-value':>9}  {'Sig?':>5}")
    print("-" * 60)
    for _, row in sens_df.iterrows():
        sig = "*" if row["significant"] else ""
        print(f"{row['noise_std']:>6.2f}  {row['task']:<14}  "
              f"{row['mean_acc']:>9.2%}  {row['std_acc']:>6.4f}  "
              f"{row['p_value']:>9.4f}  {sig:>5}")

    # ── Sensitivity plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = {"Supervisory": "o", "Operator": "s", "Unsafe Acts": "^"}
    colors  = {"Supervisory": "steelblue", "Operator": "darkorange", "Unsafe Acts": "green"}

    for task_label in ["Supervisory", "Operator", "Unsafe Acts"]:
        task_df = sens_df[sens_df["task"] == task_label]
        ax.errorbar(
            task_df["noise_std"], task_df["mean_acc"], yerr=task_df["std_acc"],
            marker=markers[task_label], color=colors[task_label],
            label=task_label, capsize=4, linewidth=1.5,
        )
        # Mark statistically significant drops
        for _, row in task_df[task_df["significant"]].iterrows():
            ax.annotate("*", (row["noise_std"], row["mean_acc"] - row["std_acc"] - 0.01),
                        ha="center", color=colors[task_label], fontsize=12)

    ax.set_xlabel("Noise Std Dev")
    ax.set_ylabel("Accuracy")
    ax.set_title("Sensitivity Analysis: Accuracy vs. Input Noise")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    sens_path = os.path.join(FIG_DIR, "lstm_sensitivity.png")
    plt.savefig(sens_path, dpi=150)
    print(f"Sensitivity plot saved to {sens_path}")
    plt.close()

if __name__ == "__main__":
    main()
