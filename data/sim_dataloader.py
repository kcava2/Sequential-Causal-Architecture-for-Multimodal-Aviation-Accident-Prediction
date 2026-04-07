import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

class HFACSDataset(Dataset):
    def __init__(self, filepath, target_col="Unsafe Conditions", fit_encoders=True,
                 encoders=None, scaler=None):
        df = pd.read_excel(filepath)

        self.target_col = target_col

        # Categorical columns to label encode
        self.cat_cols = [
            "Light Conditions",
            "Basic Meteorological Conditions",
            "Personnel Conditions",
            "Supervisory Conditions",
            "Operator Conditions",
        ]

        # Numerical columns to standardize
        self.num_cols = [
            "Employment Change vs Prior Period (%)",
            "Wind Conditions (kt)",
            "Temperature (C)",
        ]

        # Encode categoricals
        if fit_encoders:
            self.encoders = {col: LabelEncoder().fit(df[col]) for col in self.cat_cols}
            self.target_encoder = LabelEncoder().fit(df[target_col])
        else:
            self.encoders = encoders
            self.target_encoder = encoders["target"]

        for col in self.cat_cols:
            df[col] = self.encoders[col].transform(df[col])

        # Scale numericals
        if fit_encoders:
            self.scaler = StandardScaler().fit(df[self.num_cols])
        else:
            self.scaler = scaler

        df[self.num_cols] = self.scaler.transform(df[self.num_cols])

        # Build tensors
        self.X = torch.tensor(
            df[self.num_cols + self.cat_cols].values, dtype=torch.float32
        )
        self.y = torch.tensor(
            self.target_encoder.transform(df[target_col]), dtype=torch.long
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(filepath, test_split=0.2, batch_size=32, seed=42):
    full = HFACSDataset(filepath, fit_encoders=True)

    n = len(full)
    n_test = int(n * test_split)
    n_train = n - n_test

    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = torch.utils.data.random_split(
        full, [n_train, n_test], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, full

if __name__ == "__main__":
    import os
    filepath = os.path.join(os.path.dirname(__file__), "Simulated_Dataset.xlsx")
    train_loader, test_loader, dataset = get_dataloaders(filepath)

    # Check input/output dimensions
    n_features  = dataset.X.shape[1]   # 8 (3 numerical + 5 categorical)
    n_classes   = len(dataset.target_encoder.classes_)  # 5 unsafe condition types

    for X_batch, y_batch in train_loader:
        print(X_batch.shape, y_batch.shape)
        break