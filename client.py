# client.py
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from model import SimpleMLP

SELECTED_FEATURES = ["Age", "Income", "Education"]

class BankDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.astype(np.float32))
        self.y = torch.tensor(y.astype(np.float32)).unsqueeze(1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Client:
    def __init__(self, client_id, csv_path, feature_cols, target_col, edu_map, scaler_mean, scaler_scale, device="cpu"):
        self.client_id = client_id
        self.csv_path = csv_path
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.edu_map = edu_map
        self.scaler_mean = np.array(scaler_mean)
        self.scaler_scale = np.array(scaler_scale)
        self.device = torch.device(device)
        self._load_data()

    def _encode_education(self, series):
        s = series.astype(str).str.strip().str.lower()
        mapped = s.map(lambda v: self.edu_map.get(v, -1))
        if (mapped == -1).any():
            mode_val = mapped[mapped != -1].mode()
            fill = int(mode_val.iloc[0]) if len(mode_val) > 0 else 0
            mapped = mapped.replace(-1, fill)
        return mapped.values.astype(np.float32)

    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        X_df = df[self.feature_cols].copy()
        y = df[self.target_col].astype(int).values
        if "Education" in X_df.columns:
            X_df["Education"] = self._encode_education(X_df["Education"])
        X_df.fillna(X_df.median(), inplace=True)
        X = (X_df.values.astype(np.float32) - self.scaler_mean) / (self.scaler_scale + 1e-8)
        self.X, self.y = shuffle(X, y, random_state=42)

    def local_train(self, global_state_dict, local_epochs=5, batch_size=32, lr=1e-3, hidden=32):
        model = SimpleMLP(input_dim=self.X.shape[1], hidden=hidden)
        model.load_state_dict(global_state_dict)
        model.to(self.device)
        model.train()
        loader = DataLoader(BankDataset(self.X, self.y), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(local_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()

        return model.state_dict(), len(self.y)
