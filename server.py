# server.py
import os, copy
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from model import SimpleMLP
from client import Client, SELECTED_FEATURES
import matplotlib.pyplot as plt

class Server:
    def __init__(self, data_dir="data", train_ids=[1,2,3,4], test_id=5, device="cpu", hidden=32):
        self.data_dir = data_dir
        self.train_ids = train_ids
        self.test_id = test_id
        self.device = device
        self.hidden = hidden
        self.clients = []
        self.weight_movements = []
        self.metrics_history = []

    def _csv_path(self, idx):
        return os.path.join(self.data_dir, f"client_{idx}.csv")

    def _build_scaler(self):
        all_data = []
        for i in self.train_ids:
            df = pd.read_csv(self._csv_path(i))
            all_data.append(df[SELECTED_FEATURES].values)
        arr = np.vstack(all_data)
        mean, std = arr.mean(0), arr.std(0)
        std[std == 0] = 1
        return mean, std

    def create_clients(self, scaler_mean, scaler_scale):
        for i in self.train_ids:
            df = pd.read_csv(self._csv_path(i))
            target_col = "Personal Loan"
            edu_map = {str(v).lower(): j for j, v in enumerate(sorted(df["Education"].astype(str).unique()))}
            client = Client(i, self._csv_path(i), SELECTED_FEATURES, target_col, edu_map, scaler_mean, scaler_scale, self.device)
            self.clients.append(client)
            print(f"[Server] Client {i} ready with {len(client.y)} samples")

    def _aggregate(self, states, weights):
        total = sum(weights)
        avg = copy.deepcopy(states[0])
        for k in avg.keys():
            avg[k] = sum(st[k] * (w / total) for st, w in zip(states, weights))
        return avg

    def evaluate(self, model):
        df = pd.read_csv(self._csv_path(self.test_id))
        X = df[SELECTED_FEATURES].values.astype(np.float32)
        y = df["Personal Loan"].astype(int).values
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.tensor(X))).numpy().ravel()
            preds = (probs >= 0.5).astype(int)
        return {
            "acc": accuracy_score(y, preds),
            "prec": precision_score(y, preds, zero_division=0),
            "rec": recall_score(y, preds, zero_division=0),
            "auc": roc_auc_score(y, probs)
        }

    def run_federated(self, rounds=100, local_epochs=5, batch_size=32, lr=1e-3):
        mean, std = self._build_scaler()
        self.create_clients(mean, std)

        model = SimpleMLP(len(SELECTED_FEATURES), self.hidden)
        global_state = model.state_dict()

        for r in range(1, rounds + 1):
            print(f"\n--- Global Round {r}/{rounds} ---")
            local_states, local_weights = [], []

            for client in self.clients:
                state, samples = client.local_train(copy.deepcopy(global_state), local_epochs, batch_size, lr, self.hidden)
                local_states.append(state)
                local_weights.append(samples)

            new_state = self._aggregate(local_states, local_weights)

            # Track weight movement (L2 distance)
            diff = 0
            for k in global_state:
                diff += torch.norm(global_state[k] - new_state[k]).item()
            self.weight_movements.append(diff)

            global_state = new_state
            model.load_state_dict(global_state)
            metrics = self.evaluate(model)
            self.metrics_history.append(metrics)
            print(f"Round {r}: Acc={metrics['acc']:.4f}, Prec={metrics['prec']:.4f}, Rec={metrics['rec']:.4f}, AUC={metrics['auc']:.4f}")

        torch.save(global_state, "global_model.pth")
        print("[Server] Training completed, model saved as global_model.pth")

    def plot_training(self):
        acc = [m["acc"] for m in self.metrics_history]
        auc = [m["auc"] for m in self.metrics_history]

        plt.figure(figsize=(10, 5))
        plt.plot(acc, label="Accuracy")
        plt.plot(auc, label="AUC")
        plt.title("Federated Learning Performance Over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(self.weight_movements, color="red", label="Weight Movement (L2)")
        plt.title("Global Model Weight Movement Over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Weight L2 Distance")
        plt.legend()
        plt.grid(True)
        plt.show()
