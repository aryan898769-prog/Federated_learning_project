import torch
import pandas as pd
import numpy as np
from model import SimpleMLP
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

def evaluate_model(data_path="data/client_5.csv", model_path="global_model.pth"):
    df = pd.read_csv(data_path)
    X = df[["Age", "Income", "Education"]].values.astype(np.float32)
    y = df["Personal Loan"].astype(int).values

    model = SimpleMLP(input_dim=3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X))).numpy().ravel()
        preds = (probs >= 0.5).astype(int)

    print("Confusion Matrix:\n", confusion_matrix(y, preds))
    print(f"Accuracy: {accuracy_score(y, preds):.4f}")
    print(f"Precision: {precision_score(y, preds, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y, preds, zero_division=0):.4f}")
    print(f"AUC: {roc_auc_score(y, probs):.4f}")

if __name__ == "__main__":
    evaluate_model()
