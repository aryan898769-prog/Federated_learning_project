"""
visualize_results.py
---------------------------------------
Standalone visualization script for the federated learning project.
It uses the global model and test client data to generate visual insights:
- Accuracy and AUC trend
- Weight movement trend
- Feature importance
- Confusion matrix heatmap
- ROC curve
- Combined dashboard (optional)
---------------------------------------
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from model import SimpleMLP  # use your existing model
from evaluate import evaluate_global_model  # reuses your existing evaluator

# -----------------------------
# 1️⃣  Load metrics (if saved)
# -----------------------------
metrics_path = os.path.join("results", "metrics_history.csv")

if os.path.exists(metrics_path):
    df = pd.read_csv(metrics_path)
    print("✅ Loaded metrics from:", metrics_path)
else:
    print("⚠️ No saved metrics found. You can export metrics from server during training as CSV.")
    df = None

# -----------------------------
# 2️⃣  Accuracy & AUC Trend
# -----------------------------
if df is not None:
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["acc"], label="Accuracy", marker="o")
    plt.plot(df["round"], df["auc"], label="AUC", marker="s")
    plt.xlabel("Global Round")
    plt.ylabel("Metric Value")
    plt.title("Global Model Performance Over Rounds")
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------------
# 3️⃣  Weight Movement Visualization
# -----------------------------
if df is not None and "weight_movement" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["weight_movement"], color="purple", marker="d")
    plt.xlabel("Global Round")
    plt.ylabel("Weight L2 Distance")
    plt.title("Global Weight Movement Across Rounds")
    plt.grid(True)
    plt.show()

# -----------------------------
# 4️⃣  Feature Importance (from first layer)
# -----------------------------
model_path = os.path.join("results", "global_model.pth")
if os.path.exists(model_path):
    model = SimpleMLP(input_dim=3)  # adjust input_dim if needed
    model.load_state_dict(torch.load(model_path))
    model.eval()

    first_layer_weights = model.net[0].weight.detach().numpy().flatten()
    feature_names = ["Age", "Income", "Education"]  # adjust if you used different ones

    plt.figure(figsize=(6, 4))
    plt.bar(feature_names, np.abs(first_layer_weights), color="teal")
    plt.title("Feature Importance (First Layer Absolute Weights)")
    plt.ylabel("Weight Magnitude")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.show()
else:
    print("⚠️ No saved global model found for feature importance.")

# -----------------------------
# 5️⃣  Confusion Matrix + ROC Curve
# -----------------------------
test_data_path = os.path.join("data", "client_5.csv")

if os.path.exists(test_data_path):
    test_data = pd.read_csv(test_data_path)
    X_test = torch.tensor(test_data[["Age", "Income", "Education"]].values, dtype=torch.float32)
    y_test = torch.tensor(test_data["PersonalLoan"].values, dtype=torch.float32)

    # Evaluate model on test data
    with torch.no_grad():
        probs = model(X_test).squeeze()
        preds = (probs >= 0.5).int()
        y_true = y_test.int()

    # Confusion Matrix
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Client 5 Test Data)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_value = roc_auc_score(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}", color="darkorange")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Client 5 Test Data)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
else:
    print("⚠️R Client 5 data not found. Skipping ROC and Confusion Matrix visualization.")

# -----------------------------
# 6️⃣  Combined Dashboard (Optional)
# -----------------------------
if df is not None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(df["round"], df["acc"], label="Accuracy", color="green")
    axes[0].set_title("Accuracy Trend")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Accuracy")

    axes[1].plot(df["round"], df["auc"], label="AUC", color="blue")
    axes[1].set_title("AUC Trend")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("AUC")

    if "weight_movement" in df.columns:
        axes[2].plot(df["round"], df["weight_movement"], label="Weight Movement", color="purple")
        axes[2].set_title("Weight Change")
        axes[2].set_xlabel("Round")
        axes[2].set_ylabel("Distance")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    plt.suptitle("Federated Learning Metrics Dashboard", fontsize=14)
    plt.tight_layout()
    plt.show()
