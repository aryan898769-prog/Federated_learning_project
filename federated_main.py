# main.py
from server import Server
import torch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Main] Using device: {device}")

    server = Server(data_dir="data", train_ids=[1,2,3,4], test_id=5, device=device, hidden=32)
    server.run_federated(rounds=20, local_epochs=5, batch_size=32, lr=1e-3)
    server.plot_training()
