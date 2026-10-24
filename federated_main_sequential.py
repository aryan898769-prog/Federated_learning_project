# federated_main_sequential.py
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from server import Server
from client import Client
from model import SimpleMLP

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Main] Using device: {device}")

    # Initialize server and prepare preprocessing
    server = Server(data_dir=args.data_dir,
                    train_ids=args.train_clients,
                    test_id=args.test_client,
                    device=device,
                    hidden=args.hidden)
    
    server.find_target_and_features()
    server.build_education_map_and_scaler()

    # Create clients with proper arguments
    clients = []
    for i in args.train_clients:
        client = Client(
            client_id=i,
            csv_path=f"{args.data_dir}/client_{i}.csv",
            feature_cols=server.actual_feature_cols,
            target_col=server.target_col,
            edu_map=server.edu_map,
            scaler_mean=server.scaler_mean,
            scaler_scale=server.scaler_scale,
            device=device
        )
        clients.append(client)
        print(f"[Main] Client {i} ready with {client.n_samples} samples")

    # Initialize global model
    input_dim = len(server.actual_feature_cols)
    global_model = SimpleMLP(input_dim=input_dim, hidden=args.hidden).to(device)

    # To track weight movements
    weight_history = {i: [] for i in range(len(clients))}

    print("\n=== Sequential Federated Learning Start ===")
    for r in range(1, args.global_rounds + 1):
        print(f"\n--- Global Round {r}/{args.global_rounds} ---")

        for idx, client in enumerate(clients):
            print(f"[Client {client.client_id}] Starting local training...")
            local_state, n_samples = client.local_train(
                global_state_dict=global_model.state_dict(),
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                hidden=args.hidden
            )

            # Track weight changes: L2 norm difference
            diff = 0.0
            for k in global_model.state_dict().keys():
                diff += torch.norm(global_model.state_dict()[k] - local_state[k]).item()
            weight_history[idx].append(diff)

            # Aggregate sequentially (weighted by client samples)
            total_samples = n_samples
            for k in global_model.state_dict().keys():
                global_model.state_dict()[k].data = (
                    global_model.state_dict()[k] * 0.0 + local_state[k] * (total_samples / total_samples)
                )

            print(f"[Client {client.client_id}] Finished local training, weight change={diff:.4f}")

        # Evaluate on test set after each global round
        metrics = server.evaluate_on_test(global_model)
        auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A"
        print(f"[Server] Eval -> Acc: {metrics['acc']:.4f}, Prec: {metrics['precision']:.4f}, "
              f"Rec: {metrics['recall']:.4f}, AUC: {auc_str}")

    # Save final global model
    torch.save(global_model.state_dict(), args.save_path)
    print(f"[Server] Training completed, model saved as {args.save_path}")

    # Plot weight movements
    plt.figure(figsize=(10,6))
    color_map = plt.get_cmap("tab10", len(clients))
    for idx in range(len(clients)):
        plt.plot(range(1, args.global_rounds + 1), weight_history[idx],
                 marker='o', label=f"Client {clients[idx].client_id}", color=color_map(idx))
    plt.title("Sequential Federated Learning: Client Weight Movement")
    plt.xlabel("Global Round")
    plt.ylabel("L2 Norm of Weight Change")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Folder containing client CSVs")
    parser.add_argument("--train_clients", nargs="+", type=int, default=[1,2,3,4])
    parser.add_argument("--test_client", type=int, default=5)
    parser.add_argument("--global_rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="global_model_seq.pth")
    args = parser.parse_args()
    main(args)

