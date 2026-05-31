import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt


def build_full_edge_index(num_nodes: int = 6):
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    edge_index = np.array(edges, dtype=np.int64).T
    return edge_index


def edge_weight_with_delay_mask(edge_index: np.ndarray, delay_mask_value: int, delay_node_idx: int = 5):
    w = np.ones(edge_index.shape[1], dtype=np.float32)
    if int(delay_mask_value) == 0:
        src = edge_index[0]
        dst = edge_index[1]
        cut = (src == delay_node_idx) | (dst == delay_node_idx)
        w[cut] = 0.0
    return w


class SixNodeGCNRegressor(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(6 * hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight))
        h = F.relu(self.conv3(h, edge_index, edge_weight))
        h = h.reshape(1, -1)
        h = F.relu(self.fc1(h))
        return self.fc2(h).squeeze(0)


def eval_split(model, X, Y, M, edge_index_np, device):
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = X.shape[0]
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)

    with torch.no_grad():
        for i in range(n):
            x = torch.tensor(X[i].reshape(6, 1), dtype=torch.float32, device=device)
            y = torch.tensor(Y[i], dtype=torch.float32, device=device)
            ew = edge_weight_with_delay_mask(edge_index_np, M[i])
            edge_weight = torch.tensor(ew, dtype=torch.float32, device=device)

            y_hat = model(x, edge_index, edge_weight)
            mse_sum += F.mse_loss(y_hat, y).item()
            mae_sum += F.l1_loss(y_hat, y).item()

    mse = mse_sum / max(n, 1)
    rmse = float(np.sqrt(mse))
    mae = mae_sum / max(n, 1)
    return mse, rmse, mae


def main():
    parser = argparse.ArgumentParser(description="Train 6-node GCN with delay mask")
    parser.add_argument("--data_npz", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--model_out", default="/home/evawang/T-GCN/data/auv_gcn6_delay_model.pt")
    parser.add_argument("--history_out", default="/home/evawang/T-GCN/data/auv_gcn6_delay_history.csv")
    parser.add_argument("--curve_out", default="/home/evawang/T-GCN/data/auv_gcn6_delay_loss_curve.png")
    args = parser.parse_args()

    d = np.load(args.data_npz, allow_pickle=True)
    X_train, Y_train, M_train = d["X_train"], d["Y_train"], d["M_train"]
    X_test, Y_test, M_test = d["X_test"], d["Y_test"], d["M_test"]

    edge_index_np = build_full_edge_index(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)

    model = SixNodeGCNRegressor(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_rmse = float("inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss_sum = 0.0
        n = X_train.shape[0]
        for i in range(n):
            x = torch.tensor(X_train[i].reshape(6, 1), dtype=torch.float32, device=device)
            y = torch.tensor(Y_train[i], dtype=torch.float32, device=device)
            ew = edge_weight_with_delay_mask(edge_index_np, M_train[i])
            edge_weight = torch.tensor(ew, dtype=torch.float32, device=device)

            y_hat = model(x, edge_index, edge_weight)
            loss_sum = loss_sum + F.mse_loss(y_hat, y)

        mean_loss = loss_sum / max(n, 1)
        mean_loss.backward()
        optimizer.step()

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            train_mse = mean_loss.item()
            train_rmse = float(np.sqrt(train_mse))
            test_mse, test_rmse, test_mae = eval_split(model, X_test, Y_test, M_test, edge_index_np, device)
            print(
                f"Epoch {epoch:03d} | "
                f"train_mse={train_mse:.6f} train_rmse={train_rmse:.6f} | "
                f"test_mse={test_mse:.6f} test_rmse={test_rmse:.6f} test_mae={test_mae:.6f}"
            )
            history.append([epoch, train_mse, train_rmse, test_mse, test_rmse, test_mae])
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        out = Path(args.model_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "hidden_dim": args.hidden_dim,
                "best_test_rmse": best_rmse,
                "data_npz": args.data_npz,
            },
            out,
        )
        print(f"Saved best model to: {out}")
        print(f"Best test RMSE: {best_rmse:.6f}")

    if history:
        hist = np.asarray(history, dtype=np.float64)
        hist_path = Path(args.history_out)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            hist_path,
            hist,
            delimiter=",",
            header="epoch,train_mse,train_rmse,test_mse,test_rmse,test_mae",
            comments="",
        )
        print(f"Saved history csv to: {hist_path}")

        plt.figure(figsize=(8, 4))
        plt.plot(hist[:, 0], hist[:, 2], label="train_rmse")
        plt.plot(hist[:, 0], hist[:, 4], label="test_rmse")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE (m)")
        plt.title("GNN Training Curve (RMSE vs Epoch)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        curve_path = Path(args.curve_out)
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(curve_path, dpi=170)
        plt.close()
        print(f"Saved loss curve to: {curve_path}")


if __name__ == "__main__":
    main()
