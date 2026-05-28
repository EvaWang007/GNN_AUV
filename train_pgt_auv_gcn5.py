import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Make sure we can import local pytorch_geometric_temporal clone
PGT_ROOT = "/home/evawang/pytorch_geometric_temporal"
if PGT_ROOT not in sys.path:
    sys.path.insert(0, PGT_ROOT)

from torch_geometric_temporal.signal import StaticGraphTemporalSignal


def build_edge_index_complete_undirected(num_nodes: int = 5):
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    edge_index = np.array(edges, dtype=np.int64).T  # [2, E]
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    return edge_index, edge_weight


def npz_to_signal(X: np.ndarray, Y: np.ndarray, edge_index: np.ndarray, edge_weight: np.ndarray):
    """
    Convert:
      X: [N, 5]
      Y: [N, 2]
    into StaticGraphTemporalSignal snapshots:
      x_t: [5, 1]
      y_t: [2]
    """
    features = [x.astype(np.float32).reshape(5, 1) for x in X]
    targets = [y.astype(np.float32) for y in Y]
    return StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight, features=features, targets=targets)


class FiveNodeGCNRegressor(torch.nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_channels=1, out_channels=hidden_dim)
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.conv3 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.fc1 = torch.nn.Linear(5 * hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_weight)
        h = F.relu(h)
        h = self.conv3(h, edge_index, edge_weight)
        h = F.relu(h)

        h = h.reshape(1, -1)  # [1, 5*hidden]
        h = F.relu(self.fc1(h))
        out = self.fc2(h).squeeze(0)  # [2]
        return out


def eval_dataset(model, dataset, device):
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    with torch.no_grad():
        for snapshot in dataset:
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_attr.to(device) if snapshot.edge_attr is not None else None
            y = snapshot.y.to(device)

            y_hat = model(x, edge_index, edge_weight)
            mse_sum += F.mse_loss(y_hat, y).item()
            mae_sum += F.l1_loss(y_hat, y).item()
            n += 1

    mse = mse_sum / max(n, 1)
    rmse = float(np.sqrt(mse))
    mae = mae_sum / max(n, 1)
    return mse, rmse, mae


def main():
    parser = argparse.ArgumentParser(description="Train 5-node GCN regressor on AUV 5D dataset")
    parser.add_argument("--data_npz", required=True, help="Path to prepared npz")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--model_out", default="/home/evawang/T-GCN/data/auv_gcn5_model.pt")
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    data_path = Path(args.data_npz)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    npz = np.load(data_path, allow_pickle=True)
    X_train = npz["X_train"]
    Y_train = npz["Y_train"]
    X_test = npz["X_test"]
    Y_test = npz["Y_test"]

    edge_index, edge_weight = build_edge_index_complete_undirected(num_nodes=5)
    train_dataset = npz_to_signal(X_train, Y_train, edge_index, edge_weight)
    test_dataset = npz_to_signal(X_test, Y_test, edge_index, edge_weight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FiveNodeGCNRegressor(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_test_rmse = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss_sum = 0.0
        n = 0
        for snapshot in train_dataset:
            x = snapshot.x.to(device)
            edge_idx = snapshot.edge_index.to(device)
            edge_w = snapshot.edge_attr.to(device) if snapshot.edge_attr is not None else None
            y = snapshot.y.to(device)

            y_hat = model(x, edge_idx, edge_w)
            loss = F.mse_loss(y_hat, y)
            loss_sum += loss
            n += 1

        mean_loss = loss_sum / max(n, 1)
        mean_loss.backward()
        optimizer.step()

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            train_mse = mean_loss.item()
            train_rmse = float(np.sqrt(train_mse))
            test_mse, test_rmse, test_mae = eval_dataset(model, test_dataset, device)
            print(
                f"Epoch {epoch:03d} | "
                f"train_mse={train_mse:.6f} train_rmse={train_rmse:.6f} | "
                f"test_mse={test_mse:.6f} test_rmse={test_rmse:.6f} test_mae={test_mae:.6f}"
            )

            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        out_path = Path(args.model_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "hidden_dim": args.hidden_dim,
                "best_test_rmse": best_test_rmse,
                "data_npz": str(data_path),
            },
            out_path,
        )
        print(f"Saved best model to: {out_path}")
        print(f"Best test RMSE: {best_test_rmse:.6f}")


if __name__ == "__main__":
    main()
