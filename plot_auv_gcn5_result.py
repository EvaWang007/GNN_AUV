import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

PGT_ROOT = "/home/evawang/pytorch_geometric_temporal"
if PGT_ROOT not in sys.path:
    sys.path.insert(0, PGT_ROOT)


class FiveNodeGCNRegressor(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(5 * hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight))
        h = F.relu(self.conv3(h, edge_index, edge_weight))
        h = h.reshape(1, -1)
        h = F.relu(self.fc1(h))
        return self.fc2(h).squeeze(0)


def build_edge_index_complete_undirected(num_nodes=5):
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(np.array(edges, dtype=np.int64).T, dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    return edge_index, edge_weight


def main():
    npz_path = "/home/evawang/T-GCN/data/auv5d_20210428_1_0_train4000_test1000.npz"
    model_path = "/home/evawang/T-GCN/data/auv_gcn5_model.pt"

    d = np.load(npz_path, allow_pickle=True)
    X_test = d["X_test"]
    Y_test = d["Y_test"]

    ckpt = torch.load(model_path, map_location="cpu")
    hidden_dim = int(ckpt.get("hidden_dim", 64))

    model = FiveNodeGCNRegressor(hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    edge_index, edge_weight = build_edge_index_complete_undirected(5)

    preds = []
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            x = torch.tensor(X_test[i].reshape(5, 1), dtype=torch.float32)
            y_hat = model(x, edge_index, edge_weight).numpy()
            preds.append(y_hat)
    preds = np.asarray(preds)

    err_xy = preds - Y_test
    err = np.linalg.norm(err_xy, axis=1)
    mean_err = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # Bias analysis
    bias_x = float(np.mean(err_xy[:, 0]))
    bias_y = float(np.mean(err_xy[:, 1]))

    # First-point aligned trajectory (remove global translation at first test point)
    offset = preds[0] - Y_test[0]
    preds_aligned = preds - offset
    err_aligned = np.linalg.norm(preds_aligned - Y_test, axis=1)
    mean_err_aligned = float(np.mean(err_aligned))
    rmse_aligned = float(np.sqrt(np.mean(err_aligned ** 2)))

    out_dir = "/home/evawang/T-GCN/data"

    plt.figure(figsize=(6, 6))
    plt.plot(Y_test[:, 0], Y_test[:, 1], label="Ground Truth")
    plt.plot(preds[:, 0], preds[:, 1], label="Prediction")
    plt.title("Trajectory: GT vs Pred")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p1 = f"{out_dir}/plot_traj.png"
    plt.savefig(p1, dpi=160)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(Y_test[:, 0], Y_test[:, 1], label="Ground Truth")
    plt.plot(preds_aligned[:, 0], preds_aligned[:, 1], label="Prediction (Aligned)")
    plt.title("Trajectory: GT vs Pred (First-point Aligned)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p1b = f"{out_dir}/plot_traj_aligned.png"
    plt.savefig(p1b, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(err)
    plt.title("Per-step Localization Error")
    plt.xlabel("Test Step")
    plt.ylabel("Error (m)")
    plt.grid(True)
    plt.tight_layout()
    p2 = f"{out_dir}/plot_error_curve.png"
    plt.savefig(p2, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(err_xy[:, 0], label="dx = x_pred - x_gt")
    plt.plot(err_xy[:, 1], label="dy = y_pred - y_gt")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Coordinate Bias Over Time")
    plt.xlabel("Test Step")
    plt.ylabel("Bias (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p2b = f"{out_dir}/plot_bias_curve.png"
    plt.savefig(p2b, dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(err, bins=40)
    plt.title("Error Histogram")
    plt.xlabel("Error (m)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    p3 = f"{out_dir}/plot_error_hist.png"
    plt.savefig(p3, dpi=160)
    plt.close()

    print("Saved figures:")
    print(p1)
    print(p1b)
    print(p2)
    print(p2b)
    print(p3)
    print(f"Mean error: {mean_err:.6f} m")
    print(f"RMSE: {rmse:.6f} m")
    print(f"Mean bias_x: {bias_x:.6f} m")
    print(f"Mean bias_y: {bias_y:.6f} m")
    print(f"Aligned mean error: {mean_err_aligned:.6f} m")
    print(f"Aligned RMSE: {rmse_aligned:.6f} m")


if __name__ == "__main__":
    main()
