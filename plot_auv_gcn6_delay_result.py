import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def build_full_edge_index(num_nodes: int = 6):
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    return np.array(edges, dtype=np.int64).T


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


def main():
    npz_path = "/home/evawang/T-GCN/data/auv6d_delay_20210428_train4000_test1000.npz"
    model_path = "/home/evawang/T-GCN/data/auv_gcn6_delay_model.pt"
    out_dir = "/home/evawang/T-GCN/data"

    d = np.load(npz_path, allow_pickle=True)
    X_test, Y_test, M_test = d["X_test"], d["Y_test"], d["M_test"]

    ckpt = torch.load(model_path, map_location="cpu")
    hidden_dim = int(ckpt.get("hidden_dim", 64))

    model = SixNodeGCNRegressor(hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    edge_index_np = build_full_edge_index(6)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    preds = []
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            x = torch.tensor(X_test[i].reshape(6, 1), dtype=torch.float32)
            ew = edge_weight_with_delay_mask(edge_index_np, int(M_test[i]))
            edge_weight = torch.tensor(ew, dtype=torch.float32)
            y_hat = model(x, edge_index, edge_weight).numpy()
            preds.append(y_hat)
    preds = np.asarray(preds)

    err_xy = preds - Y_test
    err = np.linalg.norm(err_xy, axis=1)
    mean_err = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias_x = float(np.mean(err_xy[:, 0]))
    bias_y = float(np.mean(err_xy[:, 1]))

    offset = preds[0] - Y_test[0]
    preds_aligned = preds - offset
    err_aligned = np.linalg.norm(preds_aligned - Y_test, axis=1)
    mean_err_aligned = float(np.mean(err_aligned))
    rmse_aligned = float(np.sqrt(np.mean(err_aligned ** 2)))

    plt.figure(figsize=(6, 6))
    plt.plot(Y_test[:, 0], Y_test[:, 1], label="Ground Truth")
    plt.plot(preds[:, 0], preds[:, 1], label="Prediction (6D+delay)")
    plt.title("Trajectory: GT vs Pred (6D+delay)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p1 = f"{out_dir}/plot_traj_gcn6_delay.png"
    plt.savefig(p1, dpi=160)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(Y_test[:, 0], Y_test[:, 1], label="Ground Truth")
    plt.plot(preds_aligned[:, 0], preds_aligned[:, 1], label="Prediction Aligned")
    plt.title("Trajectory: First-point Aligned (6D+delay)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p1b = f"{out_dir}/plot_traj_gcn6_delay_aligned.png"
    plt.savefig(p1b, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(err)
    plt.title("Per-step Localization Error (6D+delay)")
    plt.xlabel("Test Step")
    plt.ylabel("Error (m)")
    plt.grid(True)
    plt.tight_layout()
    p2 = f"{out_dir}/plot_error_curve_gcn6_delay.png"
    plt.savefig(p2, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(err_xy[:, 0], label="dx")
    plt.plot(err_xy[:, 1], label="dy")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Coordinate Bias Over Time (6D+delay)")
    plt.xlabel("Test Step")
    plt.ylabel("Bias (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p2b = f"{out_dir}/plot_bias_curve_gcn6_delay.png"
    plt.savefig(p2b, dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(err, bins=40)
    plt.title("Error Histogram (6D+delay)")
    plt.xlabel("Error (m)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    p3 = f"{out_dir}/plot_error_hist_gcn6_delay.png"
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
