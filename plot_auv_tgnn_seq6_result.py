import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv


class TGNNRegressor(nn.Module):
    def __init__(self, gcn_hidden=128, lstm_hidden=128, dropout=0.2):
        super().__init__()
        self.gcn1 = GCNConv(1, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.lstm = nn.LSTM(
            input_size=6 * gcn_hidden,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.fc1 = nn.Linear(lstm_hidden, lstm_hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(lstm_hidden, 2)

    def encode_step(self, x_step, edge_index, edge_weight):
        h = F.relu(self.gcn1(x_step, edge_index, edge_weight))
        h = self.gcn2(h, edge_index, edge_weight)
        return h.reshape(-1)

    def forward_one_sequence(self, x_seq, m_seq, edge_index_np):
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        embs = []
        for t in range(x_seq.shape[0]):
            x_step = torch.tensor(x_seq[t].reshape(6, 1), dtype=torch.float32)
            ew = edge_weight_with_delay_mask(edge_index_np, int(m_seq[t]))
            edge_weight = torch.tensor(ew, dtype=torch.float32)
            embs.append(self.encode_step(x_step, edge_index, edge_weight))
        embs = torch.stack(embs, dim=0).unsqueeze(0)
        out, _ = self.lstm(embs)
        h = out[:, -1, :]
        y = F.relu(self.fc1(h))
        y = self.drop(y)
        y = self.fc2(y)
        return y.squeeze(0)


def build_edge_index(num_nodes=6):
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    return np.array(edges, dtype=np.int64).T


def edge_weight_with_delay_mask(edge_index, delay_mask_value, delay_node_idx=5):
    w = np.ones(edge_index.shape[1], dtype=np.float32)
    if int(delay_mask_value) == 0:
        src, dst = edge_index[0], edge_index[1]
        cut = (src == delay_node_idx) | (dst == delay_node_idx)
        w[cut] = 0.0
    return w


def build_windows(X, Y, M, window_size=20):
    xs, ms, ys = [], [], []
    for t in range(window_size - 1, X.shape[0]):
        xs.append(X[t - window_size + 1 : t + 1])
        ms.append(M[t - window_size + 1 : t + 1])
        ys.append(Y[t])
    return np.asarray(xs), np.asarray(ms), np.asarray(ys)


def main():
    npz_path = "/home/evawang/T-GCN/data/lbl_aqualoc_seq6_6tuple_train4000_test1000.npz"
    model_path = "/home/evawang/T-GCN/data/auv_tgnn_seq6_model.pt"
    out_dir = "/home/evawang/T-GCN/data"

    d = np.load(npz_path, allow_pickle=True)
    X_test, Y_test, M_test = d["X_test"], d["Y_test"], d["M_test"]

    ckpt = torch.load(model_path, map_location="cpu")
    gcn_hidden = int(ckpt.get("gcn_hidden", 128))
    lstm_hidden = int(ckpt.get("lstm_hidden", 128))
    dropout = float(ckpt.get("dropout", 0.2))
    window_size = int(ckpt.get("window_size", 20))

    Xw, Mw, Yw = build_windows(X_test, Y_test, M_test, window_size=window_size)

    model = TGNNRegressor(gcn_hidden=gcn_hidden, lstm_hidden=lstm_hidden, dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    edge_index_np = build_edge_index(6)

    preds = []
    with torch.no_grad():
        for i in range(Xw.shape[0]):
            yhat = model.forward_one_sequence(Xw[i], Mw[i], edge_index_np).numpy()
            preds.append(yhat)
    preds = np.asarray(preds)

    err_xy = preds - Yw
    err = np.linalg.norm(err_xy, axis=1)
    mean_err = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # aligned
    offset = preds[0] - Yw[0]
    preds_aligned = preds - offset
    err_aligned = np.linalg.norm(preds_aligned - Yw, axis=1)
    mean_err_aligned = float(np.mean(err_aligned))
    rmse_aligned = float(np.sqrt(np.mean(err_aligned ** 2)))

    plt.figure(figsize=(6, 6))
    plt.plot(Yw[:, 0], Yw[:, 1], label="COLMAP Ground Truth")
    plt.plot(preds[:, 0], preds[:, 1], label="TGNN Prediction")
    plt.title("TGNN Trajectory: COLMAP vs Pred")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p1 = f"{out_dir}/plot_traj_tgnn_seq6.png"
    plt.savefig(p1, dpi=170)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(Yw[:, 0], Yw[:, 1], label="COLMAP Ground Truth")
    plt.plot(preds_aligned[:, 0], preds_aligned[:, 1], label="TGNN Prediction (Aligned)")
    plt.title("TGNN Trajectory (First-point Aligned)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p2 = f"{out_dir}/plot_traj_tgnn_seq6_aligned.png"
    plt.savefig(p2, dpi=170)
    plt.close()

    plt.figure(figsize=(8, 3.5))
    plt.plot(err)
    plt.title("TGNN Per-step Error")
    plt.xlabel("test step")
    plt.ylabel("error (m)")
    plt.grid(True)
    plt.tight_layout()
    p3 = f"{out_dir}/plot_error_tgnn_seq6.png"
    plt.savefig(p3, dpi=170)
    plt.close()

    print("Saved figures:")
    print(p1)
    print(p2)
    print(p3)
    print(f"Mean error: {mean_err:.6f} m")
    print(f"RMSE: {rmse:.6f} m")
    print(f"Aligned mean error: {mean_err_aligned:.6f} m")
    print(f"Aligned RMSE: {rmse_aligned:.6f} m")


if __name__ == "__main__":
    main()
