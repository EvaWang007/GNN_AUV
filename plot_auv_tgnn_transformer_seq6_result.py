import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TGNNTransformerRegressor(nn.Module):
    def __init__(self, gcn_hidden=128, tf_d_model=128, tf_nhead=4, tf_layers=2, tf_ff=256, dropout=0.2):
        super().__init__()
        self.gcn1 = GCNConv(1, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        in_dim = 6 * gcn_hidden
        self.proj = nn.Linear(in_dim, tf_d_model) if in_dim != tf_d_model else nn.Identity()
        self.pos = PositionalEncoding(tf_d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=tf_d_model,
            nhead=tf_nhead,
            dim_feedforward=tf_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=tf_layers)
        self.fc1 = nn.Linear(tf_d_model, tf_d_model)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(tf_d_model, 2)

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
        embs = self.proj(embs)
        embs = self.pos(embs)
        h = self.encoder(embs)
        h_last = h[:, -1, :]
        y = F.relu(self.fc1(h_last))
        y = self.drop(y)
        y = self.fc2(y)
        return y.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Plot TGNN-Transformer seq6 results")
    parser.add_argument("--data_npz", default="/home/evawang/T-GCN/data/lbl_aqualoc_seq6_6tuple_train4000_test1000.npz")
    parser.add_argument("--model_path", default="/home/evawang/T-GCN/data/auv_tgnn_transformer_seq6_model.pt")
    parser.add_argument("--out_dir", default="/home/evawang/T-GCN/data")
    parser.add_argument("--suffix", default="", help="Optional suffix for output filenames, e.g. _smoke")
    args = parser.parse_args()

    npz_path = args.data_npz
    model_path = args.model_path
    out_dir = args.out_dir
    suffix = args.suffix

    d = np.load(npz_path, allow_pickle=True)
    X_test, Y_test, M_test = d["X_test"], d["Y_test"], d["M_test"]

    ckpt = torch.load(model_path, map_location="cpu")
    gcn_hidden = int(ckpt.get("gcn_hidden", 128))
    tf_d_model = int(ckpt.get("tf_d_model", 128))
    tf_nhead = int(ckpt.get("tf_nhead", 4))
    tf_layers = int(ckpt.get("tf_layers", 2))
    tf_ff = int(ckpt.get("tf_ff", 256))
    dropout = float(ckpt.get("dropout", 0.2))
    window_size = int(ckpt.get("window_size", 20))

    Xw, Mw, Yw = build_windows(X_test, Y_test, M_test, window_size=window_size)

    model = TGNNTransformerRegressor(
        gcn_hidden=gcn_hidden,
        tf_d_model=tf_d_model,
        tf_nhead=tf_nhead,
        tf_layers=tf_layers,
        tf_ff=tf_ff,
        dropout=dropout,
    )
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

    offset = preds[0] - Yw[0]
    preds_aligned = preds - offset
    err_aligned = np.linalg.norm(preds_aligned - Yw, axis=1)
    mean_err_aligned = float(np.mean(err_aligned))
    rmse_aligned = float(np.sqrt(np.mean(err_aligned ** 2)))

    plt.figure(figsize=(6, 6))
    plt.plot(Yw[:, 0], Yw[:, 1], label="COLMAP Ground Truth")
    plt.plot(preds[:, 0], preds[:, 1], label="TGNN-Transformer Prediction")
    plt.title("TGNN-Transformer Trajectory: COLMAP vs Pred")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p1 = f"{out_dir}/plot_traj_tgnn_transformer_seq6{suffix}.png"
    plt.savefig(p1, dpi=170)
    plt.close()

    # Common valid interval view: only the GT interval that has predictions.
    plt.figure(figsize=(6, 6))
    plt.plot(Yw[:, 0], Yw[:, 1], label="COLMAP (Common Valid Interval)", linewidth=2)
    plt.plot(preds[:, 0], preds[:, 1], label="TGNN-Transformer Prediction", linewidth=2)
    plt.title(f"TGNN-Transformer vs COLMAP (Common Valid Interval, N={len(Yw)})")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p1c = f"{out_dir}/plot_traj_tgnn_transformer_seq6_common_interval{suffix}.png"
    plt.savefig(p1c, dpi=170)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(Yw[:, 0], Yw[:, 1], label="COLMAP Ground Truth")
    plt.plot(preds_aligned[:, 0], preds_aligned[:, 1], label="TGNN-Transformer Pred (Aligned)")
    plt.title("TGNN-Transformer Trajectory (First-point Aligned)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p2 = f"{out_dir}/plot_traj_tgnn_transformer_seq6_aligned{suffix}.png"
    plt.savefig(p2, dpi=170)
    plt.close()

    plt.figure(figsize=(8, 3.5))
    plt.plot(err)
    plt.title("TGNN-Transformer Per-step Error")
    plt.xlabel("test step")
    plt.ylabel("error (m)")
    plt.grid(True)
    plt.tight_layout()
    p3 = f"{out_dir}/plot_error_tgnn_transformer_seq6{suffix}.png"
    plt.savefig(p3, dpi=170)
    plt.close()

    print("Saved figures:")
    print(p1)
    print(p1c)
    print(p2)
    print(p3)
    print(f"Mean error: {mean_err:.6f} m")
    print(f"RMSE: {rmse:.6f} m")
    print(f"Aligned mean error: {mean_err_aligned:.6f} m")
    print(f"Aligned RMSE: {rmse_aligned:.6f} m")


if __name__ == "__main__":
    main()
