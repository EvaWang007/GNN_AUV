import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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


class PureTransformerRegressor(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=4, num_layers=2, ff_dim=256, dropout=0.2):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model, 2)

    def forward(self, x_seq):
        h = self.embed(x_seq)
        h = self.pos(h)
        h = self.encoder(h)
        h_last = h[:, -1, :]
        y = F.relu(self.fc1(h_last))
        y = self.drop(y)
        y = self.fc2(y)
        return y


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


def build_windows(X, Y, M, window_size=20):
    xs, ms, ys = [], [], []
    for t in range(window_size - 1, X.shape[0]):
        xs.append(X[t - window_size + 1 : t + 1])
        ms.append(M[t - window_size + 1 : t + 1])
        ys.append(Y[t])
    return np.asarray(xs), np.asarray(ms), np.asarray(ys)


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


def run_pure_tf_prediction(data_npz, model_pt):
    d = np.load(data_npz, allow_pickle=True)
    X_test = d["X_test"]
    Y_test = d["Y_test"]
    M_test = d["M_test"]
    ckpt = torch.load(model_pt, map_location="cpu")
    window_size = int(ckpt.get("window_size", 20))
    Xw, Mw, Yw = build_windows(X_test, Y_test, M_test, window_size=window_size)
    model = PureTransformerRegressor(
        input_dim=Xw.shape[-1],
        d_model=int(ckpt.get("d_model", 128)),
        nhead=int(ckpt.get("nhead", 4)),
        num_layers=int(ckpt.get("num_layers", 2)),
        ff_dim=int(ckpt.get("ff_dim", 256)),
        dropout=float(ckpt.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(Xw, dtype=torch.float32)).numpy()
    return preds, Yw


def run_tgnn_transformer_prediction(data_npz, model_pt):
    d = np.load(data_npz, allow_pickle=True)
    X_test = d["X_test"]
    Y_test = d["Y_test"]
    M_test = d["M_test"]

    ckpt = torch.load(model_pt, map_location="cpu")
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
    return preds, Yw


def cdf_xy(err):
    s = np.sort(err)
    y = np.arange(1, len(s) + 1) / len(s)
    return s, y


def main():
    root = Path("/home/evawang/T-GCN/data")
    data_npz = root / "lbl_aqualoc_seq6_6tuple_train4000_test1000.npz"
    pure_tf_model = root / "auv_pure_tf_seq6_model.pt"
    pure_tf_phy_model = root / "auv_pure_tf_phy_seq6_model.pt"
    tgnn_tf_model = root / "auv_tgnn_transformer_seq6_model.pt"
    ekf_joint = root / "ekf_seq6_joint_result.npz"

    pred_pure_tf, gt_pure_tf = run_pure_tf_prediction(data_npz, pure_tf_model)
    pred_pure_tf_phy, gt_pure_tf_phy = run_pure_tf_prediction(data_npz, pure_tf_phy_model)
    pred_tgnn_tf, gt_tgnn_tf = run_tgnn_transformer_prediction(data_npz, tgnn_tf_model)

    e2 = np.load(ekf_joint, allow_pickle=True)
    pred_e2 = e2["pred_xy"]

    gt_xy = gt_tgnn_tf
    base_len = len(gt_xy)
    pred_pure_tf = pred_pure_tf[-base_len:]
    pred_pure_tf_phy = pred_pure_tf_phy[-base_len:]
    pred_tgnn_tf = pred_tgnn_tf[-base_len:]
    pred_e2 = pred_e2[-base_len:]

    n = min(len(gt_xy), len(pred_pure_tf), len(pred_pure_tf_phy), len(pred_tgnn_tf), len(pred_e2))
    gt_xy = gt_xy[:n]
    pred_pure_tf = pred_pure_tf[:n]
    pred_pure_tf_phy = pred_pure_tf_phy[:n]
    pred_tgnn_tf = pred_tgnn_tf[:n]
    pred_e2 = pred_e2[:n]

    pred_pure_tf_a = pred_pure_tf - (pred_pure_tf[0] - gt_xy[0])
    pred_pure_tf_phy_a = pred_pure_tf_phy - (pred_pure_tf_phy[0] - gt_xy[0])
    pred_tgnn_tf_a = pred_tgnn_tf - (pred_tgnn_tf[0] - gt_xy[0])
    pred_e2_a = pred_e2 - (pred_e2[0] - gt_xy[0])

    err_pure_tf = np.linalg.norm(pred_pure_tf_a - gt_xy, axis=1)
    err_pure_tf_phy = np.linalg.norm(pred_pure_tf_phy_a - gt_xy, axis=1)
    err_tgnn_tf = np.linalg.norm(pred_tgnn_tf_a - gt_xy, axis=1)
    err_e2 = np.linalg.norm(pred_e2_a - gt_xy, axis=1)

    plt.figure(figsize=(7.2, 5.2))
    for err, name, color in [
        (err_e2, "EKF-Joint", "#e377c2"),
        (err_pure_tf, "Pure Transformer", "#ff7f0e"),
        (err_pure_tf_phy, "Transformer+Physics", "#2ca02c"),
        (err_tgnn_tf, "Transformer+GNN", "#d62728"),
    ]:
        sx, sy = cdf_xy(err)
        plt.plot(sx, sy, label=name, linewidth=2.2, color=color)

    plt.title("Error CDF Comparison (First-point Aligned)")
    plt.xlabel("Error (m)")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out = root / "compare_error_cdf_seq6_paper4.png"
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"Saved paper CDF to: {out}")


if __name__ == "__main__":
    main()
