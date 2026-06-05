import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.nn import GCNConv
import torch.nn as nn
import math


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


class TGNNTransformerEDRegressor(nn.Module):
    def __init__(self, gcn_hidden=128, d_model=128, nhead=4, num_layers=2, ff_dim=256, dropout=0.2):
        super().__init__()
        self.gcn1 = GCNConv(1, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        in_dim = 6 * gcn_hidden
        self.proj = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fc1 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model, 2)

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
        src = torch.stack(embs, dim=0).unsqueeze(0)
        src = self.proj(src)
        src = self.pos(src)
        tgt = self.start_token.expand(1, 1, -1)
        tgt = self.pos(tgt)
        out = self.transformer(src=src, tgt=tgt)
        h = out[:, -1, :]
        y = F.relu(self.fc1(h))
        y = self.drop(y)
        y = self.fc2(y)
        return y.squeeze(0)


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


def eval_metrics(pred_xy, gt_xy):
    err_xy = pred_xy - gt_xy
    err = np.linalg.norm(err_xy, axis=1)
    mean_error = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias_x = float(np.mean(err_xy[:, 0]))
    bias_y = float(np.mean(err_xy[:, 1]))

    off = pred_xy[0] - gt_xy[0]
    pred_a = pred_xy - off
    err_a = np.linalg.norm(pred_a - gt_xy, axis=1)
    return {
        'err': err,
        'mean_error': mean_error,
        'rmse': rmse,
        'bias_x': bias_x,
        'bias_y': bias_y,
        'aligned_rmse': float(np.sqrt(np.mean(err_a ** 2))),
    }


def run_gnn_prediction(data_npz, model_pt):
    d = np.load(data_npz, allow_pickle=True)
    X_test = d['X_test']
    Y_test = d['Y_test']
    M_test = d['M_test']

    ckpt = torch.load(model_pt, map_location='cpu')
    hidden_dim = int(ckpt.get('hidden_dim', 64))

    model = SixNodeGCNRegressor(hidden_dim=hidden_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    edge_index_np = build_edge_index(6)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    preds = []
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            x = torch.tensor(X_test[i].reshape(6, 1), dtype=torch.float32)
            ew = edge_weight_with_delay_mask(edge_index_np, int(M_test[i]))
            ew = torch.tensor(ew, dtype=torch.float32)
            yhat = model(x, edge_index, ew).numpy()
            preds.append(yhat)
    preds = np.asarray(preds)
    return preds, Y_test


def build_windows(X, Y, M, window_size=20):
    xs, ms, ys = [], [], []
    for t in range(window_size - 1, X.shape[0]):
        xs.append(X[t - window_size + 1 : t + 1])
        ms.append(M[t - window_size + 1 : t + 1])
        ys.append(Y[t])
    return np.asarray(xs), np.asarray(ms), np.asarray(ys)


def run_tgnn_prediction(data_npz, model_pt):
    d = np.load(data_npz, allow_pickle=True)
    X_test = d["X_test"]
    Y_test = d["Y_test"]
    M_test = d["M_test"]

    ckpt = torch.load(model_pt, map_location="cpu")
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


def run_tgnn_transformer_ed_prediction(data_npz, model_pt):
    d = np.load(data_npz, allow_pickle=True)
    X_test = d["X_test"]
    Y_test = d["Y_test"]
    M_test = d["M_test"]

    ckpt = torch.load(model_pt, map_location="cpu")
    gcn_hidden = int(ckpt.get("gcn_hidden", 128))
    d_model = int(ckpt.get("d_model", 128))
    nhead = int(ckpt.get("nhead", 4))
    num_layers = int(ckpt.get("num_layers", 2))
    ff_dim = int(ckpt.get("ff_dim", 256))
    dropout = float(ckpt.get("dropout", 0.2))
    window_size = int(ckpt.get("window_size", 20))

    Xw, Mw, Yw = build_windows(X_test, Y_test, M_test, window_size=window_size)
    model = TGNNTransformerEDRegressor(
        gcn_hidden=gcn_hidden,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ff_dim=ff_dim,
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


def run_pure_tf_phy_prediction(data_npz, model_pt):
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


def main():
    root = Path('/home/evawang/T-GCN/data')
    gnn_data = root / 'lbl_aqualoc_seq6_6tuple_train4000_test1000.npz'
    gnn_model = root / 'auv_gcn6_delay_seq6_model.pt'
    tgnn_model = root / 'auv_tgnn_seq6_model.pt'
    tgnn_tf_model = root / 'auv_tgnn_transformer_seq6_model.pt'
    tgnn_tf_ed_model = root / 'auv_tgnn_transformer_ed_seq6_model.pt'
    pure_tf_model = root / 'auv_pure_tf_seq6_model.pt'
    pure_tf_phy_model = root / 'auv_pure_tf_phy_seq6_model.pt'
    ekf_joint = root / 'ekf_seq6_joint_result.npz'

    pred_gnn, gt = run_gnn_prediction(gnn_data, gnn_model)
    m_gnn = eval_metrics(pred_gnn, gt)
    pred_tgnn, gt_tgnn = run_tgnn_prediction(gnn_data, tgnn_model)
    m_tgnn = eval_metrics(pred_tgnn, gt_tgnn)
    pred_tgnn_tf, gt_tgnn_tf = run_tgnn_transformer_prediction(gnn_data, tgnn_tf_model)
    m_tgnn_tf = eval_metrics(pred_tgnn_tf, gt_tgnn_tf)
    pred_tgnn_tf_ed, gt_tgnn_tf_ed = run_tgnn_transformer_ed_prediction(gnn_data, tgnn_tf_ed_model)
    m_tgnn_tf_ed = eval_metrics(pred_tgnn_tf_ed, gt_tgnn_tf_ed)
    pred_pure_tf0, gt_pure_tf0 = run_pure_tf_prediction(gnn_data, pure_tf_model)
    m_pure_tf0 = eval_metrics(pred_pure_tf0, gt_pure_tf0)
    pred_pure_tf, gt_pure_tf = run_pure_tf_phy_prediction(gnn_data, pure_tf_phy_model)
    m_pure_tf = eval_metrics(pred_pure_tf, gt_pure_tf)

    e2 = np.load(ekf_joint, allow_pickle=True)
    pred_e2 = e2['pred_xy']
    pred_e2 = pred_e2
    m_e2 = eval_metrics(pred_e2, e2['gt_xy'])

    # Use TGNN-Transformer time base (windowed output)
    gt_xy = gt_tgnn_tf_ed
    base_len = len(gt_xy)
    pred_gnn = pred_gnn[-base_len:]
    pred_tgnn = pred_tgnn[-base_len:]
    pred_e2 = pred_e2[-base_len:]
    pred_tgnn_tf = pred_tgnn_tf[-base_len:]
    pred_tgnn_tf_ed = pred_tgnn_tf_ed[-base_len:]
    pred_pure_tf0 = pred_pure_tf0[-base_len:]
    pred_pure_tf = pred_pure_tf[-base_len:]
    n = min(len(gt_xy), len(pred_e2), len(pred_gnn), len(pred_tgnn), len(pred_tgnn_tf), len(pred_tgnn_tf_ed), len(pred_pure_tf0), len(pred_pure_tf))
    gt_xy = gt_xy[:n]
    pred_tgnn_tf = pred_tgnn_tf[:n]
    pred_tgnn_tf_ed = pred_tgnn_tf_ed[:n]
    pred_pure_tf0 = pred_pure_tf0[:n]
    pred_pure_tf = pred_pure_tf[:n]
    pred_tgnn = pred_tgnn[:n]
    pred_gnn = pred_gnn[:n]
    pred_e2 = pred_e2[:n]

    out_dir = root

    # Trajectory (focus near x=0 by using COLMAP-first-point centered frame)
    gt0 = gt_xy[0].copy()
    gt_plot = gt_xy - gt0
    tgnn_tf_ed_plot = pred_tgnn_tf_ed - gt0
    pure_tf0_plot = pred_pure_tf0 - gt0
    pure_tf_plot = pred_pure_tf - gt0
    tgnn_tf_plot = pred_tgnn_tf - gt0
    tgnn_plot = pred_tgnn - gt0
    gnn_plot = pred_gnn - gt0
    e2_plot = pred_e2 - gt0

    plt.figure(figsize=(7, 7))
    plt.scatter(gt_plot[:, 0], gt_plot[:, 1], label='COLMAP Ground Truth', s=14, alpha=0.9)
    plt.scatter(tgnn_tf_ed_plot[:, 0], tgnn_tf_ed_plot[:, 1], label='TGNN-Transformer-ED', s=12, alpha=0.85)
    plt.scatter(pure_tf0_plot[:, 0], pure_tf0_plot[:, 1], label='PureTF', s=12, alpha=0.85)
    plt.scatter(pure_tf_plot[:, 0], pure_tf_plot[:, 1], label='PureTF+Physics', s=12, alpha=0.85)
    plt.scatter(tgnn_tf_plot[:, 0], tgnn_tf_plot[:, 1], label='TGNN-Transformer', s=12, alpha=0.85)
    plt.scatter(tgnn_plot[:, 0], tgnn_plot[:, 1], label='TGNN', s=12, alpha=0.85)
    plt.scatter(gnn_plot[:, 0], gnn_plot[:, 1], label='GCN-6D', s=12, alpha=0.8)
    plt.scatter(e2_plot[:, 0], e2_plot[:, 1], label='EKF-Joint', s=12, alpha=0.8)
    plt.title('Trajectory Comparison: COLMAP vs 7 Methods (Seq6, x~0 Focus)')
    plt.xlabel('x (m, centered at COLMAP first point)')
    plt.ylabel('y (m, centered at COLMAP first point)')
    plt.xlim(-1.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p1 = out_dir / 'compare_traj_seq6_7methods_x-1_1_scatter.png'
    plt.savefig(p1, dpi=170)
    plt.close()

    # Error curves
    err_ttfed = np.linalg.norm(pred_tgnn_tf_ed - gt_xy, axis=1)
    err_ptf0 = np.linalg.norm(pred_pure_tf0 - gt_xy, axis=1)
    err_ptf = np.linalg.norm(pred_pure_tf - gt_xy, axis=1)
    err_ttf = np.linalg.norm(pred_tgnn_tf - gt_xy, axis=1)
    err_tg = np.linalg.norm(pred_tgnn - gt_xy, axis=1)
    err_g = np.linalg.norm(pred_gnn - gt_xy, axis=1)
    err_e2 = np.linalg.norm(pred_e2 - gt_xy, axis=1)
    plt.figure(figsize=(9, 4))
    plt.plot(err_ttfed, label='TGNN-Transformer-ED')
    plt.plot(err_ptf0, label='PureTF')
    plt.plot(err_ptf, label='PureTF+Physics')
    plt.plot(err_ttf, label='TGNN-Transformer')
    plt.plot(err_tg, label='TGNN')
    plt.plot(err_g, label='GCN-6D')
    plt.plot(err_e2, label='EKF-Joint')
    plt.title('Per-step Error Comparison')
    plt.xlabel('Test step')
    plt.ylabel('Error (m)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p2 = out_dir / 'compare_error_curve_seq6.png'
    plt.savefig(p2, dpi=170)
    plt.close()

    # CDF (first-point aligned for fair shape comparison)
    pred_tgnn_tf_ed_a = pred_tgnn_tf_ed - (pred_tgnn_tf_ed[0] - gt_xy[0])
    pred_pure_tf0_a = pred_pure_tf0 - (pred_pure_tf0[0] - gt_xy[0])
    pred_pure_tf_a = pred_pure_tf - (pred_pure_tf[0] - gt_xy[0])
    pred_tgnn_tf_a = pred_tgnn_tf - (pred_tgnn_tf[0] - gt_xy[0])
    pred_tgnn_a = pred_tgnn - (pred_tgnn[0] - gt_xy[0])
    pred_gnn_a = pred_gnn - (pred_gnn[0] - gt_xy[0])
    pred_e2_a = pred_e2 - (pred_e2[0] - gt_xy[0])
    err_ttfed_a = np.linalg.norm(pred_tgnn_tf_ed_a - gt_xy, axis=1)
    err_ptf0_a = np.linalg.norm(pred_pure_tf0_a - gt_xy, axis=1)
    err_ptf_a = np.linalg.norm(pred_pure_tf_a - gt_xy, axis=1)
    err_ttf_a = np.linalg.norm(pred_tgnn_tf_a - gt_xy, axis=1)
    err_tg_a = np.linalg.norm(pred_tgnn_a - gt_xy, axis=1)
    err_g_a = np.linalg.norm(pred_gnn_a - gt_xy, axis=1)
    err_e2_a = np.linalg.norm(pred_e2_a - gt_xy, axis=1)

    def cdf_xy(err):
        s = np.sort(err)
        y = np.arange(1, len(s) + 1) / len(s)
        return s, y
    plt.figure(figsize=(7, 5))
    for err, name in [(err_ttfed_a, 'TGNN-Transformer-ED'), (err_ptf0_a, 'PureTF'), (err_ptf_a, 'PureTF+Physics'), (err_ttf_a, 'TGNN-Transformer'), (err_tg_a, 'TGNN'), (err_g_a, 'GCN-6D'), (err_e2_a, 'EKF-Joint')]:
        sx, sy = cdf_xy(err)
        plt.plot(sx, sy, label=name)
    plt.title('Error CDF Comparison (First-point Aligned)')
    plt.xlabel('Error (m)')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p3 = out_dir / 'compare_error_cdf_seq6.png'
    plt.savefig(p3, dpi=170)
    plt.close()

    # Bar chart RMSE/ADE
    names = ['TGNN-TF-ED', 'PureTF', 'PureTF+Phy', 'TGNN-TF', 'TGNN', 'GCN-6D', 'EKF-Joint']
    ade = [m_tgnn_tf_ed['mean_error'], m_pure_tf0['mean_error'], m_pure_tf['mean_error'], m_tgnn_tf['mean_error'], m_tgnn['mean_error'], m_gnn['mean_error'], m_e2['mean_error']]
    rmse = [m_tgnn_tf_ed['rmse'], m_pure_tf0['rmse'], m_pure_tf['rmse'], m_tgnn_tf['rmse'], m_tgnn['rmse'], m_gnn['rmse'], m_e2['rmse']]
    x = np.arange(len(names))
    w = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - w/2, ade, width=w, label='ADE')
    plt.bar(x + w/2, rmse, width=w, label='RMSE')
    plt.xticks(x, names)
    plt.ylabel('Error (m)')
    plt.title('Metric Comparison')
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    p4 = out_dir / 'compare_metrics_bar_seq6.png'
    plt.savefig(p4, dpi=170)
    plt.close()

    print('=== Comparison Metrics (same metric definition) ===')
    for name, m in [('TGNN-Transformer-ED', m_tgnn_tf_ed), ('PureTF', m_pure_tf0), ('PureTF+Physics', m_pure_tf), ('TGNN-Transformer', m_tgnn_tf), ('TGNN', m_tgnn), ('GCN-6D', m_gnn), ('EKF-Joint', m_e2)]:
        print(f"{name:11s} | mean={m['mean_error']:.4f} m | rmse={m['rmse']:.4f} m | "
              f"bias_x={m['bias_x']:.4f} m | bias_y={m['bias_y']:.4f} m | aligned_rmse={m['aligned_rmse']:.4f} m")

    print('Saved plots:')
    for p in [p1, p2, p3, p4]:
        print(p)


if __name__ == '__main__':
    main()
