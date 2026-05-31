import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.nn import GCNConv


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
    return np.asarray(preds), Y_test


def run_tgnn_prediction(data_npz, model_pt):
    d = np.load(data_npz, allow_pickle=True)
    X_test = d['X_test']
    Y_test = d['Y_test']
    M_test = d['M_test']

    ckpt = torch.load(model_pt, map_location='cpu')
    gcn_hidden = int(ckpt.get('gcn_hidden', 128))
    lstm_hidden = int(ckpt.get('lstm_hidden', 128))
    dropout = float(ckpt.get('dropout', 0.2))
    window_size = int(ckpt.get('window_size', 20))

    Xw, Mw, Yw = build_windows(X_test, Y_test, M_test, window_size=window_size)

    model = TGNNRegressor(gcn_hidden=gcn_hidden, lstm_hidden=lstm_hidden, dropout=dropout)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    edge_index_np = build_edge_index(6)
    preds = []
    with torch.no_grad():
        for i in range(Xw.shape[0]):
            yhat = model.forward_one_sequence(Xw[i], Mw[i], edge_index_np).numpy()
            preds.append(yhat)
    return np.asarray(preds), Yw


def main():
    root = Path('/home/evawang/T-GCN/data')
    data_npz = root / 'lbl_aqualoc_seq6_6tuple_train4000_test1000.npz'
    gnn_model = root / 'auv_gcn6_delay_seq6_model.pt'
    tgnn_model = root / 'auv_tgnn_seq6_model.pt'
    ekf_joint_npz = root / 'ekf_seq6_joint_result.npz'

    pred_gnn, gt_gnn = run_gnn_prediction(data_npz, gnn_model)
    pred_tgnn, gt_tgnn = run_tgnn_prediction(data_npz, tgnn_model)
    e = np.load(ekf_joint_npz, allow_pickle=True)
    pred_ekf = e['pred_xy']

    # align time bases to TGNN (shorter because of window)
    n = len(gt_tgnn)
    gt = gt_tgnn
    pred_tgnn = pred_tgnn[:n]
    pred_gnn = pred_gnn[-n:]
    pred_ekf = pred_ekf[-n:]

    # first-point alignment: shift each prediction so its first point matches GT first point
    tgnn_aligned = pred_tgnn - (pred_tgnn[0] - gt[0])
    gnn_aligned = pred_gnn - (pred_gnn[0] - gt[0])
    ekf_aligned = pred_ekf - (pred_ekf[0] - gt[0])

    # then center by GT first point for readability
    gt0 = gt[0].copy()
    gt_p = gt - gt0
    tgnn_p = tgnn_aligned - gt0
    gnn_p = gnn_aligned - gt0
    ekf_p = ekf_aligned - gt0

    plt.figure(figsize=(7, 7))
    plt.plot(gt_p[:, 0], gt_p[:, 1], label='COLMAP Ground Truth', linewidth=2)
    plt.plot(tgnn_p[:, 0], tgnn_p[:, 1], label='TGNN')
    plt.plot(gnn_p[:, 0], gnn_p[:, 1], label='GNN (GCN-6D)')
    plt.plot(ekf_p[:, 0], ekf_p[:, 1], label='EKF-Joint')
    plt.title('Trajectory Comparison (First-point Aligned): TGNN vs GNN vs EKF')
    plt.xlabel('x (m, centered)')
    plt.ylabel('y (m, centered)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = root / 'compare_traj_tgnn_gnn_ekf_seq6_aligned.png'
    plt.savefig(out, dpi=180)
    plt.close()

    print('Saved figure:')
    print(out)


if __name__ == '__main__':
    main()
