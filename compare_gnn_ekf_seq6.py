import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.nn import GCNConv
import torch.nn as nn


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


def main():
    root = Path('/home/evawang/T-GCN/data')
    gnn_data = root / 'lbl_aqualoc_seq6_6tuple_train4000_test1000.npz'
    gnn_model = root / 'auv_gcn6_delay_seq6_model.pt'
    tgnn_model = root / 'auv_tgnn_seq6_model.pt'
    ekf_single = root / 'ekf_seq6_result.npz'
    ekf_joint = root / 'ekf_seq6_joint_result.npz'

    pred_gnn, gt = run_gnn_prediction(gnn_data, gnn_model)
    m_gnn = eval_metrics(pred_gnn, gt)
    pred_tgnn, gt_tgnn = run_tgnn_prediction(gnn_data, tgnn_model)
    m_tgnn = eval_metrics(pred_tgnn, gt_tgnn)

    e1 = np.load(ekf_single, allow_pickle=True)
    pred_e1 = e1['pred_xy']
    gt_e1 = e1['gt_xy']
    m_e1 = eval_metrics(pred_e1, gt_e1)

    e2 = np.load(ekf_joint, allow_pickle=True)
    pred_e2 = e2['pred_xy']
    gt_e2 = e2['gt_xy']
    m_e2 = eval_metrics(pred_e2, gt_e2)

    # Use GT from GNN path for plots
    gt_xy = gt_tgnn
    gnn_offset = len(gt) - len(gt_tgnn)
    pred_gnn = pred_gnn[gnn_offset:]
    pred_e1 = pred_e1[-len(gt_tgnn):]
    pred_e2 = pred_e2[-len(gt_tgnn):]
    n = min(len(gt_xy), len(pred_e1), len(pred_e2), len(pred_gnn), len(pred_tgnn))
    gt_xy = gt_xy[:n]
    pred_tgnn = pred_tgnn[:n]
    pred_gnn = pred_gnn[:n]
    pred_e1 = pred_e1[:n]
    pred_e2 = pred_e2[:n]

    out_dir = root

    # Trajectory (focus near x=0 by using COLMAP-first-point centered frame)
    gt0 = gt_xy[0].copy()
    gt_plot = gt_xy - gt0
    tgnn_plot = pred_tgnn - gt0
    gnn_plot = pred_gnn - gt0
    e1_plot = pred_e1 - gt0
    e2_plot = pred_e2 - gt0

    plt.figure(figsize=(7, 7))
    plt.scatter(gt_plot[:, 0], gt_plot[:, 1], label='COLMAP Ground Truth', s=14, alpha=0.9)
    plt.scatter(tgnn_plot[:, 0], tgnn_plot[:, 1], label='TGNN', s=12, alpha=0.85)
    plt.scatter(gnn_plot[:, 0], gnn_plot[:, 1], label='GCN-6D', s=12, alpha=0.8)
    plt.scatter(e1_plot[:, 0], e1_plot[:, 1], label='EKF-Single', s=12, alpha=0.65)
    plt.scatter(e2_plot[:, 0], e2_plot[:, 1], label='EKF-Joint', s=12, alpha=0.8)
    plt.title('Trajectory Comparison: COLMAP vs TGNN/GNN/EKF (Seq6, x~0 Focus)')
    plt.xlabel('x (m, centered at COLMAP first point)')
    plt.ylabel('y (m, centered at COLMAP first point)')
    plt.xlim(-1.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p1 = out_dir / 'compare_traj_seq6_tgnn_x-1_1_scatter.png'
    plt.savefig(p1, dpi=170)
    plt.close()

    # Error curves
    err_g = np.linalg.norm(pred_gnn - gt_xy, axis=1)
    err_e1 = np.linalg.norm(pred_e1 - gt_xy, axis=1)
    err_e2 = np.linalg.norm(pred_e2 - gt_xy, axis=1)
    plt.figure(figsize=(9, 4))
    err_tg = np.linalg.norm(pred_tgnn - gt_xy, axis=1)
    plt.plot(err_tg, label='TGNN')
    plt.plot(err_g, label='GCN-6D')
    plt.plot(err_e1, label='EKF-Single')
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

    # CDF
    def cdf_xy(err):
        s = np.sort(err)
        y = np.arange(1, len(s) + 1) / len(s)
        return s, y
    plt.figure(figsize=(7, 5))
    for err, name in [(err_tg, 'TGNN'), (err_g, 'GCN-6D'), (err_e1, 'EKF-Single'), (err_e2, 'EKF-Joint')]:
        sx, sy = cdf_xy(err)
        plt.plot(sx, sy, label=name)
    plt.title('Error CDF Comparison')
    plt.xlabel('Error (m)')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p3 = out_dir / 'compare_error_cdf_seq6.png'
    plt.savefig(p3, dpi=170)
    plt.close()

    # Bar chart RMSE/ADE
    names = ['TGNN', 'GCN-6D', 'EKF-Single', 'EKF-Joint']
    ade = [m_tgnn['mean_error'], m_gnn['mean_error'], m_e1['mean_error'], m_e2['mean_error']]
    rmse = [m_tgnn['rmse'], m_gnn['rmse'], m_e1['rmse'], m_e2['rmse']]
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
    for name, m in [('TGNN', m_tgnn), ('GCN-6D', m_gnn), ('EKF-Single', m_e1), ('EKF-Joint', m_e2)]:
        print(f"{name:11s} | mean={m['mean_error']:.4f} m | rmse={m['rmse']:.4f} m | "
              f"bias_x={m['bias_x']:.4f} m | bias_y={m['bias_y']:.4f} m | aligned_rmse={m['aligned_rmse']:.4f} m")

    print('Saved plots:')
    for p in [p1, p2, p3, p4]:
        print(p)


if __name__ == '__main__':
    main()
