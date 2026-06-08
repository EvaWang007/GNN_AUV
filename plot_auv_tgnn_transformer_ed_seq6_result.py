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
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


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

    def encode_graph_step(self, x_step, edge_index, edge_weight):
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
            embs.append(self.encode_graph_step(x_step, edge_index, edge_weight))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_npz', default='/home/evawang/T-GCN/data/lbl_aqualoc_seq6_6tuple_train4000_test1000.npz')
    parser.add_argument('--model_path', default='/home/evawang/T-GCN/data/auv_tgnn_transformer_ed_seq6_model.pt')
    parser.add_argument('--out_dir', default='/home/evawang/T-GCN/data')
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()

    d = np.load(args.data_npz, allow_pickle=True)
    X_test, Y_test, M_test = d['X_test'], d['Y_test'], d['M_test']

    ckpt = torch.load(args.model_path, map_location='cpu')
    model = TGNNTransformerEDRegressor(
        gcn_hidden=int(ckpt.get('gcn_hidden', 128)),
        d_model=int(ckpt.get('d_model', 128)),
        nhead=int(ckpt.get('nhead', 4)),
        num_layers=int(ckpt.get('num_layers', 2)),
        ff_dim=int(ckpt.get('ff_dim', 256)),
        dropout=float(ckpt.get('dropout', 0.2)),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    window_size = int(ckpt.get('window_size', 20))
    Xw, Mw, Yw = build_windows(X_test, Y_test, M_test, window_size=window_size)

    edge_index_np = build_edge_index(6)
    preds = []
    with torch.no_grad():
        for i in range(Xw.shape[0]):
            preds.append(model.forward_one_sequence(Xw[i], Mw[i], edge_index_np).numpy())
    preds = np.asarray(preds)

    err = np.linalg.norm(preds - Yw, axis=1)
    rmse = float(np.sqrt(np.mean(err**2)))

    plt.figure(figsize=(6, 6))
    plt.plot(Yw[:, 0], Yw[:, 1], label='COLMAP Ground Truth')
    plt.plot(preds[:, 0], preds[:, 1], label='TGNN-Transformer-ED Prediction')
    plt.title('TGNN-Transformer-ED Trajectory: COLMAP vs Pred')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p1 = f"{args.out_dir}/plot_traj_tgnn_transformer_ed_seq6{args.suffix}.png"
    plt.savefig(p1, dpi=170)
    plt.close()

    print('Saved figure:')
    print(p1)
    print(f'RMSE: {rmse:.6f} m')


if __name__ == '__main__':
    main()
