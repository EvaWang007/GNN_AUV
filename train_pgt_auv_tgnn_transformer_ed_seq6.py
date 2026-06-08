import argparse
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
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

    def forward(self, x_seq, m_seq, edge_index_np, device):
        bsz, win, _ = x_seq.shape
        edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
        seq_embeddings = []
        for b in range(bsz):
            step_embs = []
            for t in range(win):
                x_step = x_seq[b, t].reshape(6, 1)
                ew = edge_weight_with_delay_mask(edge_index_np, int(m_seq[b, t].item()))
                edge_weight = torch.tensor(ew, dtype=torch.float32, device=device)
                z = self.encode_graph_step(x_step, edge_index, edge_weight)
                step_embs.append(z)
            seq_embeddings.append(torch.stack(step_embs, dim=0))

        src = torch.stack(seq_embeddings, dim=0)
        src = self.proj(src)
        src = self.pos(src)

        tgt = self.start_token.expand(bsz, 1, -1)
        tgt = self.pos(tgt)

        out = self.transformer(src=src, tgt=tgt)  # [B,1,D]
        h = out[:, -1, :]
        y = F.relu(self.fc1(h))
        y = self.drop(y)
        y = self.fc2(y)
        return y


def eval_loader(model, loader, edge_index_np, device):
    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            pred = model(xb, mb, edge_index_np, device)
            mse = F.mse_loss(pred, yb)
            mae = F.l1_loss(pred, yb)
            bs = xb.size(0)
            mse_sum += mse.item() * bs
            mae_sum += mae.item() * bs
            n += bs
    mse = mse_sum / max(n, 1)
    rmse = float(np.sqrt(mse))
    mae = mae_sum / max(n, 1)
    return mse, rmse, mae


def main():
    parser = argparse.ArgumentParser(description='Train TGNN Transformer Encoder-Decoder on seq6')
    parser.add_argument('--data_npz', default='/home/evawang/T-GCN/data/lbl_aqualoc_seq6_6tuple_train4000_test1000.npz')
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gcn_hidden', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--ff_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--train_limit', type=int, default=0)
    parser.add_argument('--test_limit', type=int, default=0)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--model_out', default='/home/evawang/T-GCN/data/auv_tgnn_transformer_ed_seq6_model.pt')
    parser.add_argument('--history_out', default='/home/evawang/T-GCN/data/auv_tgnn_transformer_ed_seq6_history.csv')
    parser.add_argument('--curve_out', default='/home/evawang/T-GCN/data/auv_tgnn_transformer_ed_seq6_loss_curve.png')
    args = parser.parse_args()

    d = np.load(args.data_npz, allow_pickle=True)
    X_train, Y_train, M_train = d['X_train'], d['Y_train'], d['M_train']
    X_test, Y_test, M_test = d['X_test'], d['Y_test'], d['M_test']

    Xw_train, Mw_train, Yw_train = build_windows(X_train, Y_train, M_train, window_size=args.window_size)
    Xw_test, Mw_test, Yw_test = build_windows(X_test, Y_test, M_test, window_size=args.window_size)

    if args.train_limit > 0:
        Xw_train, Mw_train, Yw_train = Xw_train[:args.train_limit], Mw_train[:args.train_limit], Yw_train[:args.train_limit]
    if args.test_limit > 0:
        Xw_test, Mw_test, Yw_test = Xw_test[:args.test_limit], Mw_test[:args.test_limit], Yw_test[:args.test_limit]

    train_ds = TensorDataset(torch.tensor(Xw_train, dtype=torch.float32), torch.tensor(Mw_train, dtype=torch.int64), torch.tensor(Yw_train, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(Xw_test, dtype=torch.float32), torch.tensor(Mw_test, dtype=torch.int64), torch.tensor(Yw_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    edge_index_np = build_full_edge_index(6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TGNNTransformerEDRegressor(
        gcn_hidden=args.gcn_hidden,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_rmse = float('inf')
    best_epoch = -1
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_mse_sum, train_n = 0.0, 0
        for xb, mb, yb in train_loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, mb, edge_index_np, device)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            bs = xb.size(0)
            train_mse_sum += loss.item() * bs
            train_n += bs

        train_mse = train_mse_sum / max(train_n, 1)
        train_rmse = float(np.sqrt(train_mse))
        test_mse, test_rmse, test_mae = eval_loader(model, test_loader, edge_index_np, device)
        history.append([epoch, train_mse, train_rmse, test_mse, test_rmse, test_mae])

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(f'Epoch {epoch:03d} | train_mse={train_mse:.6f} train_rmse={train_rmse:.6f} | test_mse={test_mse:.6f} test_rmse={test_rmse:.6f} test_mae={test_mae:.6f}')

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': best_state,
        'gcn_hidden': args.gcn_hidden,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'ff_dim': args.ff_dim,
        'dropout': args.dropout,
        'window_size': args.window_size,
        'best_test_rmse': best_rmse,
        'best_epoch': best_epoch,
        'data_npz': args.data_npz,
    }, out)

    hist = np.asarray(history, dtype=np.float64)
    hist_path = Path(args.history_out)
    np.savetxt(hist_path, hist, delimiter=',', header='epoch,train_mse,train_rmse,test_mse,test_rmse,test_mae', comments='')

    plt.figure(figsize=(8, 4))
    plt.plot(hist[:, 0], hist[:, 2], label='train_rmse')
    plt.plot(hist[:, 0], hist[:, 4], label='test_rmse')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('TGNN Transformer ED Training Curve (RMSE vs Epoch)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    curve = Path(args.curve_out)
    plt.savefig(curve, dpi=170)
    plt.close()

    print(f'Saved best model to: {out}')
    print(f'Best epoch: {best_epoch}')
    print(f'Best test RMSE: {best_rmse:.6f}')
    print(f'Saved history csv to: {hist_path}')
    print(f'Saved loss curve to: {curve}')


if __name__ == '__main__':
    main()
