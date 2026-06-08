import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = x + self.pe[:, :x.size(1), :]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz", default="/home/evawang/T-GCN/data/lbl_aqualoc_seq6_6tuple_train4000_test1000.npz")
    parser.add_argument("--model_path", default="/home/evawang/T-GCN/data/auv_pure_tf_seq6_model.pt")
    parser.add_argument("--out_dir", default="/home/evawang/T-GCN/data")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    d = np.load(args.data_npz, allow_pickle=True)
    X_test, Y_test, M_test = d["X_test"], d["Y_test"], d["M_test"]

    ckpt = torch.load(args.model_path, map_location="cpu")
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

    err_xy = preds - Yw
    err = np.linalg.norm(err_xy, axis=1)
    rmse = float(np.sqrt(np.mean(err**2)))
    mean_err = float(np.mean(err))

    offset = preds[0] - Yw[0]
    preds_aligned = preds - offset
    err_a = np.linalg.norm(preds_aligned - Yw, axis=1)
    rmse_a = float(np.sqrt(np.mean(err_a**2)))

    plt.figure(figsize=(6, 6))
    plt.plot(Yw[:, 0], Yw[:, 1], label="COLMAP Ground Truth")
    plt.plot(preds[:, 0], preds[:, 1], label="Pure Transformer")
    plt.title("Pure Transformer Trajectory")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p1 = f"{args.out_dir}/plot_traj_pure_tf_seq6{args.suffix}.png"
    plt.savefig(p1, dpi=170)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(Yw[:, 0], Yw[:, 1], label="COLMAP Ground Truth")
    plt.plot(preds_aligned[:, 0], preds_aligned[:, 1], label="Pure Transformer (Aligned)")
    plt.title("Pure Transformer Trajectory (Aligned)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    p2 = f"{args.out_dir}/plot_traj_pure_tf_seq6_aligned{args.suffix}.png"
    plt.savefig(p2, dpi=170)
    plt.close()

    plt.figure(figsize=(8, 3.5))
    plt.plot(err)
    plt.title("Pure Transformer Per-step Error")
    plt.xlabel("test step")
    plt.ylabel("error (m)")
    plt.grid(True)
    plt.tight_layout()
    p3 = f"{args.out_dir}/plot_error_pure_tf_seq6{args.suffix}.png"
    plt.savefig(p3, dpi=170)
    plt.close()

    print("Saved figures:")
    print(p1)
    print(p2)
    print(p3)
    print(f"Mean error: {mean_err:.6f} m")
    print(f"RMSE: {rmse:.6f} m")
    print(f"Aligned RMSE: {rmse_a:.6f} m")


if __name__ == "__main__":
    main()
