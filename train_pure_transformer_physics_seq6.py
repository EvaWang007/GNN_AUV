import argparse
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def build_windows(X, Y, M, window_size=20):
    xs, ms, ys = [], [], []
    for t in range(window_size - 1, X.shape[0]):
        xs.append(X[t - window_size + 1 : t + 1])  # [W,6]
        ms.append(M[t - window_size + 1 : t + 1])  # [W]
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
        # x_seq [B,W,F]
        h = self.embed(x_seq)
        h = self.pos(h)
        h = self.encoder(h)
        h_last = h[:, -1, :]
        y = F.relu(self.fc1(h_last))
        y = self.drop(y)
        y = self.fc2(y)
        return y


def physics_losses(xb, pred_xy, dt=1.0, sound_speed=1500.0, lambda_kin=0.0, lambda_smooth=0.0, lambda_range=0.0):
    """
    xb: [B,W,6], feature order assumed [x, y, v, a, theta, delay] in standardized space.
    pred_xy: [B,2] predicted absolute xy in standardized target space.
    """
    device = pred_xy.device
    zeros = torch.tensor(0.0, device=device)

    # last observed state in window
    x_prev = xb[:, -1, 0]
    y_prev = xb[:, -1, 1]
    v_last = xb[:, -1, 2]
    theta_last = xb[:, -1, 4]
    delay_last = xb[:, -1, 5]

    l_kin = zeros
    if lambda_kin > 0:
        dx = pred_xy[:, 0] - x_prev
        dy = pred_xy[:, 1] - y_prev
        dx_phy = v_last * torch.cos(theta_last) * dt
        dy_phy = v_last * torch.sin(theta_last) * dt
        l_kin = F.mse_loss(dx, dx_phy) + F.mse_loss(dy, dy_phy)

    l_smooth = zeros
    if lambda_smooth > 0 and xb.shape[1] >= 3:
        x_t = xb[:, -1, 0]
        x_t1 = xb[:, -2, 0]
        x_t2 = xb[:, -3, 0]
        y_t = xb[:, -1, 1]
        y_t1 = xb[:, -2, 1]
        y_t2 = xb[:, -3, 1]
        # second difference continuity including predicted point
        ddx = pred_xy[:, 0] - 2 * x_t + x_t1
        ddy = pred_xy[:, 1] - 2 * y_t + y_t1
        prev_ddx = x_t - 2 * x_t1 + x_t2
        prev_ddy = y_t - 2 * y_t1 + y_t2
        l_smooth = F.mse_loss(ddx, prev_ddx) + F.mse_loss(ddy, prev_ddy)

    l_range = zeros
    if lambda_range > 0:
        # simple range consistency to origin in normalized space as weak prior placeholder.
        # mask only where delay exists
        mask = (delay_last != 0).float()
        if mask.sum() > 0:
            # approximate measured range proxy by c*delay in normalized feature space scale unknown,
            # so use relative consistency: larger delay -> larger distance via rank-preserving mse.
            r_pred = torch.sqrt(pred_xy[:, 0] ** 2 + pred_xy[:, 1] ** 2 + 1e-6)
            d_proxy = torch.abs(delay_last) * sound_speed
            # normalize proxy to prevent scale explosion
            d_proxy = d_proxy / (d_proxy.mean().detach() + 1e-6)
            r_pred = r_pred / (r_pred.mean().detach() + 1e-6)
            l_range = (((r_pred - d_proxy) ** 2) * mask).sum() / (mask.sum() + 1e-6)

    return l_kin, l_smooth, l_range


def eval_loader(model, loader, device):
    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, mb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
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
    parser = argparse.ArgumentParser(description="Train pure Transformer + physics constraints on seq6")
    parser.add_argument("--data_npz", default="/home/evawang/T-GCN/data/lbl_aqualoc_seq6_6tuple_train4000_test1000.npz")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lambda_kin", type=float, default=0.1)
    parser.add_argument("--lambda_smooth", type=float, default=0.05)
    parser.add_argument("--lambda_range", type=float, default=0.01)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--model_out", default="/home/evawang/T-GCN/data/auv_pure_tf_phy_seq6_model.pt")
    parser.add_argument("--history_out", default="/home/evawang/T-GCN/data/auv_pure_tf_phy_seq6_history.csv")
    parser.add_argument("--curve_out", default="/home/evawang/T-GCN/data/auv_pure_tf_phy_seq6_loss_curve.png")
    args = parser.parse_args()

    d = np.load(args.data_npz, allow_pickle=True)
    X_train, Y_train, M_train = d["X_train"], d["Y_train"], d["M_train"]
    X_test, Y_test, M_test = d["X_test"], d["Y_test"], d["M_test"]

    Xw_train, Mw_train, Yw_train = build_windows(X_train, Y_train, M_train, window_size=args.window_size)
    Xw_test, Mw_test, Yw_test = build_windows(X_test, Y_test, M_test, window_size=args.window_size)

    if args.train_limit > 0:
        Xw_train, Mw_train, Yw_train = Xw_train[:args.train_limit], Mw_train[:args.train_limit], Yw_train[:args.train_limit]
    if args.test_limit > 0:
        Xw_test, Mw_test, Yw_test = Xw_test[:args.test_limit], Mw_test[:args.test_limit], Yw_test[:args.test_limit]

    train_ds = TensorDataset(
        torch.tensor(Xw_train, dtype=torch.float32),
        torch.tensor(Mw_train, dtype=torch.int64),
        torch.tensor(Yw_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(Xw_test, dtype=torch.float32),
        torch.tensor(Mw_test, dtype=torch.int64),
        torch.tensor(Yw_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PureTransformerRegressor(
        input_dim=Xw_train.shape[-1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_rmse = float("inf")
    best_epoch = -1
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_mse_sum, train_n = 0.0, 0

        for xb, mb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            l_data = F.mse_loss(pred, yb)
            l_kin, l_smooth, l_range = physics_losses(
                xb, pred,
                lambda_kin=args.lambda_kin,
                lambda_smooth=args.lambda_smooth,
                lambda_range=args.lambda_range,
            )
            loss = l_data + args.lambda_kin * l_kin + args.lambda_smooth * l_smooth + args.lambda_range * l_range
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            train_mse_sum += l_data.item() * bs
            train_n += bs

        train_mse = train_mse_sum / max(train_n, 1)
        train_rmse = float(np.sqrt(train_mse))
        test_mse, test_rmse, test_mae = eval_loader(model, test_loader, device)
        history.append([epoch, train_mse, train_rmse, test_mse, test_rmse, test_mae])

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | train_mse={train_mse:.6f} train_rmse={train_rmse:.6f} | "
                f"test_mse={test_mse:.6f} test_rmse={test_rmse:.6f} test_mae={test_mae:.6f}"
            )

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
            "window_size": args.window_size,
            "best_test_rmse": best_rmse,
            "best_epoch": best_epoch,
            "data_npz": args.data_npz,
        },
        out,
    )

    hist = np.asarray(history, dtype=np.float64)
    hist_path = Path(args.history_out)
    np.savetxt(hist_path, hist, delimiter=",", header="epoch,train_mse,train_rmse,test_mse,test_rmse,test_mae", comments="")

    plt.figure(figsize=(8, 4))
    plt.plot(hist[:, 0], hist[:, 2], label="train_rmse")
    plt.plot(hist[:, 0], hist[:, 4], label="test_rmse")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Pure Transformer+Physics Training Curve (RMSE vs Epoch)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    curve = Path(args.curve_out)
    plt.savefig(curve, dpi=170)
    plt.close()

    print(f"Saved best model to: {out}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best test RMSE: {best_rmse:.6f}")
    print(f"Saved history csv to: {hist_path}")
    print(f"Saved loss curve to: {curve}")


if __name__ == "__main__":
    main()
