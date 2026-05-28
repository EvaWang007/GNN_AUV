import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def latlon_to_local_xy_m(lat_deg: np.ndarray, lon_deg: np.ndarray):
    lat0 = np.deg2rad(lat_deg[0])
    lon0 = np.deg2rad(lon_deg[0])
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    r_earth = 6371000.0
    x = (lon - lon0) * np.cos(lat0) * r_earth
    y = (lat - lat0) * r_earth
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Build 6D AUV dataset: [x,y,v,a,theta,delay] + delay_mask")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--sound_speed", type=float, default=1500.0)
    parser.add_argument("--delay_noise_std", type=float, default=0.002)
    parser.add_argument("--delay_keep_every", type=int, default=2, help="keep one delay every k steps")
    parser.add_argument("--anchor_x", type=float, default=None)
    parser.add_argument("--anchor_y", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    input_csv = Path(args.input_csv)
    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    required_cols = ["lat", "lon", "yaw", "vn", "ve", "ax", "ay", "time"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[required_cols].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

    lat = data["lat"].to_numpy(dtype=np.float64)
    lon = data["lon"].to_numpy(dtype=np.float64)
    yaw = data["yaw"].to_numpy(dtype=np.float64)
    vn = data["vn"].to_numpy(dtype=np.float64)
    ve = data["ve"].to_numpy(dtype=np.float64)
    ax = data["ax"].to_numpy(dtype=np.float64)
    ay = data["ay"].to_numpy(dtype=np.float64)

    x_nav, y_nav = latlon_to_local_xy_m(lat, lon)
    v = np.sqrt(vn * vn + ve * ve)
    a = np.sqrt(ax * ax + ay * ay)
    theta = yaw

    # Build pseudo delay from a fixed anchor (master AUV)
    anchor_x = float(args.anchor_x) if args.anchor_x is not None else float(np.mean(x_nav[:200]))
    anchor_y = float(args.anchor_y) if args.anchor_y is not None else float(np.mean(y_nav[:200]))

    dist = np.sqrt((x_nav - anchor_x) ** 2 + (y_nav - anchor_y) ** 2)
    delay_true = dist / args.sound_speed
    delay_noisy = delay_true + rng.normal(0.0, args.delay_noise_std, size=delay_true.shape[0])

    delay_mask = np.zeros_like(delay_noisy, dtype=np.int64)
    delay_mask[:: args.delay_keep_every] = 1
    delay_obs = np.where(delay_mask == 1, delay_noisy, 0.0)

    # One-step supervised
    X = np.stack([x_nav[:-1], y_nav[:-1], v[:-1], a[:-1], theta[:-1], delay_obs[:-1]], axis=1)
    Y = np.stack([x_nav[1:], y_nav[1:]], axis=1)
    M = delay_mask[:-1]

    needed = args.train_samples + args.test_samples
    if X.shape[0] < needed:
        raise ValueError(f"Not enough samples: have {X.shape[0]}, need {needed}")

    X_train = X[: args.train_samples]
    Y_train = Y[: args.train_samples]
    M_train = M[: args.train_samples]

    X_test = X[args.train_samples : args.train_samples + args.test_samples]
    Y_test = Y[args.train_samples : args.train_samples + args.test_samples]
    M_test = M[args.train_samples : args.train_samples + args.test_samples]

    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0

    # Delay normalization should ignore missing samples (mask==0)
    valid_delay = X_train[M_train == 1, 5]
    if valid_delay.shape[0] > 0:
        feat_mean[5] = valid_delay.mean()
        feat_std[5] = max(valid_delay.std(), 1e-8)

    X_train_norm = (X_train - feat_mean) / feat_std
    X_test_norm = (X_test - feat_mean) / feat_std

    # keep missing delay as 0 after normalization
    X_train_norm[M_train == 0, 5] = 0.0
    X_test_norm[M_test == 0, 5] = 0.0

    np.savez(
        output_npz,
        X_train=X_train_norm.astype(np.float32),
        Y_train=Y_train.astype(np.float32),
        M_train=M_train.astype(np.int64),
        X_test=X_test_norm.astype(np.float32),
        Y_test=Y_test.astype(np.float32),
        M_test=M_test.astype(np.int64),
        feat_mean=feat_mean.astype(np.float32),
        feat_std=feat_std.astype(np.float32),
        anchor_x=np.float32(anchor_x),
        anchor_y=np.float32(anchor_y),
        source_csv=str(input_csv),
    )

    print(f"Saved: {output_npz}")
    print(f"Train: X={X_train_norm.shape}, Y={Y_train.shape}, M={M_train.shape}")
    print(f"Test : X={X_test_norm.shape}, Y={Y_test.shape}, M={M_test.shape}")
    print(f"Anchor: ({anchor_x:.3f}, {anchor_y:.3f})")


if __name__ == "__main__":
    main()
