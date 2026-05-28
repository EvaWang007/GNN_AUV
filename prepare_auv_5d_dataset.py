import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def latlon_to_local_xy_m(lat_deg: np.ndarray, lon_deg: np.ndarray):
    """Convert lat/lon to local tangent-plane x/y in meters (equirectangular approx)."""
    lat0 = np.deg2rad(lat_deg[0])
    lon0 = np.deg2rad(lon_deg[0])
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    r_earth = 6371000.0
    x = (lon - lon0) * np.cos(lat0) * r_earth
    y = (lat - lat0) * r_earth
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Build 5D AUV dataset: [x_nav,y_nav,v,a,theta] -> [x_next,y_next]")
    parser.add_argument("--input_csv", required=True, help="Path to AUV CSV")
    parser.add_argument("--output_npz", required=True, help="Output .npz path")
    parser.add_argument("--train_samples", type=int, default=4000)
    parser.add_argument("--test_samples", type=int, default=1000)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required_cols = ["lat", "lon", "yaw", "vn", "ve", "ax", "ay", "time"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep required cols, drop invalid rows, and sort by time.
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
    v = np.sqrt(np.square(vn) + np.square(ve))
    a = np.sqrt(np.square(ax) + np.square(ay))
    theta = yaw

    # Build one-step supervised samples.
    # X_t = [x_nav(t), y_nav(t), v(t), a(t), theta(t)]
    # Y_t = [x_nav(t+1), y_nav(t+1)]
    X = np.stack([x_nav[:-1], y_nav[:-1], v[:-1], a[:-1], theta[:-1]], axis=1)
    Y = np.stack([x_nav[1:], y_nav[1:]], axis=1)

    needed = args.train_samples + args.test_samples
    if X.shape[0] < needed:
        raise ValueError(
            f"Not enough samples after cleaning: {X.shape[0]}, need at least {needed}."
        )

    X_train = X[: args.train_samples]
    Y_train = Y[: args.train_samples]
    X_test = X[args.train_samples : args.train_samples + args.test_samples]
    Y_test = Y[args.train_samples : args.train_samples + args.test_samples]

    # Standardize using train statistics only.
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_train_norm = (X_train - feat_mean) / feat_std
    X_test_norm = (X_test - feat_mean) / feat_std

    np.savez(
        output_npz,
        X_train=X_train_norm.astype(np.float32),
        Y_train=Y_train.astype(np.float32),
        X_test=X_test_norm.astype(np.float32),
        Y_test=Y_test.astype(np.float32),
        feat_mean=feat_mean.astype(np.float32),
        feat_std=feat_std.astype(np.float32),
        raw_train_start=0,
        raw_train_end=args.train_samples,
        raw_test_start=args.train_samples,
        raw_test_end=args.train_samples + args.test_samples,
        source_csv=str(input_csv),
    )

    print(f"Saved: {output_npz}")
    print(f"Train: X={X_train_norm.shape}, Y={Y_train.shape}")
    print(f"Test : X={X_test_norm.shape}, Y={Y_test.shape}")


if __name__ == "__main__":
    main()
