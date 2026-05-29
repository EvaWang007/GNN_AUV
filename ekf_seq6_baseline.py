import argparse
from pathlib import Path
import numpy as np


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def denorm_features(X_norm: np.ndarray, feat_mean: np.ndarray, feat_std: np.ndarray) -> np.ndarray:
    return X_norm * feat_std.reshape(1, -1) + feat_mean.reshape(1, -1)


def fit_similarity_2d(src_xy: np.ndarray, dst_xy: np.ndarray):
    """Estimate 2D similarity transform: dst ~= s * R @ src + t."""
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    if src.shape[0] < 2:
        return 1.0, np.eye(2), np.zeros(2)

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    scale = 1.0 if var_src < 1e-12 else float(np.sum(S) / var_src)
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def apply_similarity_2d(xy: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    return (scale * (xy @ R.T)) + t.reshape(1, 2)


def run_ekf_on_seq6_3d(
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    M: np.ndarray,
    t: np.ndarray,
    buoy_x: float,
    buoy_y: float,
    buoy_z: float,
    sound_speed: float,
    q_pos: float,
    q_z: float,
    q_v: float,
    q_theta: float,
    r_range: float,
):
    """
    EKF state: [x, y, z, v, theta]
    Input per step k: [x_nav, y_nav, v_obs, a_obs, theta_obs, delay_obs]
    Measurement (when M[k]==1): z_k = delay_k * c = 3D slant range to buoy
    Label target Y[k] = [x_gt(k+1), y_gt(k+1)]
    """
    n = X.shape[0]

    # init from first input state
    x0, y0, v0, _, th0, _ = X[0]
    z0 = float(Z[0])
    s = np.array([x0, y0, z0, max(v0, 0.0), th0], dtype=np.float64)

    P = np.diag([10.0, 10.0, 25.0, 2.0, 0.5]).astype(np.float64)

    Q = np.diag([q_pos, q_pos, q_z, q_v, q_theta]).astype(np.float64)

    pred_xy = np.zeros((n, 2), dtype=np.float64)

    for k in range(n):
        x_nav, y_nav, v_obs, a_obs, theta_obs, delay_obs = X[k]

        if k < n - 1:
            dt = max(float(t[k + 1] - t[k]), 1e-3)
        else:
            dt = max(float(t[k] - t[k - 1]), 1e-3)

        # ---------------------------
        # Predict
        # ---------------------------
        x, y, z, v, th = s

        # Motion update using acceleration input
        v_pred = max(v + a_obs * dt, 0.0)
        x_pred = x + v_pred * np.cos(th) * dt
        y_pred = y + v_pred * np.sin(th) * dt
        # z channel is treated as slowly-varying and corrected by observed depth/altitude proxy
        z_obs = float(Z[k])
        z_pred = 0.9 * z + 0.1 * z_obs

        # Blend heading with observed heading to reduce drift
        th_pred = wrap_angle(0.9 * th + 0.1 * theta_obs)

        s_pred = np.array([x_pred, y_pred, z_pred, v_pred, th_pred], dtype=np.float64)

        # Jacobian F = df/ds
        F = np.eye(5, dtype=np.float64)
        F[0, 3] = np.cos(th) * dt
        F[0, 4] = -v_pred * np.sin(th) * dt
        F[1, 3] = np.sin(th) * dt
        F[1, 4] = v_pred * np.cos(th) * dt
        # z_pred = 0.9*z + 0.1*z_obs -> dz/dz = 0.9
        F[2, 2] = 0.9

        P_pred = F @ P @ F.T + Q

        # ---------------------------
        # Update (range measurement)
        # ---------------------------
        if int(M[k]) == 1:
            z = float(delay_obs) * sound_speed

            dx = s_pred[0] - buoy_x
            dy = s_pred[1] - buoy_y
            dz = s_pred[2] - buoy_z
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            d = max(d, 1e-6)

            h = d
            H = np.zeros((1, 5), dtype=np.float64)
            H[0, 0] = dx / d
            H[0, 1] = dy / d
            H[0, 2] = dz / d

            R = np.array([[r_range]], dtype=np.float64)
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)

            y_res = np.array([z - h], dtype=np.float64)
            s_upd = s_pred + (K @ y_res).reshape(-1)
            P_upd = (np.eye(5) - K @ H) @ P_pred

            s = s_upd
            s[4] = wrap_angle(s[4])
            P = P_upd
        else:
            s = s_pred
            P = P_pred

        pred_xy[k, 0] = s[0]
        pred_xy[k, 1] = s[1]

    gt_xy = Y.astype(np.float64)
    err_xy = pred_xy - gt_xy
    err = np.linalg.norm(err_xy, axis=1)

    mean_error = float(np.mean(err))
    rmse_pos = float(np.sqrt(np.mean(err ** 2)))
    bias_x = float(np.mean(err_xy[:, 0]))
    bias_y = float(np.mean(err_xy[:, 1]))

    # First-point aligned metrics
    offset = pred_xy[0] - gt_xy[0]
    pred_aligned = pred_xy - offset
    err_a = np.linalg.norm(pred_aligned - gt_xy, axis=1)
    mean_error_a = float(np.mean(err_a))
    rmse_pos_a = float(np.sqrt(np.mean(err_a ** 2)))

    return {
        "pred_xy": pred_xy,
        "gt_xy": gt_xy,
        "err": err,
        "mean_error": mean_error,
        "rmse_pos": rmse_pos,
        "bias_x": bias_x,
        "bias_y": bias_y,
        "mean_error_aligned": mean_error_a,
        "rmse_pos_aligned": rmse_pos_a,
    }


def main():
    parser = argparse.ArgumentParser(description="EKF baseline on LBL-AQUALOC seq6 six-tuple dataset")
    parser.add_argument("--data_npz", default="/home/evawang/T-GCN/data/lbl_aqualoc_seq6_6tuple_train4000_test1000.npz")
    parser.add_argument("--buoy_x", type=float, default=-150.0)
    parser.add_argument("--buoy_y", type=float, default=75.0)
    parser.add_argument("--buoy_z", type=float, default=-377.0)
    parser.add_argument("--sound_speed", type=float, default=1452.57)
    parser.add_argument("--q_pos", type=float, default=0.05)
    parser.add_argument("--q_z", type=float, default=0.2)
    parser.add_argument("--q_v", type=float, default=0.1)
    parser.add_argument("--q_theta", type=float, default=0.01)
    parser.add_argument("--r_range", type=float, default=4.0)
    parser.add_argument("--z_const", type=float, default=-2.0, help="Fallback z if npz has no Z_test.")
    parser.add_argument("--save_npz", default="/home/evawang/T-GCN/data/ekf_seq6_result.npz")
    args = parser.parse_args()

    data_path = Path(args.data_npz)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    d = np.load(data_path, allow_pickle=True)
    X_train_n = d["X_train"]
    Y_train = d["Y_train"]
    M_train = d["M_train"]
    X_test_n = d["X_test"]
    Y_test = d["Y_test"]
    M_test = d["M_test"]
    t_test = d["t_test"]
    feat_mean = d["feat_mean"]
    feat_std = d["feat_std"]

    # use raw-valued features in EKF
    X_train = denorm_features(X_train_n, feat_mean, feat_std)
    X_test = denorm_features(X_test_n, feat_mean, feat_std)
    # keep missing delay at zero
    X_train[M_train == 0, 5] = 0.0
    X_test[M_test == 0, 5] = 0.0

    # Z source for 3D EKF:
    # 1) prefer npz key 'Z_test' (and optional 'Z_train')
    # 2) fallback to constant depth/altitude proxy
    if "Z_test" in d.files:
        Z_test = d["Z_test"].astype(np.float64)
        if Z_test.shape[0] != X_test.shape[0]:
            raise ValueError(f"Z_test length {Z_test.shape[0]} != X_test length {X_test.shape[0]}")
    else:
        Z_test = np.full((X_test.shape[0],), float(args.z_const), dtype=np.float64)

    # -----------------------------------
    # Coordinate alignment (fix -> colmap GT)
    # -----------------------------------
    src_train_xy = X_train[:, :2]
    dst_train_xy = Y_train
    scale, R_align, t_align = fit_similarity_2d(src_train_xy, dst_train_xy)

    X_train[:, :2] = apply_similarity_2d(X_train[:, :2], scale, R_align, t_align)
    X_test[:, :2] = apply_similarity_2d(X_test[:, :2], scale, R_align, t_align)

    rot_angle = float(np.arctan2(R_align[1, 0], R_align[0, 0]))
    X_train[:, 4] = np.vectorize(wrap_angle)(X_train[:, 4] + rot_angle)
    X_test[:, 4] = np.vectorize(wrap_angle)(X_test[:, 4] + rot_angle)

    buoy_src = np.array([[args.buoy_x, args.buoy_y]], dtype=np.float64)
    buoy_aligned = apply_similarity_2d(buoy_src, scale, R_align, t_align)[0]
    buoy_x_aligned, buoy_y_aligned = float(buoy_aligned[0]), float(buoy_aligned[1])

    res = run_ekf_on_seq6_3d(
        X=X_test,
        Z=Z_test,
        Y=Y_test,
        M=M_test,
        t=t_test,
        buoy_x=buoy_x_aligned,
        buoy_y=buoy_y_aligned,
        buoy_z=args.buoy_z,
        sound_speed=args.sound_speed,
        q_pos=args.q_pos,
        q_z=args.q_z,
        q_v=args.q_v,
        q_theta=args.q_theta,
        r_range=args.r_range,
    )

    out = Path(args.save_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        pred_xy=res["pred_xy"].astype(np.float32),
        gt_xy=res["gt_xy"].astype(np.float32),
        err=res["err"].astype(np.float32),
        mean_error=np.float32(res["mean_error"]),
        rmse_pos=np.float32(res["rmse_pos"]),
        bias_x=np.float32(res["bias_x"]),
        bias_y=np.float32(res["bias_y"]),
        mean_error_aligned=np.float32(res["mean_error_aligned"]),
        rmse_pos_aligned=np.float32(res["rmse_pos_aligned"]),
        buoy_x=np.float32(args.buoy_x),
        buoy_y=np.float32(args.buoy_y),
        buoy_z=np.float32(args.buoy_z),
        buoy_x_aligned=np.float32(buoy_x_aligned),
        buoy_y_aligned=np.float32(buoy_y_aligned),
        align_scale=np.float32(scale),
        align_rot=np.asarray(R_align, dtype=np.float32),
        align_trans=np.asarray(t_align, dtype=np.float32),
        align_rot_angle=np.float32(rot_angle),
        z_const=np.float32(args.z_const),
        sound_speed=np.float32(args.sound_speed),
    )

    print("EKF evaluation on test split:")
    print(
        f"Alignment: scale={scale:.6f}, rot_angle(rad)={rot_angle:.6f}, "
        f"trans=({t_align[0]:.3f}, {t_align[1]:.3f})"
    )
    print(f"Buoy aligned: ({buoy_x_aligned:.3f}, {buoy_y_aligned:.3f})")
    print(f"Mean error: {res['mean_error']:.6f} m")
    print(f"RMSE_pos: {res['rmse_pos']:.6f} m")
    print(f"Mean bias_x: {res['bias_x']:.6f} m")
    print(f"Mean bias_y: {res['bias_y']:.6f} m")
    print(f"Aligned mean error: {res['mean_error_aligned']:.6f} m")
    print(f"Aligned RMSE_pos: {res['rmse_pos_aligned']:.6f} m")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
