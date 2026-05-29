import argparse
from pathlib import Path
import numpy as np

from ekf_seq6_baseline import denorm_features, fit_similarity_2d, apply_similarity_2d, wrap_angle


def run_ekf_joint(
    X, Z, Y, D, M, t, buoy_xyz, sound_speed,
    q_pos, q_z, q_v, q_theta, r_range
):
    # state [x,y,z,v,theta]
    n = X.shape[0]
    x0, y0, v0, a0, th0 = X[0]
    z0 = float(Z[0])
    s = np.array([x0, y0, z0, max(v0,0.0), th0], dtype=np.float64)

    P = np.diag([10.0, 10.0, 25.0, 2.0, 0.5]).astype(np.float64)
    Q = np.diag([q_pos, q_pos, q_z, q_v, q_theta]).astype(np.float64)

    pred = np.zeros((n,2), dtype=np.float64)

    for k in range(n):
        x_nav, y_nav, v_obs, a_obs, th_obs = X[k]
        dt = max(float(t[min(k+1,n-1)] - t[max(k-1,0)]) / (2.0 if 0<k<n-1 else 1.0), 1e-3)

        x, y, z, v, th = s
        v_p = max(v + a_obs * dt, 0.0)
        x_p = x + v_p * np.cos(th) * dt
        y_p = y + v_p * np.sin(th) * dt
        z_p = 0.9*z + 0.1*float(Z[k])
        th_p = wrap_angle(0.9*th + 0.1*th_obs)
        s_p = np.array([x_p,y_p,z_p,v_p,th_p],dtype=np.float64)

        F = np.eye(5)
        F[0,3] = np.cos(th)*dt
        F[0,4] = -v_p*np.sin(th)*dt
        F[1,3] = np.sin(th)*dt
        F[1,4] = v_p*np.cos(th)*dt
        F[2,2] = 0.9
        P_p = F @ P @ F.T + Q

        # joint update with all available anchors
        avail = np.where(M[k] == 1)[0]
        if avail.size > 0:
            H_rows = []
            z_vec = []
            h_vec = []
            for j in avail:
                bx, by, bz = buoy_xyz[j]
                dx = s_p[0] - bx
                dy = s_p[1] - by
                dz = s_p[2] - bz
                d = max(np.sqrt(dx*dx + dy*dy + dz*dz), 1e-6)
                H_rows.append([dx/d, dy/d, dz/d, 0.0, 0.0])
                z_vec.append(float(D[k, j]) * sound_speed)
                h_vec.append(d)
            H = np.asarray(H_rows, dtype=np.float64)               # m x 5
            z_m = np.asarray(z_vec, dtype=np.float64).reshape(-1,1) # m x 1
            h_m = np.asarray(h_vec, dtype=np.float64).reshape(-1,1)
            Rm = np.eye(len(avail), dtype=np.float64) * r_range

            S = H @ P_p @ H.T + Rm
            K = P_p @ H.T @ np.linalg.inv(S)
            y_res = z_m - h_m
            s = s_p + (K @ y_res).reshape(-1)
            s[4] = wrap_angle(s[4])
            P = (np.eye(5) - K @ H) @ P_p
        else:
            s = s_p
            P = P_p

        pred[k,0] = s[0]
        pred[k,1] = s[1]

    err_xy = pred - Y
    err = np.linalg.norm(err_xy, axis=1)
    mean_error = float(err.mean())
    rmse_pos = float(np.sqrt(np.mean(err**2)))
    bias_x = float(err_xy[:,0].mean())
    bias_y = float(err_xy[:,1].mean())

    off = pred[0] - Y[0]
    pred_a = pred - off
    err_a = np.linalg.norm(pred_a - Y, axis=1)

    return {
        'pred_xy': pred,
        'gt_xy': Y,
        'err': err,
        'mean_error': mean_error,
        'rmse_pos': rmse_pos,
        'bias_x': bias_x,
        'bias_y': bias_y,
        'mean_error_aligned': float(err_a.mean()),
        'rmse_pos_aligned': float(np.sqrt(np.mean(err_a**2))),
    }


def main():
    parser = argparse.ArgumentParser(description='Joint multi-anchor EKF baseline for seq6')
    parser.add_argument('--data_npz', default='/home/evawang/T-GCN/data/lbl_aqualoc_seq6_multi_anchor_train4000_test1000.npz')
    parser.add_argument('--sound_speed', type=float, default=1452.57)
    parser.add_argument('--q_pos', type=float, default=0.01)
    parser.add_argument('--q_z', type=float, default=0.05)
    parser.add_argument('--q_v', type=float, default=0.2)
    parser.add_argument('--q_theta', type=float, default=0.01)
    parser.add_argument('--r_range', type=float, default=0.5)
    parser.add_argument('--save_npz', default='/home/evawang/T-GCN/data/ekf_seq6_joint_result.npz')
    args = parser.parse_args()

    d = np.load(args.data_npz, allow_pickle=True)
    X_train_n, Y_train = d['X_train'], d['Y_train']
    X_test_n, Y_test = d['X_test'], d['Y_test']
    Z_test = d['Z_test']
    D_test = d['D_test']
    M_test = d['M_test']
    t_test = d['t_test']
    feat_mean, feat_std = d['feat_mean'], d['feat_std']
    buoy_xyz = d['buoy_xyz'].astype(np.float64)

    X_train = denorm_features(X_train_n, feat_mean, feat_std)
    X_test = denorm_features(X_test_n, feat_mean, feat_std)

    # align fix->gt in XY and rotate heading
    scale, R, t = fit_similarity_2d(X_train[:, :2], Y_train)
    X_test[:, :2] = apply_similarity_2d(X_test[:, :2], scale, R, t)
    rot = float(np.arctan2(R[1,0], R[0,0]))
    X_test[:, 4] = np.vectorize(wrap_angle)(X_test[:,4] + rot)

    # align buoy xy to gt frame
    buoy_xy_aligned = apply_similarity_2d(buoy_xyz[:, :2], scale, R, t)
    buoy_xyz_aligned = buoy_xyz.copy()
    buoy_xyz_aligned[:, :2] = buoy_xy_aligned

    res = run_ekf_joint(
        X=X_test, Z=Z_test, Y=Y_test,
        D=D_test, M=M_test, t=t_test,
        buoy_xyz=buoy_xyz_aligned,
        sound_speed=args.sound_speed,
        q_pos=args.q_pos, q_z=args.q_z, q_v=args.q_v, q_theta=args.q_theta, r_range=args.r_range,
    )

    out = Path(args.save_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        pred_xy=res['pred_xy'].astype(np.float32),
        gt_xy=res['gt_xy'].astype(np.float32),
        err=res['err'].astype(np.float32),
        mean_error=np.float32(res['mean_error']),
        rmse_pos=np.float32(res['rmse_pos']),
        bias_x=np.float32(res['bias_x']),
        bias_y=np.float32(res['bias_y']),
        mean_error_aligned=np.float32(res['mean_error_aligned']),
        rmse_pos_aligned=np.float32(res['rmse_pos_aligned']),
        align_scale=np.float32(scale),
        align_rot=np.asarray(R, dtype=np.float32),
        align_trans=np.asarray(t, dtype=np.float32),
        align_rot_angle=np.float32(rot),
        buoy_xyz_aligned=buoy_xyz_aligned.astype(np.float32),
    )

    print('Joint EKF evaluation on test split:')
    print(f"Mean error: {res['mean_error']:.6f} m")
    print(f"RMSE_pos: {res['rmse_pos']:.6f} m")
    print(f"Mean bias_x: {res['bias_x']:.6f} m")
    print(f"Mean bias_y: {res['bias_y']:.6f} m")
    print(f"Aligned mean error: {res['mean_error_aligned']:.6f} m")
    print(f"Aligned RMSE_pos: {res['rmse_pos_aligned']:.6f} m")
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
