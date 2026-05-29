import argparse
from pathlib import Path
import numpy as np


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def interp_linear(t_src, v_src, t_dst):
    return np.interp(t_dst, t_src, v_src)


def main():
    parser = argparse.ArgumentParser(description='Prepare 6-tuple dataset from LBL_Aqualoc_sequence_6 bag + colmap gt')
    parser.add_argument('--bag', default='/home/evawang/Downloads/LBL_Aqualoc_sequence_6.bag')
    parser.add_argument('--gt', default='/home/evawang/Downloads/archaeo_groundtruth_files/new_archaeo_colmap_traj_sequence_06.txt')
    parser.add_argument('--output_npz', default='/home/evawang/T-GCN/data/lbl_aqualoc_seq6_6tuple_train4000_test1000.npz')
    parser.add_argument('--train_samples', type=int, default=4000)
    parser.add_argument('--test_samples', type=int, default=1000)
    parser.add_argument('--n_points', type=int, default=5001)
    parser.add_argument('--delay_topic', default='/lbl_1')
    parser.add_argument('--z_source', choices=['fix', 'barometer', 'gt'], default='fix',
                        help='Source of z channel for EKF 3D: /fix.altitude, /barometer_node/depth, or colmap gt z.')
    parser.add_argument('--baro_depth_topic', default='/barometer_node/depth')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # lazy import so normal python env can still open file
    from rosbags.highlevel import AnyReader

    bag = Path(args.bag)
    gt_path = Path(args.gt)
    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Read rosbag topics
    # -------------------------
    fix_t, fix_x, fix_y, fix_z = [], [], [], []
    imu_t, imu_a = [], []
    baro_t, baro_depth = [], []
    lbl_t, lbl_delay = [], []

    with AnyReader([bag]) as reader:
        conns = {c.topic: c for c in reader.connections}
        required = ['/fix', '/rtimulib_node/imu', args.delay_topic]
        if args.z_source == 'barometer':
            required.append(args.baro_depth_topic)
        miss = [t for t in required if t not in conns]
        if miss:
            raise ValueError(f'Missing topics in bag: {miss}')

        c_fix = conns['/fix']
        c_imu = conns['/rtimulib_node/imu']
        c_lbl = conns[args.delay_topic]
        c_baro = conns[args.baro_depth_topic] if args.z_source == 'barometer' else None

        t0_ns = None
        # choose /fix start as global reference start (same as lbl in this bag)
        for c, t, raw in reader.messages(connections=[c_fix]):
            t0_ns = t
            break
        if t0_ns is None:
            raise ValueError('Empty /fix topic')

        for c, t, raw in reader.messages(connections=[c_fix]):
            msg = reader.deserialize(raw, c.msgtype)
            tt = (t - t0_ns) / 1e9
            fix_t.append(tt)
            # In this dataset /fix appears in NED-style frame values
            fix_x.append(float(msg.latitude))
            fix_y.append(float(msg.longitude))
            fix_z.append(float(msg.altitude))

        for c, t, raw in reader.messages(connections=[c_imu]):
            msg = reader.deserialize(raw, c.msgtype)
            tt = (t - t0_ns) / 1e9
            ax = float(msg.linear_acceleration.x)
            ay = float(msg.linear_acceleration.y)
            az = float(msg.linear_acceleration.z)
            imu_t.append(tt)
            imu_a.append(float(np.sqrt(ax * ax + ay * ay + az * az)))

        for c, t, raw in reader.messages(connections=[c_lbl]):
            msg = reader.deserialize(raw, c.msgtype)
            tt = (t - t0_ns) / 1e9
            lbl_t.append(tt)
            lbl_delay.append(float(msg.arrival_time))

        if c_baro is not None:
            for c, t, raw in reader.messages(connections=[c_baro]):
                msg = reader.deserialize(raw, c.msgtype)
                tt = (t - t0_ns) / 1e9
                baro_t.append(tt)
                # In this dataset this topic stores depth-like scalar in fluid_pressure.
                baro_depth.append(float(msg.fluid_pressure))

    fix_t = np.asarray(fix_t)
    fix_x = np.asarray(fix_x)
    fix_y = np.asarray(fix_y)
    fix_z = np.asarray(fix_z)
    imu_t = np.asarray(imu_t)
    imu_a = np.asarray(imu_a)
    baro_t = np.asarray(baro_t)
    baro_depth = np.asarray(baro_depth)
    lbl_t = np.asarray(lbl_t)
    lbl_delay = np.asarray(lbl_delay)

    # -------------------------
    # 2) Read colmap GT and align time scale
    # -------------------------
    gt = np.loadtxt(gt_path)
    # file format: t, x, y, z, qx, qy, qz, qw ; t appears in 20-unit ticks
    t_gt = gt[:, 0] / 20.0
    x_gt = gt[:, 1]
    y_gt = gt[:, 2]
    z_gt = gt[:, 3]

    # -------------------------
    # 3) Build common timeline with 5001 points
    # -------------------------
    t_start = max(fix_t.min(), imu_t.min(), lbl_t.min(), t_gt.min())
    t_end = min(fix_t.max(), imu_t.max(), lbl_t.max(), t_gt.max())
    if t_end <= t_start:
        raise ValueError('Invalid overlapping time range.')

    t = np.linspace(t_start, t_end, args.n_points)

    # -------------------------
    # 4) Build features x,y,v,a,theta,delay and z-channel for EKF 3D
    # -------------------------
    x_nav = interp_linear(fix_t, fix_x, t)
    y_nav = interp_linear(fix_t, fix_y, t)

    dt = np.gradient(t)
    vx = np.gradient(x_nav) / dt
    vy = np.gradient(y_nav) / dt
    v = np.sqrt(vx * vx + vy * vy)
    theta = np.arctan2(vy, vx)
    theta = wrap_to_pi(theta)

    a = interp_linear(imu_t, imu_a, t)

    if args.z_source == 'fix':
        z_series = interp_linear(fix_t, fix_z, t)
    elif args.z_source == 'barometer':
        if len(baro_t) == 0:
            raise ValueError('Requested barometer z_source but no barometer data found.')
        z_series = interp_linear(baro_t, baro_depth, t)
    else:  # gt
        z_series = interp_linear(t_gt, z_gt, t)

    # delay and mask: keep only moments where LBL actually reports (nearest grid bin), others 0
    delay = np.zeros_like(t, dtype=np.float64)
    mask = np.zeros_like(t, dtype=np.int64)
    idx = np.searchsorted(t, lbl_t)
    idx = np.clip(idx, 0, len(t) - 1)
    # choose closest neighbor
    left = np.maximum(idx - 1, 0)
    choose_left = np.abs(t[left] - lbl_t) < np.abs(t[idx] - lbl_t)
    idx[choose_left] = left[choose_left]
    # deduplicate: keep latest sample if collision
    for k, j in enumerate(idx):
        delay[j] = lbl_delay[k]
        mask[j] = 1

    # -------------------------
    # 5) Build labels from colmap GT
    # -------------------------
    x_gt_i = interp_linear(t_gt, x_gt, t)
    y_gt_i = interp_linear(t_gt, y_gt, t)

    X_full = np.stack([x_nav, y_nav, v, a, theta, delay], axis=1)
    Y_full = np.stack([x_gt_i, y_gt_i], axis=1)

    # one-step supervised
    X = X_full[:-1]
    Y = Y_full[1:]
    M = mask[:-1]
    Z = z_series[:-1]

    needed = args.train_samples + args.test_samples
    if X.shape[0] < needed:
        raise ValueError(f'Not enough one-step samples: {X.shape[0]} < {needed}')

    X_train = X[:args.train_samples]
    Y_train = Y[:args.train_samples]
    M_train = M[:args.train_samples]
    Z_train = Z[:args.train_samples]

    X_test = X[args.train_samples:args.train_samples + args.test_samples]
    Y_test = Y[args.train_samples:args.train_samples + args.test_samples]
    M_test = M[args.train_samples:args.train_samples + args.test_samples]
    Z_test = Z[args.train_samples:args.train_samples + args.test_samples]

    # -------------------------
    # 6) Normalize using train stats
    # -------------------------
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0

    # delay normalization only on valid samples
    valid_delay = X_train[M_train == 1, 5]
    if valid_delay.shape[0] > 1:
        feat_mean[5] = valid_delay.mean()
        feat_std[5] = max(valid_delay.std(), 1e-8)

    X_train_n = (X_train - feat_mean) / feat_std
    X_test_n = (X_test - feat_mean) / feat_std
    X_train_n[M_train == 0, 5] = 0.0
    X_test_n[M_test == 0, 5] = 0.0

    np.savez(
        out,
        X_train=X_train_n.astype(np.float32),
        Y_train=Y_train.astype(np.float32),
        M_train=M_train.astype(np.int64),
        X_test=X_test_n.astype(np.float32),
        Y_test=Y_test.astype(np.float32),
        M_test=M_test.astype(np.int64),
        Z_train=Z_train.astype(np.float32),
        Z_test=Z_test.astype(np.float32),
        feat_mean=feat_mean.astype(np.float32),
        feat_std=feat_std.astype(np.float32),
        t_train=t[:args.train_samples].astype(np.float32),
        t_test=t[args.train_samples:args.train_samples + args.test_samples].astype(np.float32),
        source_bag=str(bag),
        source_gt=str(gt_path),
        delay_topic=args.delay_topic,
        z_source=args.z_source,
        common_t_start=np.float32(t_start),
        common_t_end=np.float32(t_end),
    )

    print(f'Saved: {out}')
    print(f'Timeline: {t_start:.3f}s -> {t_end:.3f}s, n_points={len(t)}, dt~{(t[1]-t[0]):.6f}s')
    print(f'Train: X={X_train_n.shape}, Y={Y_train.shape}, M={M_train.shape}')
    print(f'Test : X={X_test_n.shape}, Y={Y_test.shape}, M={M_test.shape}')
    print(f'Delay valid ratio train: {M_train.mean():.4f}, test: {M_test.mean():.4f}')
    print(f'Z source: {args.z_source}, train z range [{Z_train.min():.3f}, {Z_train.max():.3f}]')


if __name__ == '__main__':
    main()
