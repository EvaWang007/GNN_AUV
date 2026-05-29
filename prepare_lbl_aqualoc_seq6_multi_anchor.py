import argparse
from pathlib import Path
import numpy as np


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def interp_linear(t_src, v_src, t_dst):
    return np.interp(t_dst, t_src, v_src)


def main():
    parser = argparse.ArgumentParser(description='Prepare seq6 multi-anchor dataset for joint EKF.')
    parser.add_argument('--bag', default='/home/evawang/Downloads/LBL_Aqualoc_sequence_6.bag')
    parser.add_argument('--gt', default='/home/evawang/Downloads/archaeo_groundtruth_files/new_archaeo_colmap_traj_sequence_06.txt')
    parser.add_argument('--output_npz', default='/home/evawang/T-GCN/data/lbl_aqualoc_seq6_multi_anchor_train4000_test1000.npz')
    parser.add_argument('--train_samples', type=int, default=4000)
    parser.add_argument('--test_samples', type=int, default=1000)
    parser.add_argument('--n_points', type=int, default=5001)
    parser.add_argument('--z_source', choices=['fix', 'gt'], default='fix')
    args = parser.parse_args()

    from rosbags.highlevel import AnyReader

    bag = Path(args.bag)
    gt_path = Path(args.gt)
    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)

    fix_t, fix_x, fix_y, fix_z = [], [], [], []
    imu_t, imu_a = [], []
    lbl_topics = ['/lbl_1', '/lbl_2', '/lbl_3', '/lbl_4']
    lbl_t = {k: [] for k in lbl_topics}
    lbl_d = {k: [] for k in lbl_topics}
    buoy_xyz = {}

    with AnyReader([bag]) as reader:
        conns = {c.topic: c for c in reader.connections}
        required = ['/fix', '/rtimulib_node/imu'] + lbl_topics
        miss = [t for t in required if t not in conns]
        if miss:
            raise ValueError(f'Missing topics: {miss}')

        c_fix = conns['/fix']
        c_imu = conns['/rtimulib_node/imu']

        t0_ns = None
        for _, t, _ in reader.messages(connections=[c_fix]):
            t0_ns = t
            break

        for c, t, raw in reader.messages(connections=[c_fix]):
            m = reader.deserialize(raw, c.msgtype)
            tt = (t - t0_ns) / 1e9
            fix_t.append(tt)
            fix_x.append(float(m.latitude))
            fix_y.append(float(m.longitude))
            fix_z.append(float(m.altitude))

        for c, t, raw in reader.messages(connections=[c_imu]):
            m = reader.deserialize(raw, c.msgtype)
            tt = (t - t0_ns) / 1e9
            ax, ay, az = float(m.linear_acceleration.x), float(m.linear_acceleration.y), float(m.linear_acceleration.z)
            imu_t.append(tt)
            imu_a.append(float(np.sqrt(ax * ax + ay * ay + az * az)))

        for topic in lbl_topics:
            c_lbl = conns[topic]
            for c, t, raw in reader.messages(connections=[c_lbl]):
                m = reader.deserialize(raw, c.msgtype)
                tt = (t - t0_ns) / 1e9
                lbl_t[topic].append(tt)
                lbl_d[topic].append(float(m.arrival_time))
                buoy_xyz[topic] = (float(m.bouy_X), float(m.bouy_Y), float(m.bouy_Z))

    fix_t = np.asarray(fix_t)
    fix_x = np.asarray(fix_x)
    fix_y = np.asarray(fix_y)
    fix_z = np.asarray(fix_z)
    imu_t = np.asarray(imu_t)
    imu_a = np.asarray(imu_a)
    for k in lbl_topics:
        lbl_t[k] = np.asarray(lbl_t[k])
        lbl_d[k] = np.asarray(lbl_d[k])

    gt = np.loadtxt(gt_path)
    t_gt = gt[:, 0] / 20.0
    x_gt, y_gt, z_gt = gt[:, 1], gt[:, 2], gt[:, 3]

    t_start = max([fix_t.min(), imu_t.min(), t_gt.min()] + [lbl_t[k].min() for k in lbl_topics])
    t_end = min([fix_t.max(), imu_t.max(), t_gt.max()] + [lbl_t[k].max() for k in lbl_topics])
    t = np.linspace(t_start, t_end, args.n_points)

    x_nav = interp_linear(fix_t, fix_x, t)
    y_nav = interp_linear(fix_t, fix_y, t)
    dt = np.gradient(t)
    vx = np.gradient(x_nav) / dt
    vy = np.gradient(y_nav) / dt
    v = np.sqrt(vx * vx + vy * vy)
    theta = wrap_to_pi(np.arctan2(vy, vx))
    a = interp_linear(imu_t, imu_a, t)

    if args.z_source == 'fix':
        z_series = interp_linear(fix_t, fix_z, t)
    else:
        z_series = interp_linear(t_gt, z_gt, t)

    x_gt_i = interp_linear(t_gt, x_gt, t)
    y_gt_i = interp_linear(t_gt, y_gt, t)

    # Build delay matrix [T,4] and mask [T,4]
    D = np.zeros((len(t), 4), dtype=np.float64)
    M = np.zeros((len(t), 4), dtype=np.int64)
    for j, topic in enumerate(lbl_topics):
        ts = lbl_t[topic]
        ds = lbl_d[topic]
        idx = np.searchsorted(t, ts)
        idx = np.clip(idx, 0, len(t) - 1)
        left = np.maximum(idx - 1, 0)
        choose_left = np.abs(t[left] - ts) < np.abs(t[idx] - ts)
        idx[choose_left] = left[choose_left]
        for k, ii in enumerate(idx):
            D[ii, j] = ds[k]
            M[ii, j] = 1

    X_full = np.stack([x_nav, y_nav, v, a, theta], axis=1)
    Y_full = np.stack([x_gt_i, y_gt_i], axis=1)

    X = X_full[:-1]
    Y = Y_full[1:]
    Z = z_series[:-1]
    D = D[:-1]
    M = M[:-1]

    needed = args.train_samples + args.test_samples
    if X.shape[0] < needed:
        raise ValueError(f'Not enough samples {X.shape[0]} < {needed}')

    X_train, X_test = X[:args.train_samples], X[args.train_samples:args.train_samples + args.test_samples]
    Y_train, Y_test = Y[:args.train_samples], Y[args.train_samples:args.train_samples + args.test_samples]
    Z_train, Z_test = Z[:args.train_samples], Z[args.train_samples:args.train_samples + args.test_samples]
    D_train, D_test = D[:args.train_samples], D[args.train_samples:args.train_samples + args.test_samples]
    M_train, M_test = M[:args.train_samples], M[args.train_samples:args.train_samples + args.test_samples]

    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_train_n = (X_train - feat_mean) / feat_std
    X_test_n = (X_test - feat_mean) / feat_std

    np.savez(
        out,
        X_train=X_train_n.astype(np.float32),
        Y_train=Y_train.astype(np.float32),
        Z_train=Z_train.astype(np.float32),
        D_train=D_train.astype(np.float32),
        M_train=M_train.astype(np.int64),
        X_test=X_test_n.astype(np.float32),
        Y_test=Y_test.astype(np.float32),
        Z_test=Z_test.astype(np.float32),
        D_test=D_test.astype(np.float32),
        M_test=M_test.astype(np.int64),
        feat_mean=feat_mean.astype(np.float32),
        feat_std=feat_std.astype(np.float32),
        t_train=t[:args.train_samples].astype(np.float32),
        t_test=t[args.train_samples:args.train_samples + args.test_samples].astype(np.float32),
        buoy_xyz=np.array([buoy_xyz[k] for k in lbl_topics], dtype=np.float32),
        lbl_topics=np.array(lbl_topics),
        source_bag=str(bag),
        source_gt=str(gt_path),
        z_source=args.z_source,
    )

    print(f'Saved: {out}')
    print(f'Train: X={X_train_n.shape}, Y={Y_train.shape}, Z={Z_train.shape}, D={D_train.shape}, M={M_train.shape}')
    print(f'Test : X={X_test_n.shape}, Y={Y_test.shape}, Z={Z_test.shape}, D={D_test.shape}, M={M_test.shape}')
    print('Delay valid ratio per anchor (train):', M_train.mean(axis=0))


if __name__ == '__main__':
    main()
