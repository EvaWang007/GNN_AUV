import itertools
import json
from pathlib import Path
import numpy as np

from ekf_seq6_baseline import denorm_features, fit_similarity_2d, apply_similarity_2d, wrap_angle, run_ekf_on_seq6_3d


def main():
    data_npz = Path('/home/evawang/T-GCN/data/lbl_aqualoc_seq6_6tuple_train4000_test1000.npz')
    out_json = Path('/home/evawang/T-GCN/data/ekf_seq6_grid_results.json')

    d = np.load(data_npz, allow_pickle=True)
    X_train_n, Y_train, M_train = d['X_train'], d['Y_train'], d['M_train']
    X_test_n, Y_test, M_test = d['X_test'], d['Y_test'], d['M_test']
    t_test = d['t_test']
    feat_mean, feat_std = d['feat_mean'], d['feat_std']
    Z_test = d['Z_test'] if 'Z_test' in d.files else np.full((X_test_n.shape[0],), -2.0)

    X_train = denorm_features(X_train_n, feat_mean, feat_std)
    X_test = denorm_features(X_test_n, feat_mean, feat_std)
    X_train[M_train == 0, 5] = 0.0
    X_test[M_test == 0, 5] = 0.0

    # fix->gt alignment (same as baseline)
    scale, R_align, t_align = fit_similarity_2d(X_train[:, :2], Y_train)
    X_train[:, :2] = apply_similarity_2d(X_train[:, :2], scale, R_align, t_align)
    X_test[:, :2] = apply_similarity_2d(X_test[:, :2], scale, R_align, t_align)
    rot_angle = float(np.arctan2(R_align[1, 0], R_align[0, 0]))
    X_train[:, 4] = np.vectorize(wrap_angle)(X_train[:, 4] + rot_angle)
    X_test[:, 4] = np.vectorize(wrap_angle)(X_test[:, 4] + rot_angle)

    buoy_src = np.array([[-150.0, 75.0]])
    buoy_aligned = apply_similarity_2d(buoy_src, scale, R_align, t_align)[0]
    bx, by = float(buoy_aligned[0]), float(buoy_aligned[1])
    bz = -377.0

    # grids
    q_pos_grid = [0.005, 0.01, 0.05, 0.1]
    q_z_grid = [0.01, 0.05, 0.2, 1.0]
    q_v_grid = [0.01, 0.05, 0.1, 0.2]
    q_theta_grid = [0.001, 0.005, 0.01, 0.02]
    r_range_grid = [0.5, 1.0, 4.0, 9.0, 16.0, 25.0]

    combos = list(itertools.product(q_pos_grid, q_z_grid, q_v_grid, q_theta_grid, r_range_grid))
    results = []

    for i, (q_pos, q_z, q_v, q_theta, r_range) in enumerate(combos, start=1):
        res = run_ekf_on_seq6_3d(
            X=X_test,
            Z=Z_test,
            Y=Y_test,
            M=M_test,
            t=t_test,
            buoy_x=bx,
            buoy_y=by,
            buoy_z=bz,
            sound_speed=1452.57,
            q_pos=q_pos,
            q_z=q_z,
            q_v=q_v,
            q_theta=q_theta,
            r_range=r_range,
        )
        row = {
            'q_pos': q_pos,
            'q_z': q_z,
            'q_v': q_v,
            'q_theta': q_theta,
            'r_range': r_range,
            'mean_error': res['mean_error'],
            'rmse_pos': res['rmse_pos'],
            'bias_x': res['bias_x'],
            'bias_y': res['bias_y'],
            'aligned_rmse_pos': res['rmse_pos_aligned'],
        }
        results.append(row)
        if i % 50 == 0 or i == len(combos):
            print(f'[{i}/{len(combos)}] current rmse={row["rmse_pos"]:.3f}')

    results.sort(key=lambda r: r['rmse_pos'])
    best = results[0]

    out_json.write_text(json.dumps({'best': best, 'top20': results[:20], 'all_count': len(results)}, indent=2), encoding='utf-8')

    print('Best config:')
    print(best)
    print(f'Saved grid summary to {out_json}')


if __name__ == '__main__':
    main()
