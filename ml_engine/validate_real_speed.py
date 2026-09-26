"""
validate_real_speed.py

Evaluates the trained TrueTrack 1D-CNN (truetrack_model.onnx) directly against
real ground-truth GPS speed from the Chennai field recordings.

Tests zero-shot sim-to-real transfer:
1. Aligns phone mount orientation to the vehicle frame via static gravity vector R_mount.
2. Interpolates asynchronous 6-axis IMU streams onto a uniform 50 Hz grid.
3. Normalizes using global physics norm_stats.json.
4. Compares predicted speed v_hat against real GPS speed in clear-sky conditions.
"""

import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort

def evaluate_drive_log(folder_path, onnx_model_path, norm_stats_path):
    with open(norm_stats_path, 'r') as f:
        stats = json.load(f)
    means = np.array(stats['means'], dtype=np.float32)
    stds = np.array(stats['stds'], dtype=np.float32)

    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name

    df_tot = pd.read_csv(os.path.join(folder_path, 'TotalAcceleration.csv'))
    df_gyro = pd.read_csv(os.path.join(folder_path, 'Gyroscope.csv'))
    df_grav = pd.read_csv(os.path.join(folder_path, 'Gravity.csv'))
    df_loc = pd.read_csv(os.path.join(folder_path, 'Location.csv'))

    # Static mount alignment: rotate phone gravity vector to Earth vertical [+Z]
    grav_mean = df_grav[['x', 'y', 'z']].values.mean(axis=0)
    g_unit = grav_mean / np.linalg.norm(grav_mean)
    target = np.array([0.0, 0.0, 1.0])
    v = np.cross(g_unit, target)
    s = np.linalg.norm(v)
    c = np.dot(g_unit, target)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R_mount = np.eye(3) + vx + vx.dot(vx) * ((1.0 - c) / (s**2))

    t_acc = df_tot['seconds_elapsed'].values
    acc_aligned = df_tot[['x', 'y', 'z']].values.dot(R_mount.T)
    gyro_aligned = df_gyro[['x', 'y', 'z']].values.dot(R_mount.T)

    t_start = max(t_acc[0], df_loc['seconds_elapsed'].iloc[0])
    t_end = min(t_acc[-1], df_loc['seconds_elapsed'].iloc[-1])
    t_50hz = np.arange(t_start, t_end, 0.02)

    ax_50 = np.interp(t_50hz, t_acc, acc_aligned[:, 0])
    ay_50 = np.interp(t_50hz, t_acc, acc_aligned[:, 1])
    az_50 = np.interp(t_50hz, t_acc, acc_aligned[:, 2])

    gx_50 = np.interp(t_50hz, df_gyro['seconds_elapsed'], gyro_aligned[:, 0])
    gy_50 = np.interp(t_50hz, df_gyro['seconds_elapsed'], gyro_aligned[:, 1])
    gz_50 = np.interp(t_50hz, df_gyro['seconds_elapsed'], gyro_aligned[:, 2])

    v_gps = np.interp(t_50hz, df_loc['seconds_elapsed'], df_loc['speed'])
    acc_gps = np.interp(t_50hz, df_loc['seconds_elapsed'], df_loc['horizontalAccuracy'])

    raw_features = np.column_stack([ax_50, ay_50, az_50, gx_50, gy_50, gz_50])

    # Filter clear-sky moving segments (accuracy < 8m, speed > 1.0 m/s)
    mask = (acc_gps < 8.0) & (v_gps > 1.0)
    indices = [i for i in np.where(mask)[0] if i >= 50]

    preds_v, actual_v = [], []
    for idx in indices[::10]: # 5 Hz evaluation step
        win = raw_features[idx-50:idx]
        norm_win = (win - means) / stds
        x = norm_win.T[np.newaxis, :, :].astype(np.float32)
        out = session.run(None, {input_name: x})[0][0]
        preds_v.append(max(0.0, out[0]))
        actual_v.append(v_gps[idx])

    preds_v = np.array(preds_v)
    actual_v = np.array(actual_v)

    mae = np.mean(np.abs(preds_v - actual_v))
    rmse = np.sqrt(np.mean((preds_v - actual_v)**2))
    corr = np.corrcoef(preds_v, actual_v)[0, 1]

    return {
        "folder": os.path.basename(folder_path),
        "windows_evaluated": len(preds_v),
        "actual_speed_mean_kmh": float(np.mean(actual_v) * 3.6),
        "pred_speed_mean_kmh": float(np.mean(preds_v) * 3.6),
        "mae_kmh": float(mae * 3.6),
        "rmse_kmh": float(rmse * 3.6),
        "pearson_correlation": float(corr)
    }

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'ml_engine', 'truetrack_model.onnx')
    norm_path = os.path.join(base_dir, 'ml_engine', 'norm_stats.json')

    logs = [
        'Varadarajapuram-2026-09-25_11-12-54',
        'Rohini_Theatre_Koyambedu-2026-09-25_13-05-40'
    ]

    print("=================================================================")
    print("=== REAL GROUND TRUTH SPEED VALIDATION (CHENNAI DRIVE LOGS) ===")
    print("=================================================================")
    for log in logs:
        p = os.path.join(base_dir, log)
        if os.path.exists(p):
            res = evaluate_drive_log(p, model_path, norm_path)
            print(f"\n[Recording: {res['folder']}]")
            print(f"  Windows Evaluated:     {res['windows_evaluated']:,}")
            print(f"  Actual GPS Speed Mean: {res['actual_speed_mean_kmh']:.2f} km/h")
            print(f"  Predicted Speed Mean:  {res['pred_speed_mean_kmh']:.2f} km/h")
            print(f"  Speed MAE:             {res['mae_kmh']:.2f} km/h")
            print(f"  Speed RMSE:            {res['rmse_kmh']:.2f} km/h")
            print(f"  Pearson Correlation:   {res['pearson_correlation']:+.3f}")
