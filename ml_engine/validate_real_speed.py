"""
validate_real_speed.py

Evaluates the trained TrueTrack 1D-CNN (truetrack_model.onnx) directly against
real ground-truth GPS speed from the Chennai field recordings.

Tests zero-shot sim-to-real transfer:
1. Aligns phone mount orientation to vehicle frame using clean upright riding
   segments (|yaw_rate| < 0.05 rad/s) so dynamic cornering lean does not
   contaminate static mount orientation calibration.
2. Interpolates asynchronous 6-axis IMU streams onto a uniform 50 Hz grid.
3. Normalizes using global physics norm_stats.json.
4. Compares predicted speed v_hat against real GPS speed in clear-sky conditions.
5. Reports signed bias, MAE, RMSE, Pearson correlation r, and metrics for both
   overlapping and independent (non-overlapping) test windows.
"""

import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort

def evaluate_drive_log(folder_path, onnx_model_path, norm_stats_path, stride=10):
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

    # Static mount alignment: calibrate ONLY on upright straight riding (|yaw_rate| < 0.05 rad/s)
    # to avoid contaminating mount calibration with dynamic cornering lean angle.
    t_gyro = df_gyro['seconds_elapsed'].values
    gz_vals = df_gyro['z'].values
    gz_interp = np.interp(df_grav['seconds_elapsed'].values, t_gyro, gz_vals)
    mask_upright = (np.abs(gz_interp) < 0.05) & (df_grav['seconds_elapsed'] >= 5.0)

    if mask_upright.sum() > 50:
        grav_mean = df_grav.loc[mask_upright, ['x', 'y', 'z']].values.mean(axis=0)
    else:
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
    for idx in indices[::stride]:
        win = raw_features[idx-50:idx]
        norm_win = (win - means) / stds
        x = norm_win.T[np.newaxis, :, :].astype(np.float32)
        out = session.run(None, {input_name: x})[0][0]
        preds_v.append(max(0.0, float(out[0])))
        actual_v.append(float(v_gps[idx]))

    preds_v = np.array(preds_v)
    actual_v = np.array(actual_v)

    mae = float(np.mean(np.abs(preds_v - actual_v)))
    rmse = float(np.sqrt(np.mean((preds_v - actual_v)**2)))
    signed_bias = float(np.mean(preds_v - actual_v))
    corr = float(np.corrcoef(preds_v, actual_v)[0, 1])

    return {
        "folder": os.path.basename(folder_path),
        "stride": stride,
        "step_sec": stride * 0.02,
        "windows_evaluated": len(preds_v),
        "actual_speed_mean_kmh": float(np.mean(actual_v) * 3.6),
        "pred_speed_mean_kmh": float(np.mean(preds_v) * 3.6),
        "signed_bias_kmh": float(signed_bias * 3.6),
        "mae_kmh": float(mae * 3.6),
        "rmse_kmh": float(rmse * 3.6),
        "pearson_correlation": float(corr),
        "calibration_gravity_mean": [round(float(g), 3) for g in grav_mean]
    }

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'ml_engine', 'truetrack_model.onnx')
    norm_path = os.path.join(base_dir, 'ml_engine', 'norm_stats.json')

    logs = [
        'Rohini_Theatre_Koyambedu-2026-09-25_13-05-40',
        'Varadarajapuram-2026-09-25_11-12-54'
    ]

    print("==========================================================================================")
    print("=== REAL GROUND TRUTH SPEED VALIDATION (CHENNAI COMMUTER DRIVE LOGS) ===")
    print("=== Clean Upright Gravity Alignment • Signed Bias • Non-Overlapping Windows ===")
    print("==========================================================================================")

    for log in logs:
        p = os.path.join(base_dir, log)
        if os.path.exists(p):
            # 1. 5 Hz Evaluation (stride=10, 0.2s step)
            res_5hz = evaluate_drive_log(p, model_path, norm_path, stride=10)
            # 2. Independent Non-Overlapping Evaluation (stride=50, 1.0s step)
            res_indep = evaluate_drive_log(p, model_path, norm_path, stride=50)

            print(f"\n[Recording: {res_5hz['folder']}]")
            print(f"  Upright Calibrated Gravity: {res_5hz['calibration_gravity_mean']} m/s^2")
            print(f"  --- Overlapping Windows (5 Hz / 0.2s stride) ---")
            print(f"    Windows Evaluated:       {res_5hz['windows_evaluated']:,}")
            print(f"    Actual GPS Speed Mean:   {res_5hz['actual_speed_mean_kmh']:.2f} km/h")
            print(f"    Predicted Speed Mean:    {res_5hz['pred_speed_mean_kmh']:.2f} km/h")
            print(f"    Signed Bias (Pred-Act):  {res_5hz['signed_bias_kmh']:+.2f} km/h")
            print(f"    Speed MAE:               {res_5hz['mae_kmh']:.2f} km/h")
            print(f"    Speed RMSE:              {res_5hz['rmse_kmh']:.2f} km/h")
            print(f"    Pearson Correlation r:   {res_5hz['pearson_correlation']:+.3f}")
            print(f"  --- Independent Non-Overlapping Windows (1 Hz / 1.0s stride) ---")
            print(f"    Independent Windows:     {res_indep['windows_evaluated']:,}")
            print(f"    Speed MAE:               {res_indep['mae_kmh']:.2f} km/h")
            print(f"    Signed Bias:             {res_indep['signed_bias_kmh']:+.2f} km/h")
            print(f"    Pearson Correlation r:   {res_indep['pearson_correlation']:+.3f} (Zero window autocorrelation)")
