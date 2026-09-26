"""
TrueTrack - Real-World Phone Sensor Data Ingest & 50 Hz Synchronizer
===================================================================
Ingests and aligns asynchronous multi-rate smartphone sensor streams recorded
from apps such as Sensor Logger or Phyphox.

Solves the Real-World Sample-Rate & Jitter Problem:
- Consumer Android sensors stream asynchronously (e.g., ~60.8 Hz on 60 Hz display refresh)
  with microsecond timestamp jitter.
- The TrueTrack neural model expects a deterministic 50 Hz tensor: [1, 6, 50] (1.0 s window).
- This engine resamples all asynchronous streams onto a strictly uniform 50.0 Hz time grid (dt=0.02 s)
  via piece-wise linear interpolation (np.interp), guaranteeing zero time compression or dilation.

Supported Formats:
- Sensor Logger folder containing: Accelerometer.csv, Gyroscope.csv, Location.csv, Orientation.csv
- Merged single CSV containing timestamp, ax, ay, az, gx, gy, gz, roll, lat, lon, speed
"""

import os
import pandas as pd
import numpy as np

def load_sensor_logger_folder(folder_path, blackout_start_s=None, blackout_duration_s=45.0):
    """
    Ingests a raw Sensor Logger directory, synchronizes all streams onto a uniform 50 Hz
    timebase (dt = 0.02s), handles timestamp jitter, extracts roll lean angle, and outputs
    a standardized evaluation dataframe.
    """
    acc_path = os.path.join(folder_path, 'Accelerometer.csv')
    gyro_path = os.path.join(folder_path, 'Gyroscope.csv')
    loc_path = os.path.join(folder_path, 'Location.csv')
    ori_path = os.path.join(folder_path, 'Orientation.csv')

    if not (os.path.exists(acc_path) and os.path.exists(gyro_path)):
        raise FileNotFoundError(f"Missing Accelerometer.csv or Gyroscope.csv in {folder_path}")

    df_acc = pd.read_csv(acc_path)
    df_gyro = pd.read_csv(gyro_path)
    df_loc = pd.read_csv(loc_path) if os.path.exists(loc_path) else None
    df_ori = pd.read_csv(ori_path) if os.path.exists(ori_path) else None

    # Determine overlapping temporal envelope
    t_acc = df_acc['seconds_elapsed'].values
    t_gyro = df_gyro['seconds_elapsed'].values
    t_start = max(t_acc[0], t_gyro[0])
    t_end = min(t_acc[-1], t_gyro[-1])

    if df_ori is not None and len(df_ori) > 0:
        t_ori = df_ori['seconds_elapsed'].values
        t_start = max(t_start, t_ori[0])
        t_end = min(t_end, t_ori[-1])

    if t_end <= t_start:
        raise ValueError("Sensor streams have no overlapping temporal envelope.")

    # 50 Hz Uniform Time Grid (dt = 0.02 s)
    t_uniform = np.arange(t_start, t_end, 0.02)
    rel_time_s = t_uniform - t_start

    # Interpolate 6-axis IMU
    ax_50hz = np.interp(t_uniform, t_acc, df_acc['x'].values)
    ay_50hz = np.interp(t_uniform, t_acc, df_acc['y'].values)
    az_50hz = np.interp(t_uniform, t_acc, df_acc['z'].values)

    gx_50hz = np.interp(t_uniform, t_gyro, df_gyro['x'].values)
    gy_50hz = np.interp(t_uniform, t_gyro, df_gyro['y'].values)
    gz_50hz = np.interp(t_uniform, t_gyro, df_gyro['z'].values)

    # Orientation (Roll Lean Angle)
    if df_ori is not None and 'roll' in df_ori.columns:
        roll_rad_50hz = np.interp(t_uniform, df_ori['seconds_elapsed'].values, df_ori['roll'].values)
    else:
        roll_rad_50hz = np.zeros_like(t_uniform)

    # GPS (Location)
    if df_loc is not None and len(df_loc) > 0:
        t_loc = df_loc['seconds_elapsed'].values
        lat_50hz = np.interp(t_uniform, t_loc, df_loc['latitude'].values)
        lon_50hz = np.interp(t_uniform, t_loc, df_loc['longitude'].values)
        speed_50hz = np.interp(t_uniform, t_loc, df_loc['speed'].values) if 'speed' in df_loc.columns else np.zeros_like(t_uniform)
        acc_50hz = np.interp(t_uniform, t_loc, df_loc['accuracy'].values) if 'accuracy' in df_loc.columns else np.full_like(t_uniform, 5.0)
    else:
        lat_50hz = np.full_like(t_uniform, np.nan)
        lon_50hz = np.full_like(t_uniform, np.nan)
        speed_50hz = np.zeros_like(t_uniform)
        acc_50hz = np.full_like(t_uniform, np.nan)

    # De-roll accelerations via R_x(-phi)
    # [ax_c, ay_c, az_c]^T = R_x(-phi) * [ax, ay, az]^T
    cos_phi = np.cos(roll_rad_50hz)
    sin_phi = np.sin(roll_rad_50hz)
    ay_derolled = cos_phi * ay_50hz + sin_phi * az_50hz
    az_derolled = -sin_phi * ay_50hz + cos_phi * az_50hz

    # Simulated blackout flag (if requested)
    is_blackout = np.zeros(len(t_uniform), dtype=int)
    if blackout_start_s is not None:
        mask = (rel_time_s >= blackout_start_s) & (rel_time_s <= blackout_start_s + blackout_duration_s)
        is_blackout[mask] = 1

    df_out = pd.DataFrame({
        'timestamp': np.round(rel_time_s, 3),
        'imu_ax': np.round(ax_50hz, 4),
        'imu_ay': np.round(ay_50hz, 4),
        'imu_az': np.round(az_50hz, 4),
        'imu_gx': np.round(gx_50hz, 5),
        'imu_gy': np.round(gy_50hz, 5),
        'imu_gz': np.round(gz_50hz, 5),
        'roll_lean_rad': np.round(roll_rad_50hz, 4),
        'roll_lean_deg': np.round(np.degrees(roll_rad_50hz), 2),
        'imu_ay_derolled': np.round(ay_derolled, 4),
        'imu_az_derolled': np.round(az_derolled, 4),
        'gps_lat': lat_50hz,
        'gps_lon': lon_50hz,
        'true_speed_ms': np.round(speed_50hz, 3),
        'true_speed_kmh': np.round(speed_50hz * 3.6, 2),
        'gps_accuracy_m': np.round(acc_50hz, 2),
        'is_blackout': is_blackout
    })

    return df_out

if __name__ == '__main__':
    test_folders = [
        '45_46-2026-09-25_10-54-49',
        'Varadarajapuram-2026-09-25_11-12-54',
        'Rohini_Theatre_Koyambedu-2026-09-25_13-05-40'
    ]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Testing Real Drive Ingest & 50 Hz Resampling:")
    for tf in test_folders:
        p = os.path.join(base_dir, tf)
        if os.path.exists(p):
            df_sync = load_sensor_logger_folder(p)
            print(f"\n[Folder: {tf}]")
            print(f"  Synchronized 50 Hz frames: {len(df_sync):,} ({df_sync['timestamp'].iloc[-1]:.1f} s)")
            print(f"  Max Lean Angle: {df_sync['roll_lean_deg'].abs().max():.1f} deg")
            print(f"  Max Speed: {df_sync['true_speed_kmh'].max():.1f} km/h")
            print(f"  Raw ay std: {df_sync['imu_ay'].std():.3f} m/s^2 -> De-rolled ay std: {df_sync['imu_ay_derolled'].std():.3f} m/s^2")
