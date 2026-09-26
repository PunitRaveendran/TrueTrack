"""
TrueTrack - Real-World Phone Sensor Data Ingest & 50 Hz Synchronizer
===================================================================
Ingests and aligns asynchronous multi-rate smartphone sensor streams recorded
from apps such as Sensor Logger or Phyphox.

Resamples a Sensor Logger folder containing Accelerometer.csv and Gyroscope.csv,
with optional Location.csv and Orientation.csv, onto a 50 Hz time grid. The
output preserves raw sensor axes; it does not estimate the phone-to-vehicle
rotation or produce benchmark-ready ground truth.
"""

import os
import argparse
import pandas as pd
import numpy as np


def _validate_time_column(values, label):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} needs at least two finite timestamps")
    if np.any(np.diff(values) < 0):
        raise ValueError(f"{label} timestamps must be sorted in ascending order")


def _interpolate_location(t_source, values, t_target, max_gap_s):
    """Interpolate only across short GNSS intervals; preserve long outages as NaN."""
    values = np.asarray(values, dtype=float)
    result = np.interp(t_target, t_source, values)
    left = np.clip(np.searchsorted(t_source, t_target, side='right') - 1, 0, len(t_source) - 1)
    right = np.clip(np.searchsorted(t_source, t_target, side='left'), 0, len(t_source) - 1)
    valid = (
        (t_target >= t_source[0]) &
        (t_target <= t_source[-1]) &
        ((t_source[right] - t_source[left]) <= max_gap_s)
    )
    result[~valid] = np.nan
    return result

def load_sensor_logger_folder(folder_path, blackout_start_s=None, blackout_duration_s=45.0, max_gps_gap_s=2.5):
    """
    Ingests a raw Sensor Logger directory, synchronizes all streams onto a uniform 50 Hz
    timebase (dt = 0.02s) and outputs resampled sensor/GNSS channels. This does not
    calibrate the phone-to-vehicle axes or create a benchmark-ready ground truth.
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
    _validate_time_column(t_acc, "Accelerometer")
    _validate_time_column(t_gyro, "Gyroscope")
    t_start = max(t_acc[0], t_gyro[0])
    t_end = min(t_acc[-1], t_gyro[-1])

    if df_ori is not None and len(df_ori) > 0:
        t_ori = df_ori['seconds_elapsed'].values
        _validate_time_column(t_ori, "Orientation")
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
        roll_values = np.unwrap(df_ori['roll'].values.astype(float))
        roll_rad_50hz = np.interp(t_uniform, df_ori['seconds_elapsed'].values, roll_values)
    else:
        roll_rad_50hz = np.zeros_like(t_uniform)

    # GPS (Location)
    if df_loc is not None and len(df_loc) > 0:
        t_loc = df_loc['seconds_elapsed'].values
        _validate_time_column(t_loc, "Location")
        lat_50hz = _interpolate_location(t_loc, df_loc['latitude'].values, t_uniform, max_gps_gap_s)
        lon_50hz = _interpolate_location(t_loc, df_loc['longitude'].values, t_uniform, max_gps_gap_s)
        speed_50hz = _interpolate_location(t_loc, df_loc['speed'].values, t_uniform, max_gps_gap_s) if 'speed' in df_loc.columns else np.full_like(t_uniform, np.nan)
        acc_50hz = _interpolate_location(t_loc, df_loc['accuracy'].values, t_uniform, max_gps_gap_s) if 'accuracy' in df_loc.columns else np.full_like(t_uniform, np.nan)
    else:
        lat_50hz = np.full_like(t_uniform, np.nan)
        lon_50hz = np.full_like(t_uniform, np.nan)
        speed_50hz = np.full_like(t_uniform, np.nan)
        acc_50hz = np.full_like(t_uniform, np.nan)

    # Optional de-roll in the app's declared X=lateral, Y=forward, Z=vertical frame.
    # Raw imported axes remain unchanged; the log still needs phone-to-vehicle calibration.
    cos_phi = np.cos(roll_rad_50hz)
    sin_phi = np.sin(roll_rad_50hz)
    ax_derolled = cos_phi * ax_50hz - sin_phi * az_50hz
    ay_derolled = ay_50hz
    az_derolled = sin_phi * ax_50hz + cos_phi * az_50hz

    # Optional user-specified blackout mask (if requested); not automatically detected.
    is_blackout = np.zeros(len(t_uniform), dtype=int)
    if blackout_start_s is not None:
        mask = (rel_time_s >= blackout_start_s) & (rel_time_s <= blackout_start_s + blackout_duration_s)
        is_blackout[mask] = 1
        lat_50hz[mask] = np.nan
        lon_50hz[mask] = np.nan
        speed_50hz[mask] = np.nan
        acc_50hz[mask] = np.nan

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
        'imu_ax_derolled': np.round(ax_derolled, 4),
        'imu_ay_derolled': np.round(ay_derolled, 4),
        'imu_az_derolled': np.round(az_derolled, 4),
        'gps_lat': lat_50hz,
        'gps_lon': lon_50hz,
        'gps_speed_ms': np.round(speed_50hz, 3),
        'gps_speed_kmh': np.round(speed_50hz * 3.6, 2),
        'gps_accuracy_m': np.round(acc_50hz, 2),
        'is_blackout': is_blackout
    })

    return df_out

def main():
    parser = argparse.ArgumentParser(description="Resample Sensor Logger accelerometer, gyroscope, orientation, and GNSS CSV streams.")
    parser.add_argument("--input", help="Sensor Logger folder containing Accelerometer.csv and Gyroscope.csv")
    parser.add_argument("--output", help="Destination CSV for the resampled channels")
    parser.add_argument("--blackout-start", type=float, help="Optional relative time in seconds to label as a simulated blackout")
    parser.add_argument("--blackout-duration", type=float, default=45.0, help="Duration for the optional simulated blackout label")
    args = parser.parse_args()

    if args.input:
        df_sync = load_sensor_logger_folder(
            args.input,
            blackout_start_s=args.blackout_start,
            blackout_duration_s=args.blackout_duration
        )
        if args.output:
            output_dir = os.path.dirname(os.path.abspath(args.output))
            os.makedirs(output_dir, exist_ok=True)
            df_sync.to_csv(args.output, index=False)
            print(f"Saved resampled sensor log to {args.output}")
        else:
            print("No CSV was saved; pass --output to write the resampled log.")
        print(f"Resampled frames: {len(df_sync):,} ({df_sync['timestamp'].iloc[-1]:.1f} s at 50 Hz)")
        print(f"Valid GNSS frames: {df_sync['gps_lat'].notna().sum():,} / {len(df_sync):,}")
        print("The output is not a benchmark result and has no phone-to-vehicle axis calibration.")
        return

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
            print(f"  Max reported GPS speed: {df_sync['gps_speed_kmh'].max():.1f} km/h")
            print(f"  Raw ax std: {df_sync['imu_ax'].std():.3f} m/s^2 -> De-rolled ax std: {df_sync['imu_ax_derolled'].std():.3f} m/s^2")


if __name__ == '__main__':
    main()
