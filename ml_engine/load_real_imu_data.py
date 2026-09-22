"""
TrueTrack - Real-World Phone Sensor Data Ingest Engine
======================================================
Loads and parses real vehicular IMU logs recorded from consumer smartphone apps
such as 'Sensor Logger' (iOS/Android) or 'Phyphox' (iOS/Android).

Supported Schema:
- Sensor Logger: Accelerometer.csv, Gyroscope.csv, Location.csv (or merged CSV)
- Phyphox: Raw Sensors CSV
- Generic: timestamp, ax, ay, az, gx, gy, gz, lat, lon, speed

Workflow:
1. Interpolates and synchronizes asynchronous sensor streams onto a uniform 50 Hz timebase.
2. Formats features identical to synthetic_telemetry_50hz.csv for zero-friction benchmarking.
"""

import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def parse_sensor_logger_csv(csv_path, blackout_start_s=None, blackout_duration_s=45.0):
    """
    Parses a real smartphone sensor log and prepares it for TrueTrack evaluation.
    If blackout_start_s is provided, injects a simulated GPS blackout over that window
    to evaluate dead-reckoning performance against real ground-truth GPS.
    """
    df = pd.read_csv(csv_path)
    
    # Standardize column headers (case-insensitive)
    cols = {c.lower().strip(): c for c in df.columns}
    
    def find_col(candidates):
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    col_t = find_col(['time', 'timestamp', 'seconds_elapsed', 'epoch'])
    col_ax = find_col(['ax', 'accel_x', 'accelerometer_x', 'acceleration x (m/s^2)'])
    col_ay = find_col(['ay', 'accel_y', 'accelerometer_y', 'acceleration y (m/s^2)'])
    col_az = find_col(['az', 'accel_z', 'accelerometer_z', 'acceleration z (m/s^2)'])
    col_gz = find_col(['gz', 'gyro_z', 'gyroscope_z', 'angular velocity z (rad/s)'])
    col_lat = find_col(['lat', 'latitude', 'gps_lat'])
    col_lon = find_col(['lon', 'longitude', 'gps_lon'])
    col_speed = find_col(['speed', 'velocity', 'gps_speed (m/s)'])

    if not (col_t and col_ax and col_ay and col_az and col_gz):
        raise ValueError(f"CSV missing mandatory IMU columns. Detected: {list(df.columns)}")

    # Time normalization (start at 0.0s)
    raw_time = df[col_t].values
    time_s = raw_time - raw_time[0]
    if time_s[-1] > 1e6: # epoch milliseconds
        time_s = time_s / 1000.0

    # Resample to uniform 50 Hz
    duration = time_s[-1]
    uniform_t = np.arange(0, duration, 0.02)
    
    interp_ax = interp1d(time_s, df[col_ax].values, fill_value="extrapolate")(uniform_t)
    interp_ay = interp1d(time_s, df[col_ay].values, fill_value="extrapolate")(uniform_t)
    interp_az = interp1d(time_s, df[col_az].values, fill_value="extrapolate")(uniform_t)
    interp_gz = interp1d(time_s, df[col_gz].values, fill_value="extrapolate")(uniform_t)

    # GPS coordinates (if available)
    if col_lat and col_lon:
        interp_lat = interp1d(time_s, df[col_lat].values, fill_value="extrapolate")(uniform_t)
        interp_lon = interp1d(time_s, df[col_lon].values, fill_value="extrapolate")(uniform_t)
    else:
        interp_lat = np.full_like(uniform_t, np.nan)
        interp_lon = np.full_like(uniform_t, np.nan)

    if col_speed:
        interp_speed = interp1d(time_s, df[col_speed].values, fill_value="extrapolate")(uniform_t)
    else:
        interp_speed = np.zeros_like(uniform_t)

    # Blackout flag
    is_blackout = np.zeros(len(uniform_t), dtype=int)
    if blackout_start_s is not None:
        blackout_mask = (uniform_t >= blackout_start_s) & (uniform_t <= blackout_start_s + blackout_duration_s)
        is_blackout[blackout_mask] = 1

    processed_df = pd.DataFrame({
        'timestamp': np.round(uniform_t, 3),
        'imu_ax': np.round(interp_ax, 4),
        'imu_ay': np.round(interp_ay, 4),
        'imu_az': np.round(interp_az, 4),
        'imu_gz': np.round(interp_gz, 5),
        'gps_lat': interp_lat,
        'gps_lon': interp_lon,
        'true_speed': interp_speed,
        'is_blackout': is_blackout
    })
    
    return processed_df

if __name__ == '__main__':
    print("TrueTrack Real-World IMU Ingestion Ready.")
    print("Record a 2-minute motorcycle/scooter ride using 'Sensor Logger' or 'Phyphox'.")
    print("Save the CSV to 'ml_engine/real_ride.csv' and run the ingest pipeline.")
