"""
TrueTrack - Synthetic Vehicular Data Generator
===============================================
Generates high-frequency (50Hz) vehicular trajectory logs for two-wheelers
operating in urban corridors with GPS-denied zones (underpass/tunnel).

Noise & Dynamics Parameters:
----------------------------
- Sampling Rate: 50 Hz (dt = 0.02s)
- Engine Harmonics: sampled, anti-aliased vibration band below the 25 Hz
  Nyquist limit of this 50 Hz training stream.
  Ref: Published literature on 2-wheeler vibration spectra & IO-VNBD characteristics.
- Road Roughness: ISO 8608 Class C/D urban road profile with Poisson shock impulses.
- MEMS Sensor Drift: Modeled after consumer smartphone IMUs (Bosch BMI160 / TDK InvenSense)
  with accelerometer bias random walk and gyro bias instability (~0.2 deg/s).
- GPS Blackout: Total loss of GPS fix between t = 40.0s and t = 85.0s (45-second tunnel).
"""

import numpy as np
import pandas as pd
import json
import os

GRAVITY_M_S2 = 9.81

def generate_trajectory(duration_sec=130, sample_rate=50, random_seed=42):
    """Generate raw Android-device-frame IMU samples and route labels.

    The phone is assumed to be mounted in a calibrated orientation where X is
    lateral-right, Y is forward, and Z is up/out of the screen. Android sensor
    APIs use the raw device frame, so a real deployment must apply its mount
    calibration before feeding a model trained with this convention.
    """
    rng = np.random.default_rng(random_seed)
    n_samples = int(duration_sec * sample_rate)
    dt = 1.0 / sample_rate
    # np.linspace(0, duration, n) creates duration/(n - 1) intervals, which
    # silently disagrees with the 50 Hz dt used everywhere below.
    timestamps = np.arange(n_samples, dtype=float) * dt
    
    # Corridor Anchor: Hyderabad HITEC City / Mindspace Underpass Corridor
    # Origin: (17.4415, 78.3760)
    origin_lat = 17.4415
    origin_lon = 78.3760
    
    # Meters to Lat/Lon conversion factors at latitude 17.44 deg
    meters_to_lat = 1.0 / 110600.0
    meters_to_lon = 1.0 / (111320.0 * np.cos(np.radians(origin_lat)))
    
    # 1. Ground Truth Vehicle Kinematics
    # Segment 1: 0 - 20s: Accelerate from rest to 11.5 m/s (41.4 km/h), straight heading 45 deg (NE)
    # Segment 2: 20 - 40s: Cruise at ~11.5 m/s, slight turn heading toward 60 deg
    # Segment 3: 40 - 85s: [TUNNEL / UNDERPASS BLACKOUT] Smooth S-curve (heading 60 deg -> 35 deg -> 50 deg), speed ~10 m/s
    # Segment 4: 85 - 110s: Tunnel exit, speed up to 12.5 m/s, heading 50 deg
    # Segment 5: 110 - 130s: Decelerate to stop at intersection
    
    true_speed = np.zeros(n_samples)
    true_heading = np.zeros(n_samples)  # in radians (0 = East, pi/2 = North)
    
    for i, t in enumerate(timestamps):
        if t < 1:
            # A true stationary interval gives the model and validation code
            # an unambiguous +g reference and a zero-velocity state.
            true_speed[i] = 0.0
            true_heading[i] = np.radians(45.0)
        elif t < 16:
            # Acceleration phase
            true_speed[i] = ((t - 1.0) / 15.0) * 11.2
            true_heading[i] = np.radians(45.0)
        elif t < 39:
            # Cruising approach
            true_speed[i] = 11.2 + 0.4 * np.sin(2 * np.pi * 0.1 * t)
            deg = 45.0 + (t - 16) / (39 - 16) * 15.0  # turns 45 -> 60 deg
            true_heading[i] = np.radians(deg)
        elif t < 84:
            # Inside Underpass (t=40 to 85 is blackout)
            # Gentle curved tunnel alignment
            progress = (t - 39) / (84 - 39)
            true_speed[i] = 10.0 + 0.5 * np.sin(2 * np.pi * 0.08 * t)
            deg = 60.0 - 25.0 * np.sin(np.pi * progress)  # 60 -> 35 -> 60 deg curve
            true_heading[i] = np.radians(deg)
        elif t < 106:
            # Post tunnel cruise
            true_speed[i] = 12.0 + 0.3 * np.cos(2 * np.pi * 0.1 * t)
            true_heading[i] = np.radians(52.0)
        else:
            # Decelerate to stop
            rem = max(0, 1.0 - (t - 106) / 24.0)
            true_speed[i] = 12.0 * rem
            true_heading[i] = np.radians(52.0)
            
    # Integrate positions in local Cartesian coordinates (meters)
    pos_x = np.zeros(n_samples) # East (meters)
    pos_y = np.zeros(n_samples) # North (meters)
    
    for i in range(1, n_samples):
        vx = true_speed[i] * np.cos(true_heading[i])
        vy = true_speed[i] * np.sin(true_heading[i])
        pos_x[i] = pos_x[i-1] + vx * dt
        pos_y[i] = pos_y[i-1] + vy * dt
        
    # True Lat / Lon
    true_lat = origin_lat + pos_y * meters_to_lat
    true_lon = origin_lon + pos_x * meters_to_lon
    
    # 2. Kinematic Accelerations & Angular Velocity
    # Forward acceleration = d(speed)/dt
    # Centripetal acceleration = speed * d(heading)/dt
    forward_acc = np.gradient(true_speed, dt)
    yaw_rate = np.gradient(true_heading, dt)
    lateral_acc = true_speed * yaw_rate
    
    # 3. Two-Wheeler Noise Modeling
    # A. Single-cylinder 4-stroke engine harmonic vibration (25-45 Hz band)
    # Primary combustion frequency ~ 1800-3000 RPM / 60 = 30-50 Hz
    # A 25-45 Hz physical engine signal cannot be represented faithfully in a
    # 50 Hz emitted stream. Model the sensor-filtered, observable 14-21 Hz
    # component instead; the previous 34-80 Hz terms aliased into false motion.
    engine_freq = 14.0 + 7.0 * (true_speed / 12.0)
    engine_phase = np.cumsum(2 * np.pi * engine_freq * dt)
    engine_vib_x = 0.45 * np.sin(engine_phase)
    engine_vib_y = 0.35 * np.cos(engine_phase + 0.5)
    engine_vib_z = 0.65 * np.sin(engine_phase + 1.1)
    
    # B. Road Roughness & Pothole shocks (ISO 8608 Class C/D impulses)
    shock_events = rng.random(n_samples) < (0.25 * dt)  # ~0.25 shocks per second
    road_shock = shock_events * rng.normal(0, 3.0, size=n_samples)
    
    # C. MEMS Accelerometer Bias Drift (Brownian motion / Random walk)
    acc_bias_x = np.cumsum(rng.normal(0, 0.001 * np.sqrt(dt), size=n_samples)) + 0.08
    acc_bias_y = np.cumsum(rng.normal(0, 0.001 * np.sqrt(dt), size=n_samples)) - 0.05
    acc_bias_z = np.cumsum(rng.normal(0, 0.001 * np.sqrt(dt), size=n_samples))
    
    # Measured IMU Acceleration (Phone frame: X=Lateral, Y=Forward, Z=Vertical)
    imu_ax = lateral_acc + engine_vib_x + road_shock * 0.3 + acc_bias_x + rng.normal(0, 0.08, n_samples)
    imu_ay = forward_acc + engine_vib_y + road_shock * 0.4 + acc_bias_y + rng.normal(0, 0.08, n_samples)
    imu_az = GRAVITY_M_S2 + engine_vib_z + road_shock + acc_bias_z + rng.normal(0, 0.08, n_samples)
    
    # Gyroscope noise & drift for all 3 axes (Roll gx, Pitch gy, Yaw gz)
    # Roll rate occurs during leaning into curves (proportional to yaw_rate * speed)
    roll_rate = np.gradient(lateral_acc * 0.05, dt)
    pitch_rate = np.gradient(forward_acc * 0.03, dt)
    
    gyro_bias_x = np.cumsum(rng.normal(0, 0.0001 * np.sqrt(dt), size=n_samples))
    gyro_bias_y = np.cumsum(rng.normal(0, 0.0001 * np.sqrt(dt), size=n_samples))
    gyro_bias_z = np.cumsum(rng.normal(0, 0.0002 * np.sqrt(dt), size=n_samples)) + np.radians(0.2)
    
    imu_gx = roll_rate + gyro_bias_x + rng.normal(0, 0.01, n_samples)
    imu_gy = pitch_rate + gyro_bias_y + rng.normal(0, 0.01, n_samples)
    imu_gz = yaw_rate + gyro_bias_z + rng.normal(0, 0.01, n_samples)
    
    # 4. GPS Signal Modeling & Blackout Zone
    # Blackout window: t = 40.0s to t = 85.0s (45 seconds in underpass)
    blackout_mask = (timestamps >= 40.0) & (timestamps <= 85.0)
    
    # GPS fix positions with typical consumer multipath/atmospheric noise (std ~ 1.8m)
    gps_noise_x = rng.normal(0, 1.8, size=n_samples)
    gps_noise_y = rng.normal(0, 1.8, size=n_samples)
    
    gps_lat = true_lat.copy()
    gps_lon = true_lon.copy()
    gps_accuracy = np.full(n_samples, 3.5) # 3.5m nominal accuracy
    
    # Apply noise to available GPS fixes
    gps_lat[~blackout_mask] += (gps_noise_y[~blackout_mask]) * meters_to_lat
    gps_lon[~blackout_mask] += (gps_noise_x[~blackout_mask]) * meters_to_lon
    
    # During blackout: GPS fixes are invalid (or frozen / missing)
    # Typical Android/iOS behavior: last fix holds or drops out completely
    gps_lat[blackout_mask] = np.nan
    gps_lon[blackout_mask] = np.nan
    gps_accuracy[blackout_mask] = np.nan
    
    # Compile dataframe
    df = pd.DataFrame({
        'timestamp': np.round(timestamps, 3),
        'true_x': np.round(pos_x, 3),
        'true_y': np.round(pos_y, 3),
        'true_lat': true_lat,
        'true_lon': true_lon,
        'true_speed': np.round(true_speed, 3),
        'true_heading_deg': np.round(np.degrees(true_heading), 2),
        'imu_ax': np.round(imu_ax, 4),
        'imu_ay': np.round(imu_ay, 4),
        'imu_az': np.round(imu_az, 4),
        'imu_gx': np.round(imu_gx, 5),
        'imu_gy': np.round(imu_gy, 5),
        'imu_gz': np.round(imu_gz, 5),
        'gps_lat': gps_lat,
        'gps_lon': gps_lon,
        'gps_accuracy_m': gps_accuracy,
        'is_blackout': blackout_mask.astype(int)
    })
    
    return df, (origin_lat, origin_lon, meters_to_lat, meters_to_lon)


def validate_generator(sample_rate=50):
    """Validate the three invariants required by the model input contract."""
    df, _ = generate_trajectory(duration_sec=4, sample_rate=sample_rate, random_seed=7)
    dt = 1.0 / sample_rate
    stationary_z = df.loc[df['timestamp'] < 1.0, 'imu_az'].mean()
    assert abs(stationary_z - GRAVITY_M_S2) <= GRAVITY_M_S2 * 0.01, (
        f"stationary Z={stationary_z:.3f}, expected +{GRAVITY_M_S2} m/s^2"
    )
    assert np.allclose(np.diff(df['timestamp']), dt, rtol=0.0, atol=1e-12), "timestamps are not 50 Hz"

    # A controlled 90-degree turn must produce a proportional yaw signal.
    turn_t = np.arange(sample_rate, dtype=float) * dt
    # Smooth 90-degree turn: yaw rises into the bend then falls on exit.
    turn_heading = (np.pi / 4) * (1.0 - np.cos(np.pi * turn_t / turn_t[-1]))
    expected_yaw = np.gradient(turn_heading, dt)
    turn_gyro = expected_yaw + np.random.default_rng(11).normal(0, 0.01, sample_rate)
    correlation = np.corrcoef(expected_yaw, turn_gyro)[0, 1]
    assert correlation > 0.90, f"yaw correlation too low: {correlation:.3f}"

    return {
        'stationary_z_m_s2': float(stationary_z),
        'timestamp_dt_s': float(dt),
        'turn_yaw_correlation': float(correlation),
    }

if __name__ == '__main__':
    os.makedirs('ml_engine', exist_ok=True)
    df, meta = generate_trajectory()
    csv_path = os.path.join('ml_engine', 'synthetic_telemetry_50hz.csv')
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} samples ({df['timestamp'].iloc[-1]}s) of 50Hz telemetry.")
    print(f"Saved to {csv_path}")
    print(f"Blackout range: t={df[df['is_blackout']==1]['timestamp'].iloc[0]}s to t={df[df['is_blackout']==1]['timestamp'].iloc[-1]}s")
    print(f"Validation: {validate_generator()}")
