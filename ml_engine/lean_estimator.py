"""
TrueTrack - Lean Angle Estimator
=================================
Estimates continuous two-wheeler roll lean angle phi(t) at 50 Hz using a
complementary filter that blends high-frequency gyroscope roll integration (imu_gx)
with low-acceleration gravity vector observations (arctan2(imu_ay, imu_az)).

Mathematical Formulation:
-------------------------
1. Gyroscope Roll Integration:
   phi_gyro = phi_{t-1} + gx * dt

2. Gravity Vector Observation (Active during low non-gravitational acceleration):
   phi_accel = arctan2(ay, az)

3. Adaptive Gating:
   When |norm(a) - g| < accel_thresh, gravity estimate is trusted.
   phi_t = alpha * phi_gyro + (1 - alpha) * phi_accel
"""

import numpy as np
import pandas as pd

class LeanEstimator:
    def __init__(self, sample_rate=50.0, alpha=0.96, g_nominal=9.80665, accel_gate=2.5):
        """
        Args:
            sample_rate (float): IMU sampling frequency in Hz (default: 50.0).
            alpha (float): Complementary filter high-pass weighting (default: 0.96).
            g_nominal (float): Nominal Earth gravitational acceleration (m/s^2).
            accel_gate (float): Gating threshold on |norm(a) - g| to reject centripetal/road shocks.
        """
        self.dt = 1.0 / sample_rate
        self.alpha = alpha
        self.g = g_nominal
        self.accel_gate = accel_gate
        self.phi_rad = 0.0

    def update(self, ax, ay, az, gx, gy, gz):
        """
        Update the lean angle estimate with a single 6-axis IMU sample.

        Args:
            ax, ay, az: Accelerometer readings in m/s^2 (ax = forward, ay = lateral/right, az = vertical/down)
            gx, gy, gz: Gyroscope angular velocities in rad/s (gx = roll rate around forward axis)

        Returns:
            float: Estimated roll lean angle phi in radians (positive = right lean, negative = left lean).
        """
        # 1. Integrate roll rate from gyroscope
        phi_pred = self.phi_rad + gx * self.dt

        # 2. Compute total acceleration magnitude
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

        # 3. Gating check: if acceleration is close to 1g, trust gravity vector
        if abs(accel_mag - self.g) < self.accel_gate:
            # Gravity vector roll angle
            phi_meas = np.arctan2(ay, az)
            # Blend prediction and measurement
            self.phi_rad = self.alpha * phi_pred + (1.0 - self.alpha) * phi_meas
        else:
            # Under high lateral/vertical dynamic acceleration, rely on gyro integration
            self.phi_rad = phi_pred

        # Clamp physically reasonable limits for commuter two-wheelers (e.g. +/- 45 degrees)
        self.phi_rad = np.clip(self.phi_rad, np.radians(-45.0), np.radians(45.0))
        return self.phi_rad

    def reset(self, initial_phi_rad=0.0):
        self.phi_rad = initial_phi_rad

def estimate_trajectory_lean(df, sample_rate=50.0, alpha=0.96):
    """
    Process an entire telemetry dataframe and return estimated lean angles.

    Args:
        df: DataFrame containing ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']
        sample_rate: IMU sampling frequency in Hz
        alpha: Filter parameter

    Returns:
        np.ndarray: Array of estimated lean angles in radians
    """
    estimator = LeanEstimator(sample_rate=sample_rate, alpha=alpha)
    n = len(df)
    est_lean_rad = np.zeros(n)

    ax = df['imu_ax'].values
    ay = df['imu_ay'].values
    az = df['imu_az'].values
    gx = df['imu_gx'].values
    gy = df['imu_gy'].values
    gz = df['imu_gz'].values

    for i in range(n):
        est_lean_rad[i] = estimator.update(ax[i], ay[i], az[i], gx[i], gy[i], gz[i])

    return est_lean_rad

if __name__ == '__main__':
    import os
    dataset_path = os.path.join(os.path.dirname(__file__), 'synthetic_cornering_50hz.csv')
    if os.path.exists(dataset_path):
        data = pd.read_csv(dataset_path)
        est_phi = estimate_trajectory_lean(data)
        data['est_lean_rad'] = est_phi
        data['est_lean_deg'] = np.degrees(est_phi)

        if 'true_lean_deg' in data.columns:
            err = np.abs(data['est_lean_deg'] - data['true_lean_deg'])
            print(f"Lean Estimator Validation on {os.path.basename(dataset_path)}:")
            print(f"  Max True Lean: {data['true_lean_deg'].abs().max():.2f} deg")
            print(f"  Mean Absolute Error: {err.mean():.2f} deg")
            print(f"  Max Absolute Error: {err.max():.2f} deg")
