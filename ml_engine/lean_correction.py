"""
TrueTrack - Two-Wheeler Lean Angle Correction & NHC Module
===========================================================
Addresses the fundamental failure of classical 4-wheel automotive dead-reckoning
on two-wheelers during cornering in GPS-denied corridors.

The repository's declared phone/body frame is X=lateral, Y=forward, Z=vertical.
Roll is represented as a rotation of the lateral/vertical (X/Z) plane. The
synthetic generator does not currently provide an independent true-lean signal,
so this module is an implementation experiment, not a validated NHC pipeline.

The TrueTrack Lean-Correction Fix:
---------------------------------
1. De-roll IMU Acceleration (Pre-CNN Normalization):
   Rotates measured (ax, az) by the inverse roll rotation so the neural network and kinematic
   integrators receive pure vehicle-frame acceleration uncorrupted by tilted gravity.
       [ax_derolled] = [ cos(phi)  -sin(phi) ] [ ax ]
       [az_derolled]   [ sin(phi)   cos(phi) ] [ az ]

2. Lean-Corrected Non-Holonomic Constraint (NHC) in EKF:
   The corresponding lateral residual is vx*cos(phi) - vz*sin(phi).
"""

import numpy as np
import pandas as pd
try:
    from ml_engine.lean_estimator import LeanEstimator, estimate_trajectory_lean
except ImportError:
    from lean_estimator import LeanEstimator, estimate_trajectory_lean

def deroll_imu_accelerations(ax, ay, az, phi_rad):
    """
    Apply the inverse lean rotation in the declared X=lateral, Y=forward,
    Z=vertical phone/body frame. This matches LeanCorrector.kt.

    Args:
        ax: Lateral measured acceleration
        ay: Forward acceleration (unchanged by roll)
        az: Vertical measured acceleration
        phi_rad: Roll lean angle in radians (positive = lean right)

    Returns:
        tuple (ax_derolled, ay_derolled, az_derolled)
    """
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)

    ax_derolled = ax * cos_phi - az * sin_phi
    ay_derolled = ay
    az_derolled = ax * sin_phi + az * cos_phi

    return ax_derolled, ay_derolled, az_derolled

def reroll_velocity(vx_derolled, vy_derolled, vz_derolled, phi_rad):
    """
    Transform velocities from the de-rolled vehicle frame back to sensor body frame.
    """
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)

    vx = vx_derolled * cos_phi + vz_derolled * sin_phi
    vy = vy_derolled
    vz = -vx_derolled * sin_phi + vz_derolled * cos_phi

    return vx, vy, vz

def compute_nhc_residual(v_body, phi_rad):
    """
    Calculate Non-Holonomic Constraint residual under lean angle phi.
    For a two-wheeler, zero lateral speed applies in the de-rolled plane:
        v_lateral_derolled = v_x * cos(phi) - v_z * sin(phi) ~= 0
    """
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    vx = v_body[0]
    vz = v_body[2] if len(v_body) > 2 else 0.0

    lateral_slip = vx * cos_phi - vz * sin_phi
    return lateral_slip

def apply_lean_correction_to_dataset(df):
    """
    Takes a 50Hz telemetry dataframe, computes continuous roll lean phi(t),
    and appends de-rolled acceleration channels.
    """
    df_out = df.copy()
    est_phi = estimate_trajectory_lean(df_out)
    df_out['est_lean_rad'] = est_phi
    df_out['est_lean_deg'] = np.degrees(est_phi)

    ax = df_out['imu_ax'].values
    ay = df_out['imu_ay'].values
    az = df_out['imu_az'].values

    ax_c, ay_c, az_c = deroll_imu_accelerations(ax, ay, az, est_phi)
    df_out['imu_ax_derolled'] = ax_c
    df_out['imu_ay_derolled'] = ay_c
    df_out['imu_az_derolled'] = az_c

    return df_out

if __name__ == '__main__':
    import os
    dataset_path = os.path.join(os.path.dirname(__file__), 'synthetic_cornering_50hz.csv')
    if os.path.exists(dataset_path):
        data = pd.read_csv(dataset_path)
        corrected = apply_lean_correction_to_dataset(data)
        print("Lean Correction Module Test:")
        print(f"  Processed {len(corrected)} samples.")
        print(f"  Raw imu_ax std: {data['imu_ax'].std():.3f} m/s^2")
        print(f"  De-rolled imu_ax_derolled std: {corrected['imu_ax_derolled'].std():.3f} m/s^2")
        print(f"  Max estimated lean: {corrected['est_lean_deg'].abs().max():.2f} deg")
