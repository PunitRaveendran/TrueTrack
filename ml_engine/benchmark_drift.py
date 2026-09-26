"""
TrueTrack - Algorithmic Drift Benchmarking Suite
================================================
Runs a synthetic 45-second GPS-blackout scenario with modeled two-wheeler
vibration and sensor drift. These outputs are not real-world performance results.

Evaluation caveat: blackout map snapping uses the per-frame ground-truth
position and heading as an oracle. The resulting map/full-stack error metrics
are ground-truth-assisted and must not be described as leak-free evaluation.

Evaluates these positioning strategies:

1. Naive Double Integration (Classical INS: Doppler velocity initialized, raw accel double-integrated)
2. Neural Velocity Dead-Reckoning (Normalized 6-axis 1D-CNN regresses speed + yaw rate)
3. TrueTrack Full Stack (Neural Velocity + Road-Manifold Snapping + Sigmoid Reconciliation)

Loads per-channel Z-score normalization parameters from norm_stats.json.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from train_and_export_model import NeuralInertial1DCNN

def run_benchmark(random_seed=42):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    csv_path = os.path.join('ml_engine', 'synthetic_telemetry_50hz.csv')
    df = pd.read_csv(csv_path)
    
    n_samples = len(df)
    dt = 0.02 # 50Hz
    
    # Load normalization statistics
    stats_path = os.path.join('ml_engine', 'norm_stats.json')
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    means = np.array(stats['means'])
    stds = np.array(stats['stds'])
    
    # Load trained PyTorch model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralInertial1DCNN().to(device)
    model.load_state_dict(torch.load(os.path.join('ml_engine', 'truetrack_1dcnn.pth'), map_location=device))
    model.eval()
    
    # Ground truth positions (meters)
    gt_x = df['true_x'].values
    gt_y = df['true_y'].values
    gt_speed = df['true_speed'].values
    gt_heading = np.radians(df['true_heading_deg'].values)
    is_blackout = df['is_blackout'].values
    timestamps = df['timestamp'].values
    
    # Raw 6-Axis IMU
    ax = df['imu_ax'].values
    ay = df['imu_ay'].values
    az = df['imu_az'].values
    gx = df['imu_gx'].values
    gy = df['imu_gy'].values
    gz = df['imu_gz'].values
    
    # -------------------------------------------------------------
    # Method 1: Naive Double Integration (Classical Inertial)
    # -------------------------------------------------------------
    naive_x = np.zeros(n_samples)
    naive_y = np.zeros(n_samples)
    naive_vx = np.zeros(n_samples)
    naive_vy = np.zeros(n_samples)
    naive_theta = np.zeros(n_samples)
    
    # Initialize from ground truth
    naive_x[0] = gt_x[0]
    naive_y[0] = gt_y[0]
    naive_theta[0] = gt_heading[0]
    naive_vx[0] = gt_speed[0] * np.cos(gt_heading[0])
    naive_vy[0] = gt_speed[0] * np.sin(gt_heading[0])
    
    for i in range(1, n_samples):
        if is_blackout[i] == 0:
            # Clear sky: GPS receiver bounds position with 1.2m jitter
            # and Doppler velocity filter maintains realistic 0.15 m/s velocity accuracy
            naive_x[i] = gt_x[i] + np.random.normal(0, 1.2)
            naive_y[i] = gt_y[i] + np.random.normal(0, 1.2)
            naive_theta[i] = gt_heading[i]
            # Doppler velocity fix
            naive_vx[i] = gt_speed[i] * np.cos(gt_heading[i]) + np.random.normal(0, 0.15)
            naive_vy[i] = gt_speed[i] * np.sin(gt_heading[i]) + np.random.normal(0, 0.15)
        else:
            # Blackout: Unassisted double integration under vibration and MEMS bias
            naive_theta[i] = naive_theta[i-1] + gz[i] * dt
            # Rotate body acc to world frame
            acc_world_x = -ax[i] * np.sin(naive_theta[i]) + ay[i] * np.cos(naive_theta[i])
            acc_world_y =  ax[i] * np.cos(naive_theta[i]) + ay[i] * np.sin(naive_theta[i])
            
            naive_vx[i] = naive_vx[i-1] + acc_world_x * dt
            naive_vy[i] = naive_vy[i-1] + acc_world_y * dt
            naive_x[i] = naive_x[i-1] + naive_vx[i] * dt
            naive_y[i] = naive_y[i-1] + naive_vy[i] * dt
            
    # -------------------------------------------------------------
    # Method 2: Neural Velocity Dead-Reckoning (Normalized 6-Axis 1D-CNN)
    # -------------------------------------------------------------
    raw_imu = np.column_stack([ax, ay, az, gx, gy, gz]) # (N, 6)
    norm_imu = (raw_imu - means) / stds
    
    window_size = 50
    padded_imu = np.vstack([np.tile(norm_imu[0], (window_size, 1)), norm_imu])
    
    windows = []
    for i in range(n_samples):
        w = padded_imu[i : i + window_size].T # (6, 50)
        windows.append(w)
    windows = np.array(windows, dtype=np.float32)
    
    batch_size = 256
    preds = []
    with torch.no_grad():
        for b in range(0, n_samples, batch_size):
            batch_w = torch.tensor(windows[b : b + batch_size]).to(device)
            p = model(batch_w).cpu().numpy()
            preds.append(p)
    preds = np.vstack(preds)
    neural_speed = np.clip(preds[:, 0], 0, 30)
    neural_yaw_rate = preds[:, 1]
    
    neural_x = np.zeros(n_samples)
    neural_y = np.zeros(n_samples)
    neural_theta = np.zeros(n_samples)
    neural_x[0] = gt_x[0]
    neural_y[0] = gt_y[0]
    neural_theta[0] = gt_heading[0]
    
    for i in range(1, n_samples):
        if is_blackout[i] == 0:
            neural_x[i] = gt_x[i] + np.random.normal(0, 1.2)
            neural_y[i] = gt_y[i] + np.random.normal(0, 1.2)
            neural_theta[i] = gt_heading[i]
        else:
            # Dead reckoning with normalized neural predicted speed and yaw rate
            neural_theta[i] = neural_theta[i-1] + neural_yaw_rate[i] * dt
            vx = neural_speed[i] * np.cos(neural_theta[i])
            vy = neural_speed[i] * np.sin(neural_theta[i])
            neural_x[i] = neural_x[i-1] + vx * dt
            neural_y[i] = neural_y[i-1] + vy * dt

    # -------------------------------------------------------------
    # Method 3: Naive INS + Map Manifold (Ablation: Map Alone without Neural)
    # -------------------------------------------------------------
    # Shows what happens if you snap classical double-integration to the road
    naive_map_x = np.zeros(n_samples)
    naive_map_y = np.zeros(n_samples)
    naive_map_x[0] = gt_x[0]
    naive_map_y[0] = gt_y[0]
    
    for i in range(1, n_samples):
        if is_blackout[i] == 0:
            naive_map_x[i] = naive_x[i]
            naive_map_y[i] = naive_y[i]
        else:
            road_tangent = np.array([np.cos(gt_heading[i]), np.sin(gt_heading[i])])
            road_normal = np.array([-np.sin(gt_heading[i]), np.cos(gt_heading[i])])
            gt_p = np.array([gt_x[i], gt_y[i]])
            vec = np.array([naive_x[i], naive_y[i]]) - gt_p
            along_track = np.dot(vec, road_tangent)
            cross_track = np.dot(vec, road_normal)
            bounded_cross = np.clip(cross_track, -1.8, 1.8)
            snapped_p = gt_p + along_track * road_tangent + bounded_cross * road_normal
            naive_map_x[i] = snapped_p[0]
            naive_map_y[i] = snapped_p[1]

    # -------------------------------------------------------------
    # Method 4: TrueTrack Full Stack (Neural Velocity + Road Manifold Snap + Sigmoid)
    # -------------------------------------------------------------
    snapped_x = np.zeros(n_samples)
    snapped_y = np.zeros(n_samples)
    snapped_theta = np.zeros(n_samples)
    snapped_x[0] = gt_x[0]
    snapped_y[0] = gt_y[0]
    snapped_theta[0] = gt_heading[0]
    
    reconcile_start_idx = np.where(is_blackout == 1)[0][-1] + 1
    reconcile_duration_sec = 3.0
    reconcile_steps = int(reconcile_duration_sec / dt)
    
    for i in range(1, n_samples):
        if is_blackout[i] == 0 and i < reconcile_start_idx:
            snapped_x[i] = neural_x[i]
            snapped_y[i] = neural_y[i]
            snapped_theta[i] = gt_heading[i]
        elif is_blackout[i] == 1:
            road_tangent = np.array([np.cos(gt_heading[i]), np.sin(gt_heading[i])])
            road_normal = np.array([-np.sin(gt_heading[i]), np.cos(gt_heading[i])])
            
            # Blend predicted yaw with the ground-truth road heading (oracle-assisted simulation).
            snapped_theta[i] = 0.85 * (snapped_theta[i-1] + neural_yaw_rate[i] * dt) + 0.15 * gt_heading[i]
            
            # Step position forward along filtered heading at neural speed
            step_vx = neural_speed[i] * np.cos(snapped_theta[i])
            step_vy = neural_speed[i] * np.sin(snapped_theta[i])
            est_x = snapped_x[i-1] + step_vx * dt
            est_y = snapped_y[i-1] + step_vy * dt
            
            # Snap cross-track deviation to physical roadway manifold (+- 1.8m: single-lane half-width)
            gt_p = np.array([gt_x[i], gt_y[i]])
            vec = np.array([est_x, est_y]) - gt_p
            along_track = np.dot(vec, road_tangent)
            cross_track = np.dot(vec, road_normal)
            bounded_cross = np.clip(cross_track, -1.8, 1.8)
            
            snapped_p = gt_p + along_track * road_tangent + bounded_cross * road_normal
            snapped_x[i] = snapped_p[0]
            snapped_y[i] = snapped_p[1]
        else:
            step = i - reconcile_start_idx
            if step < reconcile_steps:
                prog = step / reconcile_steps
                alpha = 1.0 / (1.0 + np.exp(-10.0 * (prog - 0.5)))
                target_gps_x = gt_x[i] + np.random.normal(0, 0.8)
                target_gps_y = gt_y[i] + np.random.normal(0, 0.8)
                dr_extrap_x = snapped_x[i-1] + neural_speed[i] * np.cos(gt_heading[i]) * dt
                dr_extrap_y = snapped_y[i-1] + neural_speed[i] * np.sin(gt_heading[i]) * dt
                snapped_x[i] = (1.0 - alpha) * dr_extrap_x + alpha * target_gps_x
                snapped_y[i] = (1.0 - alpha) * dr_extrap_y + alpha * target_gps_y
            else:
                snapped_x[i] = gt_x[i] + np.random.normal(0, 0.8)
                snapped_y[i] = gt_y[i] + np.random.normal(0, 0.8)

    # -------------------------------------------------------------
    # Summarize synthetic oracle-assisted errors in the ground-truth route frame.
    # -------------------------------------------------------------
    blackout_idx = np.where(is_blackout == 1)[0]
    blackout_dur = len(blackout_idx) * dt
    
    # Pre-allocate component arrays
    n_bo = len(blackout_idx)
    along_naive = np.zeros(n_bo)
    cross_naive = np.zeros(n_bo)
    along_naive_map = np.zeros(n_bo)
    cross_naive_map = np.zeros(n_bo)
    along_neural = np.zeros(n_bo)
    cross_neural = np.zeros(n_bo)
    along_tt = np.zeros(n_bo)
    cross_tt = np.zeros(n_bo)
    
    for k, idx in enumerate(blackout_idx):
        road_t = np.array([np.cos(gt_heading[idx]), np.sin(gt_heading[idx])])
        road_n = np.array([-np.sin(gt_heading[idx]), np.cos(gt_heading[idx])])
        gt_p = np.array([gt_x[idx], gt_y[idx]])
        
        # Naive error components
        v_naive = np.array([naive_x[idx], naive_y[idx]]) - gt_p
        along_naive[k] = np.dot(v_naive, road_t)
        cross_naive[k] = np.dot(v_naive, road_n)
        
        # Naive + Map error components
        v_nmap = np.array([naive_map_x[idx], naive_map_y[idx]]) - gt_p
        along_naive_map[k] = np.dot(v_nmap, road_t)
        cross_naive_map[k] = np.dot(v_nmap, road_n)
        
        # Neural alone error components
        v_neural = np.array([neural_x[idx], neural_y[idx]]) - gt_p
        along_neural[k] = np.dot(v_neural, road_t)
        cross_neural[k] = np.dot(v_neural, road_n)
        
        # TrueTrack full stack error components
        v_tt = np.array([snapped_x[idx], snapped_y[idx]]) - gt_p
        along_tt[k] = np.dot(v_tt, road_t)
        cross_tt[k] = np.dot(v_tt, road_n)
        
    err_naive = np.sqrt(along_naive**2 + cross_naive**2)
    err_naive_map = np.sqrt(along_naive_map**2 + cross_naive_map**2)
    err_neural = np.sqrt(along_neural**2 + cross_neural**2)
    err_truetrack = np.sqrt(along_tt**2 + cross_tt**2)
    
    metrics = {
        "evaluation_type": "synthetic single-route diagnostic",
        "independent_validation": False,
        "ground_truth_assistance": "During blackout, map projection and error decomposition use per-frame ground-truth position and heading.",
        "performance_claims_supported": False,
        "model_artifact_status": "Pre-existing weights and normalization were not regenerated with the corrected chronological trainer; stored validation provenance remains random-overlapping-window.",
        "blackout_duration_sec": float(round(blackout_dur, 1)),
        "sampling_frequency_hz": 50,
        "input_channels": 6,
        "lane_half_width_constraint_m": 1.8,
        "normalization": "Per-channel Z-score",
        "synthetic_signal_assumptions": "Modeled single-cylinder vibration and road roughness; not measured device or road characteristics",
        "results": {
            "1_naive_double_integration": {
                "description": "Classical double-integration (no neural filter, no map)",
                "max_total_error_m": float(round(np.max(err_naive), 2)),
                "end_of_blackout_total_error_m": float(round(err_naive[-1], 2)),
                "max_along_track_longitudinal_m": float(round(np.max(np.abs(along_naive)), 2)),
                "max_cross_track_lateral_m": float(round(np.max(np.abs(cross_naive)), 2)),
                "end_along_track_m": float(round(along_naive[-1], 2)),
                "end_cross_track_m": float(round(cross_naive[-1], 2)),
                "rmse_m": float(round(np.sqrt(np.mean(err_naive**2)), 2)),
                "avg_drift_rate_m_per_sec": float(round(err_naive[-1] / blackout_dur, 3))
            },
            "2_naive_plus_map_manifold": {
                "description": "Classical double-integration + road manifold constraint (Ablation: Map Alone)",
                "max_total_error_m": float(round(np.max(err_naive_map), 2)),
                "end_of_blackout_total_error_m": float(round(err_naive_map[-1], 2)),
                "max_along_track_longitudinal_m": float(round(np.max(np.abs(along_naive_map)), 2)),
                "max_cross_track_lateral_m": float(round(np.max(np.abs(cross_naive_map)), 2)),
                "end_along_track_m": float(round(along_naive_map[-1], 2)),
                "end_cross_track_m": float(round(cross_naive_map[-1], 2)),
                "rmse_m": float(round(np.sqrt(np.mean(err_naive_map**2)), 2)),
                "avg_drift_rate_m_per_sec": float(round(err_naive_map[-1] / blackout_dur, 3))
            },
            "3_neural_velocity_alone": {
                "description": "Normalized 6-axis 1D-CNN (Neural speed & yaw, no map constraint)",
                "max_total_error_m": float(round(np.max(err_neural), 2)),
                "end_of_blackout_total_error_m": float(round(err_neural[-1], 2)),
                "max_along_track_longitudinal_m": float(round(np.max(np.abs(along_neural)), 2)),
                "max_cross_track_lateral_m": float(round(np.max(np.abs(cross_neural)), 2)),
                "end_along_track_m": float(round(along_neural[-1], 2)),
                "end_cross_track_m": float(round(cross_neural[-1], 2)),
                "rmse_m": float(round(np.sqrt(np.mean(err_neural**2)), 2)),
                "avg_drift_rate_m_per_sec": float(round(err_neural[-1] / blackout_dur, 3))
            },
            "4_truetrack_full_stack": {
                "description": "Neural speed/yaw + ground-truth-directed road projection (simulated stack; not an EKF)",
                "max_total_error_m": float(round(np.max(err_truetrack), 2)),
                "end_of_blackout_total_error_m": float(round(err_truetrack[-1], 2)),
                "max_along_track_longitudinal_m": float(round(np.max(np.abs(along_tt)), 2)),
                "max_cross_track_lateral_m": float(round(np.max(np.abs(cross_tt)), 2)),
                "end_along_track_m": float(round(along_tt[-1], 2)),
                "end_cross_track_m": float(round(cross_tt[-1], 2)),
                "rmse_m": float(round(np.sqrt(np.mean(err_truetrack**2)), 2)),
                "avg_drift_rate_m_per_sec": float(round(err_truetrack[-1] / blackout_dur, 3))
            }
        }
    }
    
    # Save Metrics JSON
    metrics_path = os.path.join('ml_engine', 'benchmark_results.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print("--- BENCHMARK RESULTS (SIMULATION-BASED, NORMALIZED 6-AXIS) ---")
    print(json.dumps(metrics, indent=2))
    
    # Export downsampled web trajectory
    step = 5
    origin_lat = 17.4415
    origin_lon = 78.3760
    meters_to_lat = 1.0 / 110600.0
    meters_to_lon = 1.0 / (111320.0 * np.cos(np.radians(origin_lat)))
    
    web_data = []
    for i in range(0, n_samples, step):
        web_data.append({
            "t": round(float(timestamps[i]), 2),
            "speed_kmh": round(float(gt_speed[i] * 3.6), 1),
            "heading_deg": round(float(np.degrees(gt_heading[i])), 1),
            "is_blackout": int(is_blackout[i]),
            "gt": [round(origin_lat + gt_y[i] * meters_to_lat, 6), round(origin_lon + gt_x[i] * meters_to_lon, 6)],
            "naive": [round(origin_lat + naive_y[i] * meters_to_lat, 6), round(origin_lon + naive_x[i] * meters_to_lon, 6)],
            "neural": [round(origin_lat + neural_y[i] * meters_to_lat, 6), round(origin_lon + neural_x[i] * meters_to_lon, 6)],
            "truetrack": [round(origin_lat + snapped_y[i] * meters_to_lat, 6), round(origin_lon + snapped_x[i] * meters_to_lon, 6)],
            "err_naive_m": round(float(np.sqrt((naive_x[i]-gt_x[i])**2 + (naive_y[i]-gt_y[i])**2)), 1),
            "err_truetrack_m": round(float(np.sqrt((snapped_x[i]-gt_x[i])**2 + (snapped_y[i]-gt_y[i])**2)), 1)
        })
        
    js_export_path = os.path.join('web_app', 'js', 'corridor_data.js')
    with open(js_export_path, 'w') as f:
        f.write("// Auto-generated by TrueTrack Benchmark Engine\n")
        f.write(f"const BENCHMARK_METRICS = {json.dumps(metrics, indent=2)};\n\n")
        f.write(f"const SIMULATION_TELEMETRY = {json.dumps(web_data, indent=None)};\n")
    print(f"Exported synchronized web simulation telemetry to {js_export_path}")

if __name__ == '__main__':
    run_benchmark()
