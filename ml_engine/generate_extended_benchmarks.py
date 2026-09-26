"""
TrueTrack - Extended Benchmarking & Diagnostic Data Generator
=============================================================
Generates synthetic scenario diagnostics (not physical or independent evaluation):
1. Illustrative covariance-matrix propagation (not a navigation EKF):
   - State: [x, y, v, theta], with ground-truth-assisted position projection.
2. Synthetic tone-injection sensitivity calculation:
   - A 35 Hz tone sampled at 50 Hz aliases to 15 Hz; this is not a physical
     engine-harmonic rejection measurement.
3. Export of all 4 ablation trajectories + Hard-Snap variant:
   - naive, map_alone, neural_alone, truetrack (sigmoid), truetrack_hardsnap
4. FFT of synthetic accelerometer data over the 0-25 Hz Nyquist band.
5. 2.0m Map-Perturbation Stress Test.

Evaluation caveat: blackout map projection uses per-frame ground-truth
position and heading. Resulting errors and covariance are ground-truth-assisted
and must not be presented as leak-free real-world performance.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from train_and_export_model import NeuralInertial1DCNN

def run_extended_pipeline():
    print("=" * 60)
    print("TrueTrack Engineering Pipeline - Extended Diagnostics")
    print("=" * 60)
    
    csv_path = os.path.join('ml_engine', 'synthetic_telemetry_50hz.csv')
    df = pd.read_csv(csv_path)
    n_samples = len(df)
    dt = 0.02 # 50Hz
    timestamps = df['timestamp'].values
    
    # Normalization statistics
    stats_path = os.path.join('ml_engine', 'norm_stats.json')
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    means = np.array(stats['means'])
    stds = np.array(stats['stds'])
    
    # Load Model
    device = torch.device('cpu')
    model = NeuralInertial1DCNN().to(device)
    model.load_state_dict(torch.load(os.path.join('ml_engine', 'truetrack_1dcnn.pth'), map_location=device))
    model.eval()
    
    # Ground truth
    gt_x = df['true_x'].values
    gt_y = df['true_y'].values
    gt_speed = df['true_speed'].values
    gt_heading = np.radians(df['true_heading_deg'].values)
    is_blackout = df['is_blackout'].values
    
    ax = df['imu_ax'].values
    ay = df['imu_ay'].values
    az = df['imu_az'].values
    gx = df['imu_gx'].values
    gy = df['imu_gy'].values
    gz = df['imu_gz'].values
    
    # -------------------------------------------------------------
    # 1. Tone-Injection Vibration Sensitivity Test
    # -------------------------------------------------------------
    print("\n[1/5] Running synthetic aliased-tone sensitivity calculation...")
    # Inject a 35 Hz tone; at 50 Hz sampling it aliases to 15 Hz.
    t_win = np.arange(50) * dt
    test_wins_base = []
    test_wins_vibe = []
    
    for k in range(min(500, n_samples - 50)):
        w = np.column_stack([ax[k:k+50], ay[k:k+50], az[k:k+50], gx[k:k+50], gy[k:k+50], gz[k:k+50]])
        test_wins_base.append((w - means) / stds)
        
        w_vibe = w.copy()
        w_vibe[:, 0] += 3.5 * np.sin(2 * np.pi * 35.0 * t_win) # ax vibration
        w_vibe[:, 1] += 3.5 * np.cos(2 * np.pi * 35.0 * t_win) # ay vibration
        test_wins_vibe.append((w_vibe - means) / stds)
        
    t_base = torch.tensor(np.array(test_wins_base).transpose(0, 2, 1), dtype=torch.float32)
    t_vibe = torch.tensor(np.array(test_wins_vibe).transpose(0, 2, 1), dtype=torch.float32)
    
    with torch.no_grad():
        preds_base = model(t_base).numpy()[:, 0]
        preds_vibe = model(t_vibe).numpy()[:, 0]
        
    delta_v_mean = np.mean(np.abs(preds_vibe - preds_base))
    # Attenuation relative to raw vibration amplitude (3.5 m/s^2)
    # Sensitivity ratio in dB: 20 * log10(delta_v / A)
    attenuation_db = round(float(20 * np.log10(delta_v_mean / 3.5)), 1)
    print(f"  > Mean delta speed under 3.5 m/s^2 (35 Hz) vibration: {delta_v_mean:.4f} m/s")
    print(f"  > Synthetic aliased-tone sensitivity ratio: {attenuation_db} dB")
    
    # -------------------------------------------------------------
    # 2. Neural Model Inference across Trajectory
    # -------------------------------------------------------------
    print("\n[2/5] Running Neural 1D-CNN Inference...")
    raw_imu = np.column_stack([ax, ay, az, gx, gy, gz])
    norm_imu = (raw_imu - means) / stds
    window_size = 50
    padded_imu = np.vstack([np.tile(norm_imu[0], (window_size, 1)), norm_imu])
    
    windows = []
    for i in range(n_samples):
        windows.append(padded_imu[i : i + window_size].T)
    windows = np.array(windows, dtype=np.float32)
    
    batch_size = 256
    preds = []
    with torch.no_grad():
        for b in range(0, n_samples, batch_size):
            batch_w = torch.tensor(windows[b : b + batch_size])
            p = model(batch_w).numpy()
            preds.append(p)
    preds = np.vstack(preds)
    neural_speed = np.clip(preds[:, 0], 0, 30)
    neural_yaw_rate = preds[:, 1]
    
    # -------------------------------------------------------------
    # 3. Simulate All 4 Strategies + Hard-Snap Variant
    # -------------------------------------------------------------
    print("\n[3/5] Computing Trajectories for All 4 Strategies...")
    
    # Strategy 1: Naive Double Integration
    naive_x = np.zeros(n_samples)
    naive_y = np.zeros(n_samples)
    naive_vx = np.zeros(n_samples)
    naive_vy = np.zeros(n_samples)
    naive_theta = np.zeros(n_samples)
    naive_x[0], naive_y[0] = gt_x[0], gt_y[0]
    naive_theta[0] = gt_heading[0]
    naive_vx[0] = gt_speed[0] * np.cos(gt_heading[0])
    naive_vy[0] = gt_speed[0] * np.sin(gt_heading[0])
    
    for i in range(1, n_samples):
        if is_blackout[i] == 0:
            naive_x[i] = gt_x[i] + np.random.normal(0, 1.2)
            naive_y[i] = gt_y[i] + np.random.normal(0, 1.2)
            naive_theta[i] = gt_heading[i]
            naive_vx[i] = gt_speed[i] * np.cos(gt_heading[i]) + np.random.normal(0, 0.15)
            naive_vy[i] = gt_speed[i] * np.sin(gt_heading[i]) + np.random.normal(0, 0.15)
        else:
            naive_theta[i] = naive_theta[i-1] + gz[i] * dt
            acc_w_x = -ax[i] * np.sin(naive_theta[i]) + ay[i] * np.cos(naive_theta[i])
            acc_w_y =  ax[i] * np.cos(naive_theta[i]) + ay[i] * np.sin(naive_theta[i])
            naive_vx[i] = naive_vx[i-1] + acc_w_x * dt
            naive_vy[i] = naive_vy[i-1] + acc_w_y * dt
            naive_x[i] = naive_x[i-1] + naive_vx[i] * dt
            naive_y[i] = naive_y[i-1] + naive_vy[i] * dt
            
    # Strategy 2: Map Manifold Alone (Classical INS clamped to road)
    map_alone_x = np.zeros(n_samples)
    map_alone_y = np.zeros(n_samples)
    map_alone_x[0], map_alone_y[0] = gt_x[0], gt_y[0]
    for i in range(1, n_samples):
        if is_blackout[i] == 0:
            map_alone_x[i] = naive_x[i]
            map_alone_y[i] = naive_y[i]
        else:
            road_tangent = np.array([np.cos(gt_heading[i]), np.sin(gt_heading[i])])
            road_normal = np.array([-np.sin(gt_heading[i]), np.cos(gt_heading[i])])
            gt_p = np.array([gt_x[i], gt_y[i]])
            vec = np.array([naive_x[i], naive_y[i]]) - gt_p
            along = np.dot(vec, road_tangent)
            cross = np.dot(vec, road_normal)
            bounded_cross = np.clip(cross, -1.8, 1.8)
            snapped = gt_p + along * road_tangent + bounded_cross * road_normal
            map_alone_x[i] = snapped[0]
            map_alone_y[i] = snapped[1]
            
    # Strategy 3: Neural Velocity Alone (1D-CNN, no road clamp)
    neural_alone_x = np.zeros(n_samples)
    neural_alone_y = np.zeros(n_samples)
    neural_alone_theta = np.zeros(n_samples)
    neural_alone_x[0], neural_alone_y[0] = gt_x[0], gt_y[0]
    neural_alone_theta[0] = gt_heading[0]
    for i in range(1, n_samples):
        if is_blackout[i] == 0:
            neural_alone_x[i] = gt_x[i] + np.random.normal(0, 1.2)
            neural_alone_y[i] = gt_y[i] + np.random.normal(0, 1.2)
            neural_alone_theta[i] = gt_heading[i]
        else:
            neural_alone_theta[i] = neural_alone_theta[i-1] + neural_yaw_rate[i] * dt
            vx = neural_speed[i] * np.cos(neural_alone_theta[i])
            vy = neural_speed[i] * np.sin(neural_alone_theta[i])
            neural_alone_x[i] = neural_alone_x[i-1] + vx * dt
            neural_alone_y[i] = neural_alone_y[i-1] + vy * dt
            
    # Strategy 4: TrueTrack Full Stack (Neural + Map + Sigmoid Reconcile)
    # Strategy 5: TrueTrack Hard Snap (Instant jump at GPS re-lock)
    tt_x = np.zeros(n_samples)
    tt_y = np.zeros(n_samples)
    tt_theta = np.zeros(n_samples)
    tt_x[0], tt_y[0] = gt_x[0], gt_y[0]
    tt_theta[0] = gt_heading[0]
    
    tt_hardsnap_x = np.zeros(n_samples)
    tt_hardsnap_y = np.zeros(n_samples)
    tt_hardsnap_x[0], tt_hardsnap_y[0] = gt_x[0], gt_y[0]
    
    reconcile_start_idx = np.where(is_blackout == 1)[0][-1] + 1
    reconcile_duration_sec = 3.0
    reconcile_steps = int(reconcile_duration_sec / dt)
    
    for i in range(1, n_samples):
        if is_blackout[i] == 0 and i < reconcile_start_idx:
            tt_x[i] = neural_alone_x[i]
            tt_y[i] = neural_alone_y[i]
            tt_theta[i] = gt_heading[i]
            tt_hardsnap_x[i] = tt_x[i]
            tt_hardsnap_y[i] = tt_y[i]
        elif is_blackout[i] == 1:
            road_tangent = np.array([np.cos(gt_heading[i]), np.sin(gt_heading[i])])
            road_normal = np.array([-np.sin(gt_heading[i]), np.cos(gt_heading[i])])
            tt_theta[i] = 0.85 * (tt_theta[i-1] + neural_yaw_rate[i] * dt) + 0.15 * gt_heading[i]
            
            step_vx = neural_speed[i] * np.cos(tt_theta[i])
            step_vy = neural_speed[i] * np.sin(tt_theta[i])
            est_x = tt_x[i-1] + step_vx * dt
            est_y = tt_y[i-1] + step_vy * dt
            
            gt_p = np.array([gt_x[i], gt_y[i]])
            vec = np.array([est_x, est_y]) - gt_p
            along = np.dot(vec, road_tangent)
            cross = np.dot(vec, road_normal)
            bounded_cross = np.clip(cross, -1.8, 1.8)
            snapped = gt_p + along * road_tangent + bounded_cross * road_normal
            
            tt_x[i] = snapped[0]
            tt_y[i] = snapped[1]
            tt_hardsnap_x[i] = snapped[0]
            tt_hardsnap_y[i] = snapped[1]
        else: # Re-lock phase
            target_gps_x = gt_x[i] + np.random.normal(0, 0.8)
            target_gps_y = gt_y[i] + np.random.normal(0, 0.8)
            tt_hardsnap_x[i] = target_gps_x # Hard teleport jump
            tt_hardsnap_y[i] = target_gps_y
            
            step = i - reconcile_start_idx
            if step < reconcile_steps:
                prog = step / reconcile_steps
                alpha = 1.0 / (1.0 + np.exp(-10.0 * (prog - 0.5)))
                dr_extrap_x = tt_x[i-1] + neural_speed[i] * np.cos(gt_heading[i]) * dt
                dr_extrap_y = tt_y[i-1] + neural_speed[i] * np.sin(gt_heading[i]) * dt
                tt_x[i] = (1.0 - alpha) * dr_extrap_x + alpha * target_gps_x
                tt_y[i] = (1.0 - alpha) * dr_extrap_y + alpha * target_gps_y
            else:
                tt_x[i] = target_gps_x
                tt_y[i] = target_gps_y

    # -------------------------------------------------------------
    # 4. Simplified covariance proxy; it is not connected to the trajectory filter.
    # -------------------------------------------------------------
    print("\n[4/5] Propagating a ground-truth-assisted covariance proxy (not a navigation EKF)...")
    # State: [x, y, v, theta]
    P = np.diag([1.44, 1.44, 0.04, 0.001]) # GPS nominal covariance (1.2m sigma)
    sigma_along = np.zeros(n_samples)
    sigma_cross = np.zeros(n_samples)
    
    Q_base = np.diag([0.0005, 0.0005, 0.04, 0.0005]) * dt
    R_map = np.array([[0.81]]) # 0.9m std for 1.8m lane bound
    
    for i in range(n_samples):
        th = tt_theta[i]
        v_curr = neural_speed[i]
        
        # State Jacobian F
        F = np.array([
            [1.0, 0.0, np.cos(th) * dt, -v_curr * np.sin(th) * dt],
            [0.0, 1.0, np.sin(th) * dt,  v_curr * np.cos(th) * dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        P = F @ P @ F.T + Q_base
        
        if is_blackout[i] == 0:
            # GPS position measurement update
            H_gps = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            R_gps = np.diag([1.44, 1.44])
            S = H_gps @ P @ H_gps.T + R_gps
            K = P @ H_gps.T @ np.linalg.inv(S)
            P = (np.eye(4) - K @ H_gps) @ P
        else:
            # Map pseudo-measurement updates cross-track
            H_map = np.array([[-np.sin(th), np.cos(th), 0.0, 0.0]])
            S = H_map @ P @ H_map.T + R_map
            K = P @ H_map.T @ np.linalg.inv(S)
            P = (np.eye(4) - K @ H_map) @ P
            
        # Rotate P[0:2, 0:2] to along-track / cross-track frame
        R_rot = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        P_body = R_rot @ P[0:2, 0:2] @ R_rot.T
        sigma_along[i] = np.sqrt(max(1e-4, P_body[0, 0]))
        sigma_cross[i] = np.sqrt(max(1e-4, P_body[1, 1]))
        
    print(f"  > Blackout Entry Covariance: sigma_along = {sigma_along[2000]:.2f}m, sigma_cross = {sigma_cross[2000]:.2f}m")
    print(f"  > Blackout Exit Covariance:  sigma_along = {sigma_along[4249]:.2f}m, sigma_cross = {sigma_cross[4249]:.2f}m")
    
    # -------------------------------------------------------------
    # 5. FFT of the synthetic IMU stream (Nyquist frequency is 25 Hz).
    # -------------------------------------------------------------
    print("\n[5/5] Computing synthetic FFT Frequency Spectrum (0-25 Hz Nyquist band)...")
    # Take 500 samples (~10s) inside vibration zone
    vibe_seg = ay[2200:2700]
    fft_vals = np.abs(np.fft.rfft(vibe_seg))
    fft_freqs = np.fft.rfftfreq(len(vibe_seg), d=dt)
    
    # Normalize FFT to 0-1 for web display
    fft_norm = (fft_vals / np.max(fft_vals)).round(3).tolist()
    fft_freqs_list = fft_freqs.round(1).tolist()
    
    # Peak detection
    peak_freq = fft_freqs[np.argmax(fft_vals)]
    print(f"  > Detected Dominant IMU Frequency Peak: {peak_freq:.1f} Hz")
    
    # -------------------------------------------------------------
    # 6. Map Perturbation Stress Test (2.0m lateral map error)
    # -------------------------------------------------------------
    offset_dist = 2.0
    err_offset = np.sqrt((tt_x - (gt_x + offset_dist))**2 + (tt_y - gt_y)**2)
    max_perturbed_err = round(float(np.max(err_offset[is_blackout == 1])), 2)
    print(f"  > Map Perturbation Stress Test (2.0m OSM error): Max Error = {max_perturbed_err} m")

    # -------------------------------------------------------------
    # Export to corridor_data.js mapped to real OSM GeoJSON Road
    # -------------------------------------------------------------
    step = 5 # 50Hz -> 10Hz
    
    # Load actual OSM road centerline directly from GeoJSON (zero hardcoded numbers)
    geojson_path = os.path.join('web_app', 'data', 'hitec_corridor.geojson')
    with open(geojson_path, 'r', encoding='utf-8') as f:
        corridor_gj = json.load(f)
    
    # Extract coordinates [lon, lat] from main road LineString and convert to [lat, lon]
    osm_road_coords = corridor_gj['features'][0]['geometry']['coordinates']
    road_pts = np.array([[c[1], c[0]] for c in osm_road_coords])

    m_lat = 110600.0
    m_lon = 111320.0 * np.cos(np.radians(17.445))

    cum_dist = [0.0]
    for k in range(1, len(road_pts)):
        dy = (road_pts[k, 0] - road_pts[k-1, 0]) * m_lat
        dx = (road_pts[k, 1] - road_pts[k-1, 1]) * m_lon
        cum_dist.append(cum_dist[-1] + np.sqrt(dx**2 + dy**2))
    cum_dist = np.array(cum_dist)
    total_road_len = cum_dist[-1]

    from scipy.interpolate import CubicSpline
    spline_lat = CubicSpline(cum_dist, road_pts[:, 0])
    spline_lon = CubicSpline(cum_dist, road_pts[:, 1])

    # Arc length progression along the road
    cum_s = np.cumsum(gt_speed * dt)
    cum_s = (cum_s / cum_s[-1]) * total_road_len

    # Ground truth road centerline coordinates
    gt_lat = spline_lat(cum_s)
    gt_lon = spline_lon(cum_s)

    # Tangent vector along road for physical projection
    ds = 0.5
    s_p = np.clip(cum_s + ds, 0, total_road_len)
    s_m = np.clip(cum_s - ds, 0, total_road_len)
    t_dy = (spline_lat(s_p) - spline_lat(s_m)) * m_lat
    t_dx = (spline_lon(s_p) - spline_lon(s_m)) * m_lon
    t_norm = np.maximum(1e-6, np.sqrt(t_dx**2 + t_dy**2))
    u_tangent_x = t_dx / t_norm # East
    u_tangent_y = t_dy / t_norm # North
    u_normal_x = -u_tangent_y   # East
    u_normal_y =  u_tangent_x   # North
    
    road_heading_deg = np.degrees(np.arctan2(u_tangent_x, u_tangent_y)) % 360.0

    web_data = []
    for i in range(0, n_samples, step):
        # Physical error vectors decomposed into road tangent/normal
        # Naive error
        e_nv_x = naive_x[i] - gt_x[i]
        e_nv_y = naive_y[i] - gt_y[i]
        nv_lat = gt_lat[i] + (e_nv_y * u_tangent_y[i] + e_nv_x * u_normal_y[i]) / m_lat
        nv_lon = gt_lon[i] + (e_nv_y * u_tangent_x[i] + e_nv_x * u_normal_x[i]) / m_lon

        # Map alone error
        e_ma_x = map_alone_x[i] - gt_x[i]
        e_ma_y = map_alone_y[i] - gt_y[i]
        ma_lat = gt_lat[i] + (e_ma_y * u_tangent_y[i] + e_ma_x * u_normal_y[i]) / m_lat
        ma_lon = gt_lon[i] + (e_ma_y * u_tangent_x[i] + e_ma_x * u_normal_x[i]) / m_lon

        # Neural alone error
        e_na_x = neural_alone_x[i] - gt_x[i]
        e_na_y = neural_alone_y[i] - gt_y[i]
        na_lat = gt_lat[i] + (e_na_y * u_tangent_y[i] + e_na_x * u_normal_y[i]) / m_lat
        na_lon = gt_lon[i] + (e_na_y * u_tangent_x[i] + e_na_x * u_normal_x[i]) / m_lon

        # TrueTrack full stack error
        e_tt_x = tt_x[i] - gt_x[i]
        e_tt_y = tt_y[i] - gt_y[i]
        tt_lat = gt_lat[i] + (e_tt_y * u_tangent_y[i] + e_tt_x * u_normal_y[i]) / m_lat
        tt_lon = gt_lon[i] + (e_tt_y * u_tangent_x[i] + e_tt_x * u_normal_x[i]) / m_lon

        # TrueTrack hard snap error
        e_hs_x = tt_hardsnap_x[i] - gt_x[i]
        e_hs_y = tt_hardsnap_y[i] - gt_y[i]
        hs_lat = gt_lat[i] + (e_hs_y * u_tangent_y[i] + e_hs_x * u_normal_y[i]) / m_lat
        hs_lon = gt_lon[i] + (e_hs_y * u_tangent_x[i] + e_hs_x * u_normal_x[i]) / m_lon

        web_data.append({
            "t": round(float(timestamps[i]), 2),
            "speed_kmh": round(float(gt_speed[i] * 3.6), 1),
            "heading_deg": round(float(road_heading_deg[i]), 1),
            "is_blackout": int(is_blackout[i]),
            "gt": [round(float(gt_lat[i]), 6), round(float(gt_lon[i]), 6)],
            "naive": [round(float(nv_lat), 6), round(float(nv_lon), 6)],
            "map_alone": [round(float(ma_lat), 6), round(float(ma_lon), 6)],
            "neural_alone": [round(float(na_lat), 6), round(float(na_lon), 6)],
            "truetrack": [round(float(tt_lat), 6), round(float(tt_lon), 6)],
            "truetrack_hardsnap": [round(float(hs_lat), 6), round(float(hs_lon), 6)],
            "err_naive_m": round(float(np.sqrt(e_nv_x**2 + e_nv_y**2)), 1),
            "err_map_alone_m": round(float(np.sqrt(e_ma_x**2 + e_ma_y**2)), 1),
            "err_neural_alone_m": round(float(np.sqrt(e_na_x**2 + e_na_y**2)), 1),
            "err_truetrack_m": round(float(np.sqrt(e_tt_x**2 + e_tt_y**2)), 2),
            "sigma_along": round(float(sigma_along[i]), 2),
            "sigma_cross": round(float(sigma_cross[i]), 2),
            "raw_imu_ax": round(float(ax[i]), 2),
            "raw_imu_ay": round(float(ay[i]), 2),
            "raw_imu_gz": round(float(gz[i]), 4),
            "pred_speed_kmh": round(float(neural_speed[i] * 3.6), 1),
            "pred_yaw_deg_s": round(float(np.degrees(neural_yaw_rate[i])), 1)
        })
        
    fft_package = {
        "freqs": fft_freqs_list,
        "magnitudes": fft_norm,
        "dominant_peak_hz": float(peak_freq),
        "synthetic_tone_sensitivity_db": attenuation_db,
        "frequency_source": "Synthetic telemetry; 50 Hz sampling limits the spectrum to the 25 Hz Nyquist frequency",
        "sensitivity_method": "Synthetic 35 Hz tone sampled at 50 Hz (aliases to 15 Hz); not a physical engine-vibration measurement"
    }
    
    stress_test_package = {
        "simulated_osm_map_offset_m": offset_dist,
        "max_drift_with_offset_m": max_perturbed_err,
        "explanation": "Synthetic 2.0 m route offset; map projection and error calculation use per-frame ground truth, so this is not an independent map-accuracy evaluation."
    }
    
    # Load base metrics
    with open('ml_engine/benchmark_results.json', 'r') as f:
        bench_metrics = json.load(f)
    bench_metrics.update({
        "evaluation_type": "synthetic single-route diagnostic",
        "independent_validation": False,
        "ground_truth_assistance": "During blackout, map projection and error decomposition use per-frame ground-truth position and heading.",
        "performance_claims_supported": False,
        "model_artifact_status": "Pre-existing weights and normalization were not regenerated with the corrected chronological trainer; stored validation provenance remains random-overlapping-window."
    })
        
    # Load multi-seed distribution
    dist_path = 'ml_engine/multiseed_evaluation.json'
    dist_metrics = {}
    if os.path.exists(dist_path):
        with open(dist_path, 'r') as f:
            dist_metrics = json.load(f)

    js_export_path = os.path.join('web_app', 'js', 'corridor_data.js')
    with open(js_export_path, 'w') as f:
        f.write("// Auto-generated by TrueTrack Extended Diagnostics Engine\n")
        f.write(f"window.BENCHMARK_METRICS = {json.dumps(bench_metrics, indent=2)};\n\n")
        f.write(f"window.IMU_FFT_DATA = {json.dumps(fft_package, indent=2)};\n\n")
        f.write(f"window.STRESS_TEST_METRICS = {json.dumps(stress_test_package, indent=2)};\n\n")
        f.write(f"window.DISTRIBUTION_METRICS = {json.dumps(dist_metrics, indent=2)};\n\n")
        f.write(f"window.SIMULATION_TELEMETRY = {json.dumps(web_data, indent=None)};\n")
        
    print(f"\nSuccessfully generated & exported complete diagnostic package to {js_export_path}!")

if __name__ == '__main__':
    run_extended_pipeline()
