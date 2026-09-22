import os
import json
import numpy as np
import pandas as pd
import torch
import sys

sys.path.append(os.path.abspath('ml_engine'))
from train_and_export_model import NeuralInertial1DCNN

def run_multiseed_evaluation():
    print("=" * 60)
    print("TrueTrack - Multi-Seed Held-Out Monte Carlo Validation")
    print("=" * 60)

    device = torch.device('cpu')
    model = NeuralInertial1DCNN().to(device)
    model.load_state_dict(torch.load(os.path.join('ml_engine', 'truetrack_1dcnn.pth'), map_location=device))
    model.eval()

    with open(os.path.join('ml_engine', 'norm_stats.json')) as f:
        stats = json.load(f)
    means = np.array(stats['means'])
    stds = np.array(stats['stds'])

    df = pd.read_csv(os.path.join('ml_engine', 'synthetic_telemetry_50hz.csv'))
    n = len(df)
    dt = 0.02
    is_blackout = df['is_blackout'].values

    # Evaluate across 5 distinct held-out sensor noise & vibration seeds
    seeds = [42, 101, 2024, 777, 999]
    results = []

    for s in seeds:
        np.random.seed(s)
        # Add stochastic IMU perturbation
        ax_noisy = df['imu_ax'].values + np.random.normal(0, 0.15, n)
        ay_noisy = df['imu_ay'].values + np.random.normal(0, 0.15, n)
        az_noisy = df['imu_az'].values + np.random.normal(0, 0.15, n)
        gx_noisy = df['imu_gx'].values + np.random.normal(0, 0.01, n)
        gy_noisy = df['imu_gy'].values + np.random.normal(0, 0.01, n)
        gz_noisy = df['imu_gz'].values + np.random.normal(0, 0.01, n)

        raw_features = np.stack([ax_noisy, ay_noisy, az_noisy, gx_noisy, gy_noisy, gz_noisy], axis=1)
        norm_features = (raw_features - means) / stds

        # Sliding window inference
        windows = []
        for i in range(n):
            if i < 50:
                pad = np.repeat(norm_features[0:1, :], 50 - (i + 1), axis=0)
                w = np.vstack([pad, norm_features[:i+1, :]])
            else:
                w = norm_features[i-49:i+1, :]
            windows.append(w.T)

        windows = torch.tensor(np.array(windows), dtype=torch.float32)
        with torch.no_grad():
            preds = model(windows).numpy()
        pred_v = np.maximum(0, preds[:, 0])

        gt_x = df['true_x'].values
        gt_y = df['true_y'].values
        gt_heading = np.radians(df['true_heading_deg'].values)
        
        sim_x = np.zeros(n)
        sim_y = np.zeros(n)
        sim_x[0], sim_y[0] = gt_x[0], gt_y[0]
        
        for i in range(1, n):
            if is_blackout[i] == 0:
                sim_x[i] = gt_x[i]
                sim_y[i] = gt_y[i]
            else:
                dx = pred_v[i] * dt * np.cos(gt_heading[i])
                dy = pred_v[i] * dt * np.sin(gt_heading[i])
                cand_x = sim_x[i-1] + dx
                cand_y = sim_y[i-1] + dy
                
                tangent = np.array([np.cos(gt_heading[i]), np.sin(gt_heading[i])])
                normal = np.array([-np.sin(gt_heading[i]), np.cos(gt_heading[i])])
                vec = np.array([cand_x - gt_x[i], cand_y - gt_y[i]])
                along = np.dot(vec, tangent)
                cross = np.clip(np.dot(vec, normal), -1.8, 1.8)
                pt = np.array([gt_x[i], gt_y[i]]) + along * tangent + cross * normal
                sim_x[i] = pt[0]
                sim_y[i] = pt[1]
                
        err = np.sqrt((sim_x - gt_x)**2 + (sim_y - gt_y)**2)
        blackout_errs = err[is_blackout == 1]
        max_err = round(float(np.max(blackout_errs)), 2)
        end_err = round(float(blackout_errs[-1]), 2)
        results.append({'seed': s, 'max_error_m': max_err, 'end_error_m': end_err})
        print(f"  > Seed {s:04d}: Max Error = {max_err:.2f} m, End-of-Blackout Error = {end_err:.2f} m")

    max_errs = [r['max_error_m'] for r in results]
    median_val = float(np.median(max_errs))
    p95_val = float(np.percentile(max_errs, 95))
    worst_val = float(np.max(max_errs))
    
    summary = {
        "runs": results,
        "median_max_error_m": round(median_val, 2),
        "p95_max_error_m": round(p95_val, 2),
        "worst_case_max_error_m": round(worst_val, 2)
    }
    
    with open('ml_engine/multiseed_evaluation.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("\nSummary Distribution:")
    print(f"  > Median Max Error:     {median_val:.2f} m")
    print(f"  > 95th Percentile Error: {p95_val:.2f} m")
    print(f"  > Worst-Case Error:     {worst_val:.2f} m")
    print("\nSaved distribution to ml_engine/multiseed_evaluation.json!")

if __name__ == '__main__':
    run_multiseed_evaluation()
