import os
import pandas as pd
import numpy as np

folders = [
    '45_46-2026-09-25_10-54-49',
    'Varadarajapuram-2026-09-25_11-12-54',
    'Rohini_Theatre_Koyambedu-2026-09-25_13-05-40'
]

for f in folders:
    print('='*70)
    print('ANALYZING RECORDING:', f)
    
    # Metadata
    meta_path = os.path.join(f, 'Metadata.csv')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as fp:
                print('Device / App Metadata:')
                for line in fp.readlines()[:8]:
                    print('  ', line.strip())
        except Exception as e:
            print('  Meta read error:', e)

    # Accelerometer
    acc_path = os.path.join(f, 'Accelerometer.csv')
    if os.path.exists(acc_path):
        df_acc = pd.read_csv(acc_path)
        t_col = 'seconds_elapsed' if 'seconds_elapsed' in df_acc.columns else 'time'
        t_vals = df_acc[t_col].values
        duration = t_vals[-1] - t_vals[0]
        hz = len(df_acc) / duration if duration > 0 else 0
        print(f"\n[Accelerometer]: {len(df_acc):,} samples")
        print(f"  Duration: {duration:.1f} s ({duration/60:.1f} min)")
        print(f"  Estimated Rate: {hz:.1f} Hz")
        print(f"  Mean (ax, ay, az): ({df_acc['x'].mean():.2f}, {df_acc['y'].mean():.2f}, {df_acc['z'].mean():.2f}) m/s^2")
        print(f"  Std Dev (Vibration): ({df_acc['x'].std():.2f}, {df_acc['y'].std():.2f}, {df_acc['z'].std():.2f}) m/s^2")
        total_a = np.sqrt(df_acc['x']**2 + df_acc['y']**2 + df_acc['z']**2)
        print(f"  Total Accel Norm: mean={total_a.mean():.2f} m/s^2, max={total_a.max():.2f} m/s^2 (shock spikes!)")

    # Gyroscope
    gyro_path = os.path.join(f, 'Gyroscope.csv')
    if os.path.exists(gyro_path):
        df_gyro = pd.read_csv(gyro_path)
        print(f"\n[Gyroscope]: {len(df_gyro):,} samples")
        print(f"  Mean (gx, gy, gz): ({df_gyro['x'].mean():.3f}, {df_gyro['y'].mean():.3f}, {df_gyro['z'].mean():.3f}) rad/s")
        print(f"  Std Dev: ({df_gyro['x'].std():.3f}, {df_gyro['y'].std():.3f}, {df_gyro['z'].std():.3f}) rad/s")

    # Orientation (Roll Lean Angle)
    orient_path = os.path.join(f, 'Orientation.csv')
    if os.path.exists(orient_path):
        df_orient = pd.read_csv(orient_path)
        print(f"\n[Orientation / Lean]: {len(df_orient):,} samples")
        # Roll angle
        roll_cols = [c for c in df_orient.columns if 'roll' in c.lower() or 'pitch' in c.lower() or 'yaw' in c.lower() or c in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']]
        print(f"  Orientation columns: {df_orient.columns.tolist()}")
        if 'roll' in df_orient.columns:
            print(f"  Roll (Lean) range: min={np.degrees(df_orient['roll'].min()):.1f}°, max={np.degrees(df_orient['roll'].max()):.1f}°")

    # Location / GPS
    loc_path = os.path.join(f, 'Location.csv')
    if os.path.exists(loc_path):
        df_loc = pd.read_csv(loc_path)
        t_col = 'seconds_elapsed' if 'seconds_elapsed' in df_loc.columns else 'time'
        print(f"\n[Location / GPS]: {len(df_loc)} GPS fixes")
        if 'speed' in df_loc.columns:
            speeds_kmh = df_loc['speed'] * 3.6
            print(f"  Speed: mean={speeds_kmh.mean():.1f} km/h, max={speeds_kmh.max():.1f} km/h")
        if 'accuracy' in df_loc.columns:
            acc = df_loc['accuracy']
            print(f"  Accuracy: best={acc.min():.1f} m, median={acc.median():.1f} m, worst={acc.max():.1f} m")
            degraded = (acc > 15.0).sum()
            print(f"  Degraded GPS fixes (>15m error): {degraded} fixes ({degraded/len(df_loc)*100:.1f}%)")
        
        t_loc = df_loc[t_col].values
        dt = np.diff(t_loc)
        if len(dt) > 0:
            print(f"  Update Interval: median={np.median(dt):.2f}s, max outage gap={dt.max():.2f}s")
            large_gaps = dt[dt > 3.0]
            if len(large_gaps) > 0:
                print(f"  >> DETECTED {len(large_gaps)} GPS DROPOUTS (>3s outage): {large_gaps.tolist()} seconds!")
