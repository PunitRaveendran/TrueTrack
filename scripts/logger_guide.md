# TrueTrack — Real-World On-Bike Telemetry Logging Guide

To validate TrueTrack against genuine road conditions and defuse the *"is this purely synthetic?"* question from hackathon judges, follow this 5-minute real-world data collection procedure.

---

## 1. Tooling & Smartphone Setup

Install either of the following free sensor recording apps on your Android test device:
- **Phyphox** (Recommended — RWTH Aachen University, completely open-source)
- **Sensor Logger** (Tsz-Ho Kwok, clean CSV export)

### Required Sensor Streams:
1. **Accelerometer** ( calibrated $a_x, a_y, a_z$ in $\text{m/s}^2$)
2. **Gyroscope** (calibrated $\omega_x, \omega_y, \omega_z$ in $\text{rad/s}$)
3. **Location / GNSS** (Latitude, Longitude, Speed, Accuracy, $1\text{--}10\text{ Hz}$)

### Sensor Configuration:
- **Sampling Frequency**: Set to **50 Hz** (or 100 Hz; TrueTrack downsamples to 50 Hz automatically).
- **Format**: CSV export with microsecond or millisecond timestamps.

---

## 2. Mounting & Rig Alignment

1. **Mounting Location**: Secure the smartphone in a standard two-wheeler handlebar phone holder or magnetic tank mount.
2. **Orientation**: Screen facing the rider, top of phone oriented forward along the vehicle's direction of travel ($+x = \text{forward}, +y = \text{lateral right}, +z = \text{down/gravity}$).
3. **Firm Attachment**: Ensure the mount does not rattle or pivot loosely — single-cylinder engine vibration (25–45 Hz) should transfer naturally from the chassis into the phone body.

---

## 3. Recommended 3-Minute Test Route

1. **Phase 1: Stationary Zero-Velocity Baseline (0 – 10 seconds)**:
   - Keep the vehicle stationary with the engine idling at ~1500 RPM.
   - *Purpose*: Captures the pure stationary engine vibration spectrum (25–35 Hz harmonic peak).

2. **Phase 2: Clear-Sky Alignment (10 – 40 seconds)**:
   - Ride normally in clear open sky (accelerate to 30–45 km/h).
   - *Purpose*: Solves for the initial Doppler velocity vector and phone-to-vehicle transformation matrix ($R_{\text{phone} \to \text{vehicle}}$).

3. **Phase 3: GPS Blackout Corridor / Curved Turn (40 – 90 seconds)**:
   - Enter an underpass, flyover shadow, or execute a continuous sweeping turn ($15^\circ\text{--}25^\circ$ lean angle).
   - *Purpose*: Demonstrates the Lean Angle Estimator $\phi(t)$ and Non-Holonomic Constraint (NHC) drift elimination under genuine pavement potholes and lean angles.

4. **Phase 4: Re-emergence & Lock (90 – 120 seconds)**:
   - Ride back out into full satellite visibility.
   - *Purpose*: Validates the 3.0-second Sigmoid reconciliation curve returning smoothly to GPS without teleportation.

---

## 4. Ingesting & Running Through TrueTrack

Export the `.csv` file from Phyphox or Sensor Logger to your computer and run:

```bash
# Ingest and convert raw phone log into TrueTrack 50 Hz format
python ml_engine/load_real_imu_data.py --input path/to/my_ride_log.csv --output ml_engine/real_ride_telemetry_50hz.csv

# Run the 5-way ablation benchmark on your real recording
python ml_engine/benchmark_drift.py --telemetry ml_engine/real_ride_telemetry_50hz.csv
```

The resulting trajectory and error metrics can then be loaded directly into the web cockpit or compared against the synthetic baseline.
