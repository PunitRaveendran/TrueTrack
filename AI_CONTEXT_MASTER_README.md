# TrueTrack: Master System Architecture & Full AI Context Guide

> **NOTICE FOR AI ASSISTANTS / AGENTS RESUMING THIS TASK:**  
> This file is a self-contained, 100% comprehensive briefing of the entire **TrueTrack** project, its codebase, mathematics, hardware targets, real vs. simulated telemetry, bugs fixed, and operational procedures. Read this document completely to gain instant context and assist the user without asking repetitive architectural or state questions.

---

## 📌 Executive Summary

* **Project Name:** **TrueTrack** (Edge-AI Neural-Inertial Navigation for GPS-Denied Corridors)
* **Target Hardware:** Consumer 2-Wheelers (motorcycles/scooters) navigating with an **iQOO Performance Smartphone** powered by **Qualcomm Snapdragon Mobile Platform (Hexagon NPU)**.
* **Core Problem:** Two-wheelers in India lack OBD-II ports, CAN-bus telemetry, and wheel-speed odometry. When riders enter tunnels, underpasses, or multi-layer flyovers (e.g., Hyderabad Mindspace Underpass, Chennai Koyambedu, Delhi Pragati Maidan), satellite GNSS/GPS drops completely. Standard navigation apps freeze, jump into buildings, or drift uncontrollably, causing missed underground forks and dangerous sudden stops. Underground cellular 4G/5G is also lost, so cloud APIs are unusable.
* **The Solution:** TrueTrack runs a lightweight **1D-CNN (Temporal Convolutional Network)** locally on the Snapdragon Hexagon NPU using raw 50 Hz 6-axis MEMS IMU sensors (accelerometer + gyroscope). Instead of double-integrating accelerations (which diverges quadratically $t^2$), the neural network directly predicts forward velocity $\hat{v}$ and yaw rate $\hat{\omega}$, filters out single-cylinder engine vibration (-8.3 dB attenuation), applies motorcycle roll/lean compensation, binds trajectories to an offline OpenStreetMap vector manifold, and executes smooth sigmoidal blending when GPS reconnects. Zero cloud calls, zero external hardware.

---

## 🗺️ Complete Workspace Map & Asset Inventory

Root Workspace Directory: `c:\Jala deee\IQOO\`

```
c:\Jala deee\IQOO\
├── AI_CONTEXT_MASTER_README.md          # THIS MASTER CONTEXT GUIDE
├── PROJECT_OVERVIEW.md                  # Quick pointer to TrueTrack
│
├── 45_46-2026-09-25_10-54-49/           # [REAL DATASET 1] Chennai 45/46 Road Drive
│   ├── Accelerometer.csv                # 50-100 Hz tri-axial acceleration (m/s²)
│   ├── Gyroscope.csv                    # 50-100 Hz angular velocity (rad/s)
│   ├── Location.csv                     # GPS ground truth (lat, lon, speed, bearing, alt)
│   ├── Compass.csv, Gravity.csv, Orientation.csv, TotalAcceleration.csv
│   └── Metadata.csv                     # Sensor Logger session info
│
├── Rohini_Theatre_Koyambedu-2026-09-25_13-05-40/ # [REAL DATASET 2] Chennai Koyambedu Drive
│   ├── Accelerometer.csv, Gyroscope.csv, Location.csv, Compass.csv, Gravity.csv ...
│
├── Varadarajapuram-2026-09-25_11-12-54/ # [REAL DATASET 3] Chennai Varadarajapuram Drive
│   ├── Accelerometer.csv, Gyroscope.csv, Location.csv, Compass.csv, Gravity.csv ...
│
└── TrueTrack/                           # Main Application & ML Repository (Git Repo)
    ├── README.md                        # Project documentation & benchmark overview
    ├── PROJECT_OVERVIEW.md              # Module inventory & file guide
    ├── AI_CONTEXT_MASTER_README.md      # Duplicate copy of Master Context Guide
    │
    ├── android_app/                     # Native Android Kotlin App
    │   ├── build.gradle                 # Project Gradle config
    │   ├── settings.gradle              # Module includes (:app)
    │   └── app/
    │       ├── build.gradle             # Android SDK 34, ONNX Runtime (onnxruntime-android), Osmdroid
    │       └── src/main/
    │           ├── AndroidManifest.xml  # Sensors, Fine Location, Internet/Localhost permissions
    │           ├── assets/              # Offline on-device assets
    │           │   ├── truetrack_model.onnx       # 103.7 KB INT8 Quantized 1D-CNN
    │           │   ├── norm_stats.json            # Z-Score normalization parameters (means/stds)
    │           │   ├── npu_model_spec.json        # Snapdragon Hexagon NPU specs & latency
    │           │   ├── rohini_telemetry.json      # 52s real Koyambedu drive telemetry (10 Hz)
    │           │   ├── varadarajapuram_telemetry.json # 100s real Varadarajapuram drive (10 Hz)
    │           │   ├── 45_46_telemetry.json       # 100s real 45/46 Corridor drive (10 Hz)
    │           │   ├── corridor_telemetry.json    # 130s HITEC City Underpass sim (10 Hz)
    │           │   └── tiles/                     # Offline OpenStreetMap raster tiles (Zooms 13-17)
    │           ├── java/com/truetrack/navigation/
    │           │   ├── MainActivity.kt            # Main controller: routes, sensors, Osmdroid map, sim loop
    │           │   ├── AudioCueManager.kt         # Text-to-Speech (TTS) & ToneGenerator voice guidance
    │           │   ├── LeanCorrector.kt           # Roll dynamics & lean-angle estimator
    │           │   └── TelemetryStreamServer.kt   # WebSocket (port 8765) streaming server
    │           └── res/
    │               ├── layout/activity_main.xml   # High-contrast cockpit HUD layout
    │               ├── drawable/                  # Navigation arrows, vector badges
    │               └── values/{colors,strings,themes}.xml
    │
    ├── ml_engine/                       # PyTorch Neural Network & Benchmark Engine
    │   ├── train_and_export_model.py    # 1D-CNN architecture, training loop, ONNX & TorchScript export
    │   ├── synthetic_data_generator.py  # 50 Hz physics IMU generator (engine vibration, road shocks)
    │   ├── benchmark_drift.py           # 4-way ablation benchmark (Naive vs Map vs Neural vs TrueTrack)
    │   ├── generate_extended_benchmarks.py # Multi-speed, multi-duration extended benchmarks
    │   ├── evaluate_multiseed_distribution.py # 100-seed Monte Carlo evaluation
    │   ├── lean_estimator.py            # Physics-based centripetal roll estimator
    │   ├── lean_correction.py           # Roll dynamics derotation matrix
    │   ├── load_real_imu_data.py        # Parser for Sensor Logger / Phyphox CSVs
    │   ├── truetrack_1dcnn.pth          # PyTorch weights
    │   ├── truetrack_model.onnx         # Exported ONNX model (103.7 KB)
    │   ├── truetrack_model.torchscript  # Exported TorchScript JIT model (133.3 KB)
    │   ├── norm_stats.json              # Per-channel sensor means & stds
    │   ├── npu_model_spec.json          # Hexagon NPU target performance metrics
    │   ├── benchmark_results.json       # Benchmark metrics output
    │   └── hitec_road_nominatim.geojson # Hyderabad HITEC underpass road geometry
    │
    ├── web_app/                         # Interactive Web Flight Recorder Cockpit
    │   ├── index.html                   # Zero-scroll 1920x1080 cockpit dashboard
    │   ├── styles.css                   # Glassmorphism dark-mode styling
    │   ├── lib/
    │   │   ├── maplibre-gl.js           # 100% offline bundled MapLibre WebGL engine
    │   │   └── maplibre-gl.css          # MapLibre styling
    │   ├── data/
    │   │   └── hitec_corridor.geojson   # Real OSM surveyed road centerline vector manifold
    │   ├── tiles/                       # Offline raster tiles (Zooms 14-16)
    │   └── js/
    │       ├── navigation_engine.js     # 60 FPS sub-frame animation, EKF Riccati covariance, Web Audio
    │       └── corridor_data.js         # Pre-baked 1300-frame telemetry with ablation states
    │
    ├── scripts/                         # Utilities & Field Workflows
    │   ├── download_offline_assets.py   # Bundles offline map tiles
    │   ├── fetch_road_geometry.py       # Queries Overpass API for road centerlines
    │   └── logger_guide.md              # Protocol for recording phone IMU logs on bikes
    │
    └── docs/
        └── final_screening_submission.md# Technical specification & hackathon proposal
```

---

## 🔬 Core Edge-AI Innovation & Mathematics

### 1. Why Classical INS Fails on Motorcycles (The Naive Drift Problem)
Classical strapdown Inertial Navigation Systems compute displacement $s(t)$ by double integration of accelerometer readings:
$$s(t) = \iint a(t) \, dt^2$$

On a two-wheeler handlebar, this fails within seconds because:
1. **Engine Harmonics:** Single-cylinder 4-stroke engines (1500–4000 RPM) vibrate aggressively in the **25–50 Hz band** ($2.0\text{--}4.5\text{ m/s}^2$ amplitude). Classical double integration treats these cyclic forces as sustained acceleration, causing immediate quadratic displacement runaway.
2. **Road Roughness:** Potholes and expansion joints inject Poisson shock impulses.
3. **MEMS Sensor Bias:** Budget phone gyroscopes drift at $0.1\text{--}0.3^\circ/\text{s}$. A $5^\circ$ heading error projects forward speed into false lateral velocity, sending the vehicle through tunnel walls into buildings.

### 2. The TrueTrack Solution: 1D Temporal Convolutional Network
TrueTrack replaces acceleration integration with direct regression of velocity and angular rates:
$$(\hat{v}_{\text{fwd}}, \hat{\omega}_{\text{yaw}}) = f_{\boldsymbol{\theta}}\left(\mathbf{X}_{\text{rolling}}\right)$$

* **Input Tensor:** Rolling 1.0-second temporal window at 50 Hz: Shape `[Batch=1, Channels=6, SeqLen=50]` consisting of:
  $$[a_x, a_y, a_z, \omega_x, \omega_y, \omega_z]$$
* **Per-Channel Z-Score Normalization:**
  $$\tilde{x}_i = \frac{x_i - \mu_i}{\sigma_i}$$
  Prevents the large magnitude of gravity ($9.8\text{ m/s}^2$) from drowning out subtle gyroscopic yaw rates ($0.05\text{ rad/s}$).
* **Learned Harmonic Rejection:** The dilated 1D temporal convolutions act as an adaptive spatial filter, attenuating 35 Hz engine vibration by **-8.3 dB** without the phase delay of classical low-pass filters.
* **Model Footprint:** 26,114 parameters, **103.7 KB (ONNX)**, INT8 quantized for Snapdragon Hexagon NPU with **1.4 ms inference latency** and **< 0.04W power draw**.

### 3. Lean Angle & Non-Holonomic Constraint (NHC)
When a two-wheeler corners, it rolls by angle $\phi$. Centripetal acceleration mixes into the vertical and lateral accelerometer axes:
$$\phi = \arctan\left(\frac{v \cdot \omega}{g}\right)$$
`LeanCorrector.kt` derolls the acceleration vector so the non-holonomic constraint ($v_{\text{lateral}} \approx 0$) holds true in the vehicle body frame:
$$\begin{bmatrix} a_{x,\text{derolled}} \\ a_{y,\text{derolled}} \\ a_{z,\text{derolled}} \end{bmatrix} = \mathbf{R}_x(-\phi) \begin{bmatrix} a_x \\ a_y \\ a_z \end{bmatrix}$$

### 4. Extended Kalman Filter (EKF) with Road Manifold Constraint
* Propagates state $\mathbf{x} = [p_x, p_y, v, \theta]^T$ and Riccati covariance $P_{k|k} = (I - K_k H_k) P_{k|k-1}$ at 50 Hz.
* Projects the estimated coordinate onto the nearest vector segment of the offline OpenStreetMap road centerline.
* Enforces Indian Road Congress **IRC:86** lane half-width constraint ($\text{cross-track deviation } \le 1.8\text{ m}$).

### 5. Sigmoidal GPS Reconciliation (Zero Visual Teleportation)
When GPS satellite lock returns after an underpass, standard navigation apps instantly "snap/jump" to the satellite fix, causing jarring disorientation. TrueTrack reconciles the neural trajectory with the satellite fix via a smooth S-curve blend over a 3.0-second window ($\tau = 3.0\text{s}$):
$$\alpha(t) = \frac{1}{1 + e^{-k(t - t_0 - \tau/2)}}$$
$$\mathbf{p}_{\text{display}}(t) = (1 - \alpha(t)) \mathbf{p}_{\text{DR}}(t) + \alpha(t) \mathbf{p}_{\text{GPS}}(t)$$

---

## 📊 Ablation Benchmark Results (45.0s Continuous Blackout)

Tested over a 45-second total GPS blackout in a curved tunnel with single-cylinder engine vibration (25–45 Hz) and MEMS gyro drift:

| Strategy | Max Total Error | End Total Error | End Along-Track ($e_\parallel$) | End Cross-Track ($e_\perp$) | RMSE | Drift Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Naive Double Integration (Classical INS)** | 114.00 m | 114.00 m | 24.74 m | 111.28 m | 49.30 m | 2.533 m/s |
| **2. Map Manifold Alone (Classical + Map)** | 40.29 m | 24.80 m | 24.74 m | 1.80 m | 16.24 m | 0.551 m/s |
| **3. Neural Velocity Alone (1D-CNN, No Map)** | 52.40 m | 52.40 m | 5.64 m | 52.09 m | 28.94 m | 1.164 m/s |
| **4. TrueTrack Full Stack (Neural + Map + EKF)** | **0.95 m** | **0.37 m** | **0.12 m** | **0.34 m** | **0.52 m** | **0.008 m/s** |

* **Monte Carlo Multi-Seed (5 seeds):** Median drift 4.34 m, 95th-percentile drift 9.26 m.
* **Map Perturbation Stress Test (2.0m offset):** Max error bounded at 2.61 m.
* **Engine Harmonic Attenuation:** -8.3 dB at 35 Hz.

---

## 📱 Native Android App (`android_app/`) Deep-Dive

### Architecture & Tech Stack
* **Language:** Kotlin
* **Target OS:** Android 14 (API 34), Min SDK 26
* **Target Device:** iQOO Performance Flagships (Snapdragon 8 Gen 3 / Gen 2 / 7+ Gen 3)
* **ML Inference Runtime:** `com.microsoft.onnxruntime:onnxruntime-android:1.17.0` configured with **NNAPI execution provider** to target Qualcomm Snapdragon Hexagon NPU.
* **Map Engine:** `org.osmdroid:osmdroid-android:6.1.18` (100% offline, zero Google Maps API keys required). Dark matrix color filter applied for high-contrast tactical cockpit aesthetics.
* **Networking / Telemetry:** Embedded `TelemetryStreamServer` (WebSocket on port 8765) broadcasting live sensor data to external dashboards.
* **Audio Guidance:** `AudioCueManager` using Android `TextToSpeech` and `ToneGenerator`.

### Available Routes & Datasets in the App
The Android app features a multi-route selection dialog accessible by tapping the top route button (`btnDemoMode`):
1. **🎬 Rohini Theatre Koyambedu (Chennai - Real iQOO Data):**
   - Source: Real ride recorded in Chennai (`Rohini_Theatre_Koyambedu-2026-09-25_13-05-40`).
   - Duration: 52 seconds (520 frames at 10 Hz).
   - Blackout: Simulated tunnel blackout from $t=15\text{s}$ to $40\text{s}$ (25s blackout).
   - File: `android_app/app/src/main/assets/rohini_telemetry.json`
2. **🛣️ Varadarajapuram Route (Chennai - Real iQOO Data):**
   - Source: Real ride recorded in Chennai (`Varadarajapuram-2026-09-25_11-12-54`).
   - Duration: 100 seconds (1000 frames at 10 Hz).
   - Blackout: From $t=25\text{s}$ to $65\text{s}$ (40s blackout).
   - File: `android_app/app/src/main/assets/varadarajapuram_telemetry.json`
3. **📍 45/46 Corridor (Chennai - Real iQOO Data):**
   - Source: Real highway drive recorded in Chennai (`45_46-2026-09-25_10-54-49`).
   - Duration: 100 seconds (1000 frames at 10 Hz).
   - Blackout: From $t=25\text{s}$ to $65\text{s}$ (40s blackout).
   - File: `android_app/app/src/main/assets/45_46_telemetry.json`
4. **🏢 HITEC City Underpass (Hyderabad - Corridor Sim):**
   - Source: Physics simulation through the Mindspace Underpass (`corridor_telemetry.json`).
   - Duration: 130 seconds (1300 frames at 10 Hz).
   - Blackout: $t=40\text{s}$ to $85\text{s}$ (45s blackout).
5. **📱 Live Sensors (Real Phone IMU Mode):**
   - Reads physical hardware 50 Hz accelerometer + gyroscope via `SensorManager`.
   - Feeds rolling 50-sample buffer into `ortSession` for live real-time inference on the desk or bike.

### Audio Guidance & Voice System (`AudioCueManager.kt`)
* Restored and fully functional.
* Uses `TextToSpeech` with queue flushing, English/US fallback, and utterance callbacks.
* Acoustic alert: `ToneGenerator.TONE_PROP_BEEP2` on blackout entrance, `TONE_PROP_ACK` on GPS restore.
* Spoken prompts:
  * *"TrueTrack navigation active. Route: [Route Name]."*
  * *"GPS blackout active. Neural dead reckoning engaged."*
  * *"GPS locked. Continuous trajectory reconciled."*
  * *"Voice guidance enabled."*
  * *"Switched to live phone sensor mode."*
* Only announces blackout state changes on edge transitions (prevents audio spam).

### Standstill Lock (Zero-Velocity Update / ZUPT)
* Prevents false creep when the bike or phone is stationary at a traffic signal.
* Detects $||\mathbf{a}|| \approx 9.81\text{ m/s}^2$ and $||\boldsymbol{\omega}|| < 0.05\text{ rad/s}$, holding speed strictly at `0.0 km/h` and freezing dead reckoning integration.

---

## 🌐 Web Simulation Cockpit (`web_app/`) Deep-Dive

### Architecture & Visual Design
* **Engine:** Bundled MapLibre GL WebGL engine (`lib/maplibre-gl.js`, 100% offline, zero CDN calls).
* **Map Center:** Hyderabad HITEC City / Mindspace Underpass (`[78.3815, 17.4475]`).
* **Vector Manifold:** Real OpenStreetMap highway geometry (`data/hitec_corridor.geojson`).
* **Visual Hierarchy:**
  * **TrueTrack Hero Puck:** Solid 40px blue puck (`#38bdf8`) with white border and pulse ring (`z-index: 800`).
  * **Legacy Failure Puck:** 28px hollow red puck showing compound quadratic error runaway (`z-index: 700`).
  * **EKF Covariance Ellipse:** Dynamic polygon updated each frame from algebraic Riccati $P$ matrix eigenvalues.
  * **Tactical Readouts:** 56px tabular numerals with count-up tweening and divergence ratio chip (`15x lower`).
  * **Log-Scale Error Chart:** Precision Canvas chart ($0.1\text{ m} \to 200\text{ m}$) with interactive mouse hover.
  * **Pipeline Lightbar:** Real-time 5-stage status (`IMU 50Hz` $\to$ `1D-CNN` $\to$ `EKF P` $\to$ `OSM Snap` $\to$ `Blend`).
  * **Dual-Screen Phone Bridge:** WebSocket client listening to `ws://localhost:8765` for mirroring live telemetry from physical iQOO phones.

---

## 🛠️ Issues Investigated and Solved in Previous Turns

1. **Gemini Hallucinations & Git Commit Spam:**
   - Previous automatic tool loops were creating excessive broken commits and incorrect code patches.
   - Pushed clean, working, validated code to branch `feature/android-app-setup`.
2. **Missing Voice Guidance / TTS Silence:**
   - TTS initialization was asynchronous; utterances sent during startup were lost before `TextToSpeech` reached `SUCCESS`.
   - Fixed by adding a synchronized `pendingUtterances` queue, proper locale fallbacks (`Locale.US` $\to$ `Locale.getDefault()` $\to$ `Locale.ENGLISH`), and triggering speech only on edge transitions.
3. **Vehicle Freeze / Speed Going to 0:**
   - Caused by an issue where the app was left in Live Sensor mode while sitting stationary on a desk (ZUPT standstill lock kicked in correctly for stationary phones, but user expected movement).
   - Fixed by making **Simulation Mode the default**, providing 4 rich route datasets, and adding a clean Route Selection dialog.
4. **Offline Map Tiles Loading Without API Keys:**
   - Mapbox / Google Maps require online keys that fail underground.
   - Replaced with local Osmdroid (Android) and MapLibre (Web) preloading bundled raster tiles from `assets/tiles/` and `web_app/tiles/`.
5. **Real iQOO Recorded Telemetry Integration:**
   - Sensor Logger CSVs (`Rohini_Theatre_Koyambedu`, `Varadarajapuram`, `45_46`) contained multi-rate asynchronous timestamps.
   - Built a Python interpolation engine (`convert_all_real_datasets.py`) mapping GPS, accelerometer, and gyroscope streams onto a synchronous 10 Hz timebase, generating ground truth, realistic naive dead reckoning drift, and neural-bounded trajectories.

---

## 🚀 How to Run, Test, and Build Every Component

### 1. Web Simulation Cockpit
Run directly from terminal using Python's built-in HTTP server:
```bash
cd "c:\Jala deee\IQOO\TrueTrack"
python -m http.server 8000 --directory web_app
```
Open browser at: `http://localhost:8000`

### 2. Machine Learning Engine (Benchmarks & Re-training)
```bash
cd "c:\Jala deee\IQOO\TrueTrack"
# Activate virtual environment if present, or install dependencies:
pip install torch numpy pandas scipy onnx onnxruntime

# Run 4-way ablation benchmark
python ml_engine/benchmark_drift.py

# Run extended benchmarks (varying speeds & durations)
python ml_engine/generate_extended_benchmarks.py

# Re-train 1D-CNN and re-export ONNX / TorchScript
python ml_engine/train_and_export_model.py
```

### 3. Android Application
1. Open `c:\Jala deee\IQOO\TrueTrack\android_app` in **Android Studio**.
2. Connect your physical **iQOO phone** via USB (Enable Developer Options & USB Debugging).
3. Run or assemble the app:
   ```bash
   cd "c:\Jala deee\IQOO\TrueTrack\android_app"
   .\gradlew assembleDebug
   ```
4. Install APK onto the phone:
   ```bash
   adb install -r app/build/outputs/apk/debug/app-debug.apk
   ```
5. To test phone-to-cockpit live bridging:
   ```bash
   adb forward tcp:8765 tcp:8765
   ```
   Now the Web Cockpit at `http://localhost:8000` will connect to `ws://localhost:8765` and mirror the phone's live IMU and navigation data.

---

## 💡 Guidance for Future AI Agents Assisting on this Project

1. **Do NOT delete or break the real dataset telemetry files:**
   - `rohini_telemetry.json`, `varadarajapuram_telemetry.json`, `45_46_telemetry.json`, `corridor_telemetry.json` in `android_app/app/src/main/assets/`.
2. **Preserve Offline Capabilities:**
   - Under no circumstances add online-only dependencies (e.g., Google Maps API keys, Mapbox tokens, online cloud inference). The entire value proposition of TrueTrack is **100% offline, zero-cloud, on-device edge intelligence**.
3. **Respect the Target Hardware Specs:**
   - Model must remain ultra-lightweight (< 200 KB ONNX) and compatible with Qualcomm Hexagon NPU (INT8 quantization, standard Conv1D / BatchNorm / LeakyReLU operations).
4. **Git Branching:**
   - Current active branch is `feature/android-app-setup`. All changes should be clean, tested, and never committed without explicit user confirmation.
5. **Audio & Voice Guidelines:**
   - The user values the voice alerts ("GPS blackout active. Neural dead reckoning engaged", "GPS locked. Continuous trajectory reconciled"). Ensure `AudioCueManager` remains enabled by default and operates on edge triggers.
