# TrueTrack: Complete Repository & Workspace Inventory

> **Project Name:** TrueTrack &ndash; On-Device Neural-Inertial Navigation for GPS-Denied Urban Corridors  
> **Target Hardware:** Snapdragon Mobile Platform (Hexagon NPU Runtime via QNN / TFLite)  
> **Primary Use-Case:** Autonomous offline navigation for two-wheelers in underpasses, tunnels, and urban canyons  
> **Workspace Path:** `c:\Jala deee\IQOO\TrueTrack`

---

## 📁 Repository Directory Map

```
TrueTrack/
├── README.md                           # Main Project Overview & Benchmark Report
├── PROJECT_OVERVIEW.md                 # Detailed Directory & Asset Inventory (This File)
│
├── android_app/                        # Native Android Application (Kotlin)
│   ├── build.gradle                    # Project-level Gradle configuration
│   ├── settings.gradle                 # Module settings
│   └── app/
│       ├── build.gradle                # App-level Gradle dependencies & ONNX Runtime settings
│       └── src/main/
│           ├── AndroidManifest.xml     # Application manifest & permissions
│           ├── assets/                 # Edge AI model assets & specs
│           │   ├── norm_stats.json     # Per-channel Z-score normalization parameters
│           │   ├── npu_model_spec.json # Hexagon NPU hardware deployment specs
│           │   └── truetrack_model.onnx# 103.7 KB INT8 ONNX Neural Model
│           ├── java/com/truetrack/navigation/
│           │   ├── MainActivity.kt     # App entry, sensor listeners & UI bindings
│           │   ├── AudioCueManager.kt  # Text-to-Speech audio navigation guidance
│           │   ├── LeanCorrector.kt    # Roll dynamics & lean-angle estimator
│           │   └── TelemetryStreamServer.kt # 50Hz WebSocket/HTTP sensor server
│           └── res/                    # UI layouts, colors, strings & themes
│               ├── layout/activity_main.xml
│               └── values/{colors,strings,themes}.xml
│
├── ml_engine/                          # Neural-Inertial ML Engine & Benchmark Suite
│   ├── train_and_export_model.py       # PyTorch 1D-CNN training & ONNX/TorchScript exporter
│   ├── benchmark_drift.py              # 45-second GPS blackout benchmark (4 strategies)
│   ├── generate_extended_benchmarks.py # Multi-speed, multi-duration extended benchmarks
│   ├── evaluate_multiseed_distribution.py # Stochastic noise & seed evaluation
│   ├── lean_estimator.py               # Physics-based centripetal force & lean estimator
│   ├── lean_correction.py              # Roll dynamics correction logic
│   ├── synthetic_data_generator.py     # 50Hz IMU dataset generator with engine harmonic noise
│   ├── load_real_imu_data.py           # Smartphone sensor log loader & preprocessor
│   ├── fetch_osm.py                    # OpenStreetMap road graph fetcher
│   ├── inspect_highways.py             # Highway tag and element inspector
│   ├── truetrack_1dcnn.pth             # Trained PyTorch model weights
│   ├── truetrack_model.onnx            # Exported ONNX model file (103.7 KB)
│   ├── truetrack_model.torchscript     # Exported TorchScript model file
│   ├── norm_stats.json                 # Z-score normalization statistics
│   ├── npu_model_spec.json             # NPU target performance metrics
│   ├── benchmark_results.json          # Benchmark evaluation outputs
│   ├── benchmark_cornering_results.json# Cornering & lean evaluation metrics
│   ├── multiseed_evaluation.json       # Multi-seed stochastic evaluation results
│   ├── synthetic_telemetry_50hz.csv    # Synthetic 50Hz IMU sensor telemetry
│   ├── synthetic_cornering_50hz.csv    # Synthetic 50Hz IMU cornering telemetry
│   ├── hitec_road_nominatim.geojson    # GeoJSON road corridor geometry
│   └── osm_bbox_data.json              # OSM bounding box road data
│
├── web_app/                            # Interactive Web Simulation Cockpit
│   ├── index.html                      # Web cockpit UI (Leaflet map + Telemetry charts)
│   ├── styles.css                      # Modern dark-mode & glassmorphism styling
│   ├── DESIGN.md                       # Design system & visual specifications
│   ├── js/
│   │   ├── navigation_engine.js        # Client-side EKF, neural simulation & map snapping
│   │   └── corridor_data.js            # Underpass node coordinates & blackout zones
│   ├── data/
│   │   └── hitec_corridor.geojson      # Road manifold vector data
│   └── tiles/                          # Offline raster map tiles & debug images
│       ├── 14/, 15/, 16/               # Tile hierarchy for offline map rendering
│       ├── full_corridor_stitched.png  # Stitched overview image
│       └── verified_road_alignment.png # Alignment verification image
│
├── scripts/                            # Data Utility Scripts & Field Logging Guides
│   ├── fetch_road_geometry.py          # Overpass API road element extractor
│   ├── download_offline_assets.py      # Offline map tile & Leaflet asset bundler
│   └── logger_guide.md                 # Real-world IMU sensor logging instructions
│
└── docs/                               # Project Submission & Technical Papers
    └── final_screening_submission.md   # Hackathon screening submission report
```

---

## 🔍 Module-by-Module Overview

### 1. 📱 Native Android Application (`android_app/`)
* **Purpose:** Runs local neural-inertial dead reckoning directly on the smartphone, interfacing with device IMU sensors and target Snapdragon Hexagon NPU.
* **Key Components:**
  * [`MainActivity.kt`](file:///c:/Jala%20deee/IQOO/TrueTrack/android_app/app/src/main/java/com/truetrack/navigation/MainActivity.kt): Initializes 50Hz `SensorEventListener` for 3-axis Accelerometer + 3-axis Gyroscope, streams telemetry to NPU model buffer.
  * [`AudioCueManager.kt`](file:///c:/Jala%20deee/IQOO/TrueTrack/android_app/app/src/main/java/com/truetrack/navigation/AudioCueManager.kt): Provides real-time Text-To-Speech (TTS) voice alerts during GPS blackouts (e.g., *"Tunnel blackout active. Approaching split in 100 meters"*).
  * [`LeanCorrector.kt`](file:///c:/Jala%20deee/IQOO/TrueTrack/android_app/app/src/main/java/com/truetrack/navigation/LeanCorrector.kt): Estimates motorcycle lean angle ($\theta_{\text{lean}} = \arctan(v \cdot \omega / g)$) to disentangle gravity components during turns.
  * [`TelemetryStreamServer.kt`](file:///c:/Jala%20deee/IQOO/TrueTrack/android_app/app/src/main/java/com/truetrack/navigation/TelemetryStreamServer.kt): Embedded HTTP/WebSocket server streaming live IMU readings at 50Hz to external visualization cockpits.
  * **On-Device Assets:**
    * [`truetrack_model.onnx`](file:///c:/Jala%20deee/IQOO/TrueTrack/android_app/app/src/main/assets/truetrack_model.onnx): 103.7 KB INT8 quantized neural network.
    * [`norm_stats.json`](file:///c:/Jala%20deee/IQOO/TrueTrack/android_app/app/src/main/assets/norm_stats.json): Per-channel normalization parameters ($\mu, \sigma$).
    * [`npu_model_spec.json`](file:///c:/Jala%20deee/IQOO/TrueTrack/android_app/app/src/main/assets/npu_model_spec.json): Benchmarking specifications for Hexagon NPU execution (1.4 ms latency, 0.04W power draw).

---

### 2. 🧠 Machine Learning Engine (`ml_engine/`)
* **Purpose:** Contains model architectures, synthetic IMU data generators, training routines, and benchmark evaluation suites.
* **Key Components:**
  * [`train_and_export_model.py`](file:///c:/Jala%20deee/IQOO/TrueTrack/ml_engine/train_and_export_model.py): PyTorch 1D-CNN temporal convolutional network training script. Direct regressor for forward velocity ($\hat{v}$) and yaw rate ($\hat{\omega}$) from 6-channel 1-second rolling window (`[Batch, 6, 50]`). Exports PyTorch `.pth`, TorchScript `.torchscript`, and ONNX `.onnx`.
  * [`benchmark_drift.py`](file:///c:/Jala%20deee/IQOO/TrueTrack/ml_engine/benchmark_drift.py): Primary benchmarking script evaluating 4 navigation strategies over a 45-second GPS blackout:
    1. Naive Double Integration (Classical INS)
    2. Map Manifold Alone
    3. Neural Velocity Alone
    4. **TrueTrack Full Stack (Neural + Map + EKF)**
  * [`generate_extended_benchmarks.py`](file:///c:/Jala%20deee/IQOO/TrueTrack/ml_engine/generate_extended_benchmarks.py): Tests performance across varying speed profiles (30 km/h, 50 km/h, 70 km/h) and blackout durations (15s to 90s).
  * [`evaluate_multiseed_distribution.py`](file:///c:/Jala%20deee/IQOO/TrueTrack/ml_engine/evaluate_multiseed_distribution.py): Evaluates stochastic robustness across 100 random seeds for engine vibration harmonics (25–50 Hz) and MEMS gyro random walk.
  * [`synthetic_data_generator.py`](file:///c:/Jala%20deee/IQOO/TrueTrack/ml_engine/synthetic_data_generator.py): Realistic 50Hz sensor simulator injecting single-cylinder motorcycle engine vibration, ISO 8608 road roughness shocks, and sensor bias drift.
  * [`lean_estimator.py`](file:///c:/Jala%20deee/IQOO/TrueTrack/ml_engine/lean_estimator.py) & [`lean_correction.py`](file:///c:/Jala%20deee/IQOO/TrueTrack/ml_engine/lean_correction.py): Roll dynamics modeling to correct centripetal acceleration vector projection.

---

### 3. 🌐 Web Simulation Cockpit (`web_app/`)
* **Purpose:** Interactive browser dashboard for visualizing real-time navigation during simulated GPS blackouts.
* **Key Components:**
  * [`index.html`](file:///c:/Jala%20deee/IQOO/TrueTrack/web_app/index.html): Main dashboard interface featuring side-by-side strategy trajectory tracking, live telemetry charts (speed, heading, error drift), and manual blackout triggers.
  * [`js/navigation_engine.js`](file:///c:/Jala%20deee/IQOO/TrueTrack/web_app/js/navigation_engine.js): Full client-side JavaScript implementation of the EKF, 1D-CNN simulator, map manifold constraint, and S-curve Sigmoidal GPS reconciliation.
  * [`js/corridor_data.js`](file:///c:/Jala%20deee/IQOO/TrueTrack/web_app/js/corridor_data.js): Pre-loaded waypoints and corridor bounding boxes for Hyderabad HITEC City / Mindspace Underpass.
  * [`DESIGN.md`](file:///c:/Jala%20deee/IQOO/TrueTrack/web_app/DESIGN.md): Visual design specifications and UI system document.
  * [`tiles/`](file:///c:/Jala%20deee/IQOO/TrueTrack/web_app/tiles/): Offline map raster tiles (Zoom levels 14, 15, 16) ensuring the cockpit operates with zero internet connection.

---

### 4. 🛠️ Utilities & Field Testing (`scripts/`)
* **Key Components:**
  * [`download_offline_assets.py`](file:///c:/Jala%20deee/IQOO/TrueTrack/scripts/download_offline_assets.py): Python script to download local Leaflet dependencies and OSM map tiles for offline bundling.
  * [`fetch_road_geometry.py`](file:///c:/Jala%20deee/IQOO/TrueTrack/scripts/fetch_road_geometry.py): Overpass API helper to query vector road coordinates for target underpasses.
  * [`logger_guide.md`](file:///c:/Jala%20deee/IQOO/TrueTrack/scripts/logger_guide.md): Field guide explaining how to log raw 50Hz accelerometer and gyroscope data from an Android smartphone using Sensor Logger.

---

### 5. 📄 Documentation (`docs/` & `README.md`)
* [`README.md`](file:///c:/Jala%20deee/IQOO/TrueTrack/README.md): Comprehensive repository documentation, problem statement, edge AI innovation breakdown, performance tables, and deployment guide.
* [`docs/final_screening_submission.md`](file:///c:/Jala%20deee/IQOO/TrueTrack/docs/final_screening_submission.md): Official screening submission document detailing architectural decisions, Snapdragon NPU execution specs, ablation studies, and roadmap.

---

## 📊 Summary of Benchmark Results

Evaluated over a **45-second continuous GPS blackout** through a curved underpass corridor under single-cylinder engine vibration (25–45 Hz) and MEMS gyro random walk:

| Navigation Strategy | Max Error | End Error | RMSE | Drift Rate |
| :--- | :---: | :---: | :---: | :---: |
| **1. Naive Double Integration (Classical INS)** | 114.00 m | 114.00 m | 49.30 m | 2.533 m/s |
| **2. Map Manifold Alone (Classical + Map)** | 40.29 m | 24.80 m | 16.24 m | 0.551 m/s |
| **3. Neural Velocity Alone (1D-CNN, No Map)** | 52.40 m | 52.40 m | 28.94 m | 1.164 m/s |
| **4. TrueTrack Full Stack (Neural + Map + EKF)** | **0.95 m** | **0.37 m** | **0.52 m** | **0.008 m/s** |

---

## 🚀 How to Execute Key Tasks

### 1. Run the ML Benchmarks
```bash
python ml_engine/benchmark_drift.py
python ml_engine/generate_extended_benchmarks.py
```

### 2. Train or Re-Export the Neural Model
```bash
python ml_engine/train_and_export_model.py
```

### 3. Launch the Interactive Web Simulation Cockpit
Simply open [`web_app/index.html`](file:///c:/Jala%20deee/IQOO/TrueTrack/web_app/index.html) in any modern web browser or serve via Python:
```bash
python -m http.server 8000 --directory web_app
```

---
*Created automatically as a workspace inventory guide.*
