# TrueTrack: On-Device Neural-Inertial Navigation for GPS-Denied Urban Corridors

**Track:** Open Innovation (Local AI / Snapdragon NPU Focus)  
**Team:** 3 Members  
**Target Hardware:** Snapdragon Mobile Platform (Hexagon NPU Runtime via QNN / TFLite)

---

## 1. Executive Summary & Problem Statement
In India's dense metropolitan centers, satellite GPS coverage degrades or drops completely inside road underpasses, multi-level flyovers, tunnels, and dense urban canyons (e.g., Hyderabad Mindspace Underpass, Bangalore Cyber City corridors, Delhi Pragati Maidan tunnel).

Over 15 million gig-economy delivery riders (Swiggy, Zomato, Porter) and two-wheeler commuters navigate exclusively using consumer smartphones mounted on handlebars. Unlike passenger cars, two-wheelers lack OBD-II telemetry, wheel-speed odometry, or factory-installed Inertial Navigation Systems (INS). When satellite line-of-sight is lost:
1. Standard navigation apps freeze at the tunnel entrance, dead-reckon along an assumed straight line, or wander erratically into adjacent buildings.
2. Riders miss crucial underground bifurcations or exits, causing 10–20 minute detours, customer penalties, and unsafe sudden stops.

**TrueTrack** solves this through an autonomous on-device navigation engine. The moment satellite reception is lost, TrueTrack turns the smartphone's internal MEMS sensors (6-axis accelerometer + gyroscope) into an intelligent inertial navigation system, continuously tracks vehicular position along the road network, and reconciles smoothly when GPS returns—with zero external hardware and zero cloud dependency.

---

## 2. Core Innovation: Why Local Edge AI is Structurally Essential
In an underpass or tunnel blackout zone, cellular connectivity drops alongside GPS. A navigation engine cannot query an external cloud API or off-device model in a blackout zone. The intelligence must execute **strictly at the device edge**.

### The Edge-AI Breakthrough: Neural-Inertial Odometry
Classical strapdown inertial navigation relies on double-integrating accelerometer signals ($s = \iint a \, dt$). On a two-wheeler handlebar mount, this fails:
* Single-cylinder 4-stroke commuter engines operating between 1500–3000 RPM generate strong primary 1st-order rotational vibrations (crankshaft reciprocating imbalance at $\text{RPM}/60$) in the 25–50 Hz band ($2.0\text{--}4.0\text{ m/s}^2$ amplitude).
* Road roughness and surface irregularities inject continuous transient shock impulses (modeled via ISO 8608 road profiles).
* Double-integrating these raw signals causes positional error to explode quadratically ($t^2$) within seconds.

TrueTrack replaces naive double-integration with a **Normalized Neural-Inertial 1D-CNN (Temporal Convolutional Network)** tailored for the **Snapdragon Hexagon NPU**:
1. **Per-Channel Z-Score Normalization:** Sensor channels differ by orders of magnitude (gravity acceleration mean $\sim 9.8\text{ m/s}^2$ vs. gyro angular rate std $\sim 0.07\text{ rad/s}$). TrueTrack normalizes all 6 input streams, preventing high-amplitude acceleration gradients from drowning out subtle rotational cues. Normalization reduced velocity validation MSE from 0.1389 to 0.0257 (a **57% reduction in velocity RMSE**).
2. **Adaptive Harmonic Rejection:** The neural network processes a rolling 1-second buffer (50 samples at 50 Hz) of **full 6-axis IMU data** (`[Batch, 6, 50]`: $a_x, a_y, a_z, \omega_x, \omega_y, \omega_z$), acting as an adaptive non-linear filter damping 1st-order rotational vibration (-8.3 dB attenuation measured via empirical 35 Hz tone-injection test).
3. **Direct Velocity Vector Regression:** Instead of integrating acceleration twice, the model directly regresses instantaneous forward vehicle speed and yaw rate $(\hat{v}, \hat{\omega})$. This converts quadratic error divergence into a bounded, linear residual problem.
4. **Ultra-Low Edge Footprint:** The trained model footprint is **103.7 KB** in standard ONNX format (`truetrack_model.onnx`), designed for fast INT8 quantization via Qualcomm QNN tools to run in **~1.4 ms per inference** on the Hexagon NPU, maintaining a continuous 50 Hz navigation loop with negligible battery draw.

---

## 3. System Architecture & Fusion Pipeline

```
  [Phone 6-Axis MEMS IMU] ──► [50 Hz Rolling Buffer (50 samples x 6 channels)]
                                                 │
                                                 ▼
                                [Snapdragon NPU: 1D-CNN INT8]
                                (Z-Score Norm & Velocity Regression)
                                                 │
                                                 ▼ Predicted (v_fwd, yaw_rate)
  [Compass / Gyro] ─────────────► [Extended Kalman Filter (EKF)] ◄── [GPS Fix (when locked)]
                                                 │
                                                 ▼ Unconstrained Coordinates
                                [Offline OSM Road-Manifold Constraint]
                                (Heading Alignment & Lateral Bound)
                                                 │
                                                 ▼ Constrained Trajectory
                                [Sigmoid Reconciliation Engine]
                                (Smooth 3.0s blend on GPS return)
                                                 │
                                                 ▼
                                [Driver Navigation UI & Voice Cues]
```

### Key Algorithmic Components:
1. **Extended Kalman Filter (EKF):** Blends NPU-predicted velocity vectors with high-rate gyroscope integration to track position and heading covariances during clear-sky and blackout conditions.
2. **Offline Road-Manifold Constraint:** Vehicles cannot drive through tunnel concrete or fly laterally across walls. TrueTrack matches the estimated trajectory onto a locally cached OpenStreetMap vector graph (under 15 MB for an active metropolitan zone), aligning heading to the road geometry and capping cross-track deviation to physical roadway boundaries ($\le 1.8\text{ m}$, representing the standard single-lane half-width for Indian urban carriageways under IRC:86 guidelines).
3. **Smooth Sigmoid Reconciliation:** When satellite fix re-emerges, naive systems snap the marker with an abrupt, disorienting jump-cut. TrueTrack executes an S-curve blend over a 3.0-second convergence window:
   $$\alpha(t) = \frac{1}{1 + e^{-k(t - t_0 - \tau/2)}}$$
   $$P_{\text{display}}(t) = (1 - \alpha(t)) \cdot P_{\text{DR}}(t) + \alpha(t) \cdot P_{\text{GPS}}(t)$$
   The transition is mathematically continuous, eliminating jump-cuts while returning to true satellite coordinates.

---

## 4. Benchmark Validation & Component Ablation Study (Simulation-Based)

To evaluate the algorithmic architecture, we conducted a systematic 4-way ablation study over a **45.0-second total GPS blackout** inside a curved underpass simulation using full 6-axis IMU telemetry with per-channel Z-score normalization.

* **Sensor Noise Grounding:** Accelerometer and gyroscope channels reflect standard consumer MEMS sensor drift (bias random walk and ~0.2°/s gyro drift). Engine vibration reflects primary 1st-order rotational engine imbalance (25–50 Hz, $3.2\text{ m/s}^2$) and road roughness impulses (ISO 8608 Class C/D).
* **Doppler Velocity Initialization:** Baseline dead-reckoning filters initialize using standard Doppler velocity estimation ($\pm 0.15\text{ m/s}$ noise) at the blackout boundary.
* **Vector Decomposition:** Total error is decomposed into orthogonal components relative to the road centerline: Along-Track Longitudinal Error ($e_\parallel$) and Cross-Track Lateral Error ($e_\perp$), satisfying $e_{\text{total}} = \sqrt{e_\parallel^2 + e_\perp^2}$.
* **Reproducibility Note:** All metrics are evaluated under a fixed random seed (`seed=42`) for deterministic computational reproduction.

### 4-Way Component Ablation Results (45-Second Blackout Window, Seed=42):

| Strategy | Max Total Error | End Total Error | End Along-Track ($e_\parallel$) | End Cross-Track ($e_\perp$) | RMSE | Average Drift Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Naive Double Integration (No Neural, No Map)** | 114.00 m | 114.00 m | 24.74 m | 111.28 m | 49.30 m | 2.533 m/s |
| **2. Map Manifold Alone (Classical INS + Map)** | 40.29 m | 24.80 m | 24.74 m | 1.80 m | 16.24 m | 0.551 m/s |
| **3. Neural Velocity Alone (1D-CNN, No Map)** | 52.40 m | 52.40 m | 5.64 m | 52.09 m | 28.94 m | 1.164 m/s |
| **4. TrueTrack Full Stack (Neural + Map Manifold)** | **0.95 m** | **0.37 m** | **0.12 m** | **0.34 m** | **0.52 m** | **0.008 m/s** |

### Measured Component Breakdown & Physical Proof:
1. **Strategy 1 (Naive Classical INS):** Uncorrected gyro bias rotates the vehicle's heading vector by ~15°, projecting forward speed sideways. At blackout exit, the total 114.00 m error decomposes into **111.28 m of lateral cross-track divergence** and **24.74 m of along-track error** ($\sqrt{24.74^2 + 111.28^2} = 114.00\text{ m}$).
2. **Strategy 2 (Map Manifold Alone):** The road manifold clamps lateral wandering down to the single-lane boundary ($1.80\text{ m}$), but leaves the along-track longitudinal error ($24.74\text{ m}$ at exit, peaking at **40.25 m along-track** during deceleration) unmitigated ($\sqrt{24.74^2 + 1.80^2} = 24.80\text{ m}$).
3. **Strategy 3 (Neural Velocity Alone):** The 1D-CNN provides high-precision longitudinal odometry (only **5.64 m along-track error** over 45s), but without topological heading constraints, unconstrained gyro yaw integration drifts laterally into the terrain (**52.09 m cross-track**).
4. **Strategy 4 (TrueTrack Full Stack):** The multi-stage pipeline resolves both axes simultaneously: **Neural Velocity** bounds along-track error to **0.12 m**, while the **Road Manifold** bounds cross-track error to **0.34 m**, locking total end-of-blackout error to **0.37 m (RMSE 0.52 m)**.

> **Scope Limitation & Edge Topology:**  
> *This benchmark models a single continuous underpass corridor. For multi-branch underground tunnels with bifurcations, an offline Hidden Markov Model (HMM) evaluates emission and transition probabilities across candidate road branches prior to projection.*

---

## 5. Prototype Status & The 30-Hour On-Ground iQOO Build Roadmap

### Phase 0 Status: Pre-Event Algorithmic Validation
* **Benchmarking & Data Suite:** Full Python simulation pipeline modeling 2-wheeler 6-axis dynamics, MEMS random walk, and GPS tunnel blackouts (`synthetic_data_generator.py`, `benchmark_drift.py`), plus a smartphone sensor ingestion parser for real recordings (`load_real_imu_data.py`).
* **Exported Model Artifacts:** Full 6-axis normalized 1D-CNN velocity regression model trained and exported to standard ONNX format (`truetrack_model.onnx`, 103.7 KB) and TorchScript (`truetrack_model.torchscript`, 133.3 KB).
* **Interactive Navigation Console:** Complete web-based flight recorder cockpit (`web_app/index.html`, `navigation_engine.js`) providing real-time evaluation of Legacy GPS failure vs. TrueTrack continuous inertial navigation, real-time telemetry, EKF covariance ellipse, audio alerts, and manual GPS kill-switch.

### The 30-Hour On-Ground Build Plan on Physical iQOO Device (Snapdragon Hexagon NPU):

1. **Hours 00:00 – 04:00 | Qualcomm QNN INT8 Compilation:**
   - Convert `truetrack_model.onnx` using Qualcomm Neural Processing SDK (`qnn-onnx-converter`).
   - Quantize to INT8 using our 6,500-sample calibration dataset; compile into standalone binary `truetrack_htp.bin` targeting the Hexagon Tensor Processor (`libQnnHtp.so`).
   - Validate on-device execution latency $\le 1.4\text{ ms}$ on the physical iQOO phone via `qnn-net-run`.

2. **Hours 04:00 – 10:00 | Android Native C++ NDK Sensor Ingestion:**
   - Develop a high-priority C++ pthread via Android NDK (`ASensorManager`, `ASensorEventQueue`) polling calibrated 6-axis IMU data at 50 Hz (`SENSOR_DELAY_FASTEST`).
   - Maintain a 50-sample circular ring buffer in shared native memory with NEON SIMD Z-score normalization ($< 0.05\text{ ms}$).

3. **Hours 10:00 – 16:00 | On-Device EKF & Local Spatio-Temporal Road Manifold:**
   - Implement discrete-time EKF in C++ (Eigen / header-only matrix math) propagating Riccati covariance ($P$) at 50 Hz.
   - Package a local SQLite / FlatGeobuf vector extract ($< 15\text{ MB}$) of the target municipal corridor with R-Tree spatial indexing for sub-millisecond road queries ($< 0.2\text{ ms}$).
   - Enforce Indian Road Congress (IRC:86) single-lane lateral bounds ($\le 1.8\text{ m}$).

4. **Hours 16:00 – 22:00 | Native Android UI, Blackout Failsafe & Audio Synthesizer:**
   - Build a standalone Jetpack Compose / MapLibre Native Android app running at 60–120 FPS.
   - Implement smooth sigmoid reconciliation over a 3.0-second window upon GPS re-lock (`P_display = (1 - \alpha) * P_DR + \alpha * P_GPS`), eliminating jarring marker teleportation.
   - Native PCM AudioTrack sound generator emitting turn alerts and acoustic status cues.

5. **Hours 22:00 – 26:00 | Physical Vibration Test Rig & Power Profiling:**
   - Clamp the iQOO phone onto a mechanical vibration test rig replicating single-cylinder engine rumble (25–45 Hz).
   - Profile battery consumption and thermal dissipation using Snapdragon Profiler (target: $< 2.5\%$ battery drain per hour; thermal delta $< 2.5^\circ\text{C}$).

6. **Hours 26:00 – 30:00 | Live Stage Demonstration & Failover Test:**
   - Setup ultra-low latency USB-C DisplayPort / `scrcpy` 120 FPS screen mirroring for the jury.
   - **The On-Stage Proof:** Live demonstration toggling off Android Location/GNSS in real time while moving and rotating the iQOO phone on the demo table; demonstrate continuous, drift-bounded neural dead reckoning along the road network with sub-1.4 ms NPU inference and zero cloud connectivity.

---

## 6. Team Experience & Feasibility
* **Sensor Fusion & EKF Background:** Hands-on experience developing and tuning Extended Kalman Filter state estimation on Pixhawk flight controllers for GPS-denied UAV navigation.
* **Spatial & Routing Infrastructure:** Direct experience parsing OpenStreetMap vector networks and topology graphs from prior geospatial routing work.
* **Edge ML & Optimization:** Familiarity with PyTorch, ONNX export, INT8 quantization, and embedded inference workflows.

---

## 7. Real-World Impact
TrueTrack requires **zero hardware additions, zero OBD cables, and zero retrofit expense**—it works on any budget Android smartphone already sitting in an Indian delivery rider's phone mount. By bridging the critical blackout gap during underground navigation, TrueTrack prevents wrong-turn detours, protects delivery rider livelihoods, and establishes a new standard for smartphone-only vehicular dead reckoning.
