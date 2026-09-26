# TrueTrack prototype

TrueTrack is a two-wheeler GPS-denied navigation prototype with an Android app, a browser cockpit, synthetic telemetry, and several real phone sensor-log captures. The current repository is an engineering demo, not a validated navigation product.

## Current implementation status

- **Android sensors:** accelerometer and gyroscope listeners target 50 Hz. A rolling `[1, 6, 50]` window is sent to ONNX Runtime about every 100 ms after the first full window.
- **Model runtime:** the app requests ONNX Runtime's Android NNAPI execution provider and reports the elapsed time around `OrtSession.run`. The selected provider is not confirmed in the app; this is not evidence of Qualcomm QNN/Hexagon execution. The checked-in model is FP32 ONNX, not a QNN-compiled INT8 artifact.
- **Live positioning:** Android accepts fine GPS fixes or uses the network provider when only coarse permission is granted. After a simulated blackout, it gates three consistent fixes. In live mode it integrates model speed and yaw between location fixes; it does not run a live EKF or road-map snap.
- **Lean correction:** the Android `LeanCorrector` estimates roll and can de-roll acceleration, but the toggle defaults off because the checked-in model and normalization stats were trained on uncorrected channels. It does not fuse a non-holonomic constraint (NHC) into a position filter.
- **Route deviation:** the Android warning compares the current position with one hardcoded HITEC City corridor. It is a warning stub, not a multi-route map matcher.
- **Foreground service:** a notification and heartbeat keep the service visible. Sensor acquisition and navigation state still belong to `MainActivity`; the service is not an independent navigation engine.
- **Phone bridge:** the WebSocket server binds to loopback and streams telemetry at the inference cadence (about 10 Hz), with `adb reverse tcp:8765 tcp:8765`. The bridge is telemetry-only; it does not accept GPS-control commands.
- **Browser cockpit:** its playback, benchmark table, IMU traces, and model-output traces use precomputed synthetic telemetry. The resolution slider changes the drawing and adds display-only noise; it does not rerun inference.
- **Map:** the route GeoJSON is local. The web map requests OpenStreetMap raster tiles as they are viewed, so its basemap needs internet. The Android map also uses the online Mapnik tile source. The repository still contains historic raster/debug images that the current map code no longer reads; their removal is pending review.

## Model and benchmark limits

The checked-in model was trained on a single synthetic drive. The former trainer split overlapping windows randomly, so its stored `0.0536` validation MSE is leakage-prone. The training script now uses a chronological holdout, a one-second gap, and training-only normalization, but the weights and normalization files have **not** been regenerated with that change. The Python runtime available in this workspace does not include PyTorch.

The benchmark scripts and browser telemetry also use synthetic inputs. In blackout sections, map/full-stack progression and projection use per-frame ground-truth position and heading. The five-seed output perturbs the same synthetic route; it is not held-out route validation. The cornering metrics JSON has no generator script in the repository. Treat these files as illustrative diagnostics only, not evidence of drift performance, generalization, or real-drive accuracy.

The live Android latency badge is an ONNX Runtime call measurement. It does not establish an NPU backend or a device-independent latency result. No target-phone NPU profile, Doze run, disconnect run, or end-to-end physical navigation evaluation is recorded here.

## Repository layout

```text
android_app/       Kotlin Android prototype, ONNX model asset, and UI
ml_engine/         synthetic generator, trainer, benchmark scripts, and artifacts
scripts/           OSM geometry/data utilities and logger guide
web_app/           browser cockpit, local route GeoJSON, and vendored MapLibre
docs/              install notes and screening draft
*/*-2026-*/        real phone sensor-log CSV captures (not used as model training data)
```

## Run the browser cockpit

From the repository root:

```bash
python -m http.server 8000 --directory web_app
```

Open `http://localhost:8000`. The app code and telemetry are local, but the OSM basemap loads viewed tiles over the network. The Android phone bridge can be used over USB with:

```bash
adb reverse tcp:8765 tcp:8765
```

## Python workflow

The scripts use NumPy, pandas, SciPy, PyTorch, and ONNX export dependencies. The model trainer reads `ml_engine/synthetic_telemetry_50hz.csv`; it does not train on the checked-in real phone logs. `load_real_imu_data.py` is a separate log-ingest utility and its output is not connected to model training or benchmark evaluation.

```bash
python ml_engine/train_and_export_model.py
python ml_engine/benchmark_drift.py
python ml_engine/evaluate_multiseed_distribution.py
```

The benchmark commands reproduce synthetic, ground-truth-assisted diagnostics described above. Their outputs should not be presented as leak-free or real-world results.

## Android app

Open `android_app/` in Android Studio and build/install the app there. Grant location permission for live fixes. The app begins in simulation mode; switch to live sensor mode to use phone sensors and ONNX Runtime. The map background requires network access. A device build and target-phone runtime profile have not been verified as part of this repository review.

## Map and third-party notices

The browser runtime is MapLibre GL JS 4.7.1; its license is in [`web_app/lib/LICENSE.txt`](web_app/lib/LICENSE.txt). Map data and raster tiles are served by OpenStreetMap with visible attribution. The asset downloader fetches only the pinned MapLibre runtime and does not prefetch OSM tiles.
