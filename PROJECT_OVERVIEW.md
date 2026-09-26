# Project overview

This is a prototype repository for two-wheeler dead-reckoning experiments. The Android app contains a live sensor/ONNX path; the web cockpit is primarily a precomputed synthetic replay. These paths should not be presented as one validated, end-to-end navigation system.

## Main components

| Area | Files | Current role |
|---|---|---|
| Android app | `android_app/app/src/main/java/com/truetrack/navigation/` | 50 Hz target sensor listener, lean de-rolling, ONNX Runtime inference, GPS gating, a single-corridor deviation warning, HUD, and loopback telemetry WebSocket. |
| Model workflow | `ml_engine/train_and_export_model.py`, `ml_engine/*.onnx`, `ml_engine/norm_stats.json` | Synthetic training and export of speed/yaw regression artifacts. Current checked-in weights predate the corrected chronological validation split. |
| Benchmark scripts | `ml_engine/benchmark_drift.py`, `ml_engine/generate_extended_benchmarks.py`, `ml_engine/evaluate_multiseed_distribution.py` | Synthetic diagnostics. Blackout map progression uses ground-truth position/heading; outputs are not leak-free performance measurements. |
| Real sensor captures | `45_46-*`, `Rohini_Theatre_Koyambedu-*`, `Varadarajapuram-*` | Phone logger CSV files. The model trainer does not use them. |
| Browser cockpit | `web_app/index.html`, `web_app/js/navigation_engine.js`, `web_app/styles.css` | Synthetic telemetry replay, plots, ablation controls, and optional phone telemetry display. Slider noise is display-only. |
| Corridor geometry | `web_app/data/hitec_corridor.geojson`, `ml_engine/hitec_road_nominatim.geojson` | OSM-derived corridor geometry. The Android warning still uses a hardcoded waypoint list. |
| Utilities | `scripts/` | Real-log analysis, road-geometry fetching, MapLibre runtime download, and field logging instructions. |

## Important limits

- ONNX Runtime requests NNAPI when available; the app does not verify the actual execution provider. QNN/Hexagon compilation, INT8 quantization, and device profiling are not checked in.
- The Android position update is a basic speed/yaw integration with GPS fixes. There is no live EKF or multi-road map matching.
- The browser uses precomputed synthetic trajectories. The benchmark map projection uses per-frame ground truth, and the five-seed script perturbs one route only.
- The browser and Android map basemaps request OSM tiles over the network as they are viewed. Historic raster/debug images remain tracked but are unused by current code.
- `NavigationForegroundService` provides a notification and heartbeat; sensor acquisition remains in the activity.
- The cornering benchmark JSON has no corresponding generator script in the repository.

See [`README.md`](README.md) for setup and [`docs/final_screening_submission.md`](docs/final_screening_submission.md) for the prototype status and evaluation caveats.
