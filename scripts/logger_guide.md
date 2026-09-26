# Real phone sensor log ingest

`ml_engine/load_real_imu_data.py` resamples Sensor Logger folders for inspection. It does not train the checked-in model or run the drift benchmark.

## Capture and axis notes

Export `Accelerometer.csv` and `Gyroscope.csv`; `Location.csv` and `Orientation.csv` are optional. The importer expects a `seconds_elapsed` column and `x/y/z` sensor columns. Location data is expected to use `latitude`, `longitude`, and optional `speed` and `accuracy`; orientation data may include `roll` in radians.

The checked-in model was trained on one synthetic route, not on the real phone logs. The importer keeps the raw device axes and does not estimate a complete phone-to-vehicle rotation. Its optional roll de-roll output assumes the app's declared frame (X=lateral, Y=forward, Z=vertical), so verify mounting and units before using the derived channels as model input.

## Resample a log

```bash
python ml_engine/load_real_imu_data.py --input path/to/SensorLoggerExport --output ml_engine/real_ride_resampled.csv
```

The importer interpolates accelerometer and gyroscope samples onto a 50 Hz time grid. This is resampling; it cannot recover information absent from a lower-rate recording. GNSS is interpolated only across intervals up to 2.5 seconds, with longer outages left as `NaN`.

To label an interval as a simulated blackout and remove GNSS values from that interval, pass `--blackout-start SECONDS` and optionally `--blackout-duration SECONDS`. This is a user-specified mask, not an automatically detected tunnel or proof of a real GNSS outage.

## What the output means

The CSV is synchronized sensor and GNSS input for inspection. `benchmark_drift.py` and the browser benchmark still use synthetic telemetry and ground-truth-assisted map projection; they do not accept this CSV as an evaluation dataset. No real-drive drift score, held-out route result, or lean-correction benefit can be computed from this importer alone.
