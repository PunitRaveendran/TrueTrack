# Android prototype: build and run

This guide describes the current prototype. It does not claim a successful target-phone build or a background/Doze validation run.

## Build

Open `android_app/` in Android Studio with Android SDK 34 and the project Gradle wrapper. Build and install the debug variant from Android Studio. The repository review did not run `assembleDebug`.

## First launch

1. Grant location permission. Fine permission uses the GPS provider; coarse-only permission uses the network provider.
2. The app starts in simulation mode. Use the mode button to switch to live sensors.
3. The app requests a location foreground service and may request battery-optimization exemption when fine location is granted.
4. The map uses osmdroid's online Mapnik source and needs network access for tiles. It is not an offline map package.
5. In live mode, accelerometer and gyroscope events feed a one-second normalized ONNX input window. Inference is attempted about every 100 ms after the window fills.

The latency badge measures elapsed time around `OrtSession.run`. NNAPI is requested when available, but the app does not report the selected execution provider. A displayed number is not proof of Hexagon NPU execution.

## Laptop telemetry bridge

The WebSocket server listens on Android loopback port `8765`. Connect over USB:

```bash
adb reverse tcp:8765 tcp:8765
```

Then use the web cockpit's Phone Link button at `localhost:8765`. The bridge sends sensor/model values, blackout state, position, and heading at the inference cadence (about 10 Hz). It does not accept remote GPS-control commands.

## Items still requiring device verification

- Confirm the APK builds and starts on the intended phone.
- Measure actual inference latency and identify the selected ONNX Runtime execution provider.
- Verify live GPS updates, route warning behavior, blackout handling, and reacquisition while moving.
- Run physical disconnect and screen-off/Doze checks; the foreground service heartbeat alone does not prove sensor sampling continued.
- Check battery use, thermal behavior, permissions, and map behavior without network access.
