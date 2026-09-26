# TrueTrack Android App — Build & Run on a Phone

Verified working instructions for deploying the on-device TrueTrack cockpit
(`com.truetrack.navigation`) to a physical Android phone from this Windows machine.

> Branch: `feature/android-app-setup-v2` = `feature/android-app-setup` (bc6d618: osmdroid dark map,
> preloaded offline tiles, HITEC City corridor, 1300-frame flight-recorder simulation, ZUPT lock,
> 45-degree roll clamp) merged with `main` (d9f9270: HandlerThread IMU polling + serialized
> background ONNX inference).
> Verified: `BUILD SUCCESSFUL` (Gradle 8.12 / AGP 8.2.2 / Kotlin 1.9.22 / JDK 17) producing
> `android_app/app/build/outputs/apk/debug/app-debug.apk` (~56 MB incl. the ONNX model, the
> 1300-frame simulation dump and 28 offline map tiles).

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Android SDK | Installed at `%LOCALAPPDATA%\Android\Sdk` (Android Studio). `adb` ships in `platform-tools\`. |
| **JDK 17 (or 21)** | **Mandatory.** Do **not** use Android Studio's bundled JBR (Java 25) — Gradle 8.12 / AGP 8.2.2 abort with `Unsupported class file major version 69`. |
| Phone | Android 8.0+ (`minSdk 26`); the APK ships `arm64-v8a`, `armeabi-v7a` and `x86_64` libs. |
| USB cable | A data cable, plus USB debugging enabled on the phone. |

The JDK used for the verified build on this machine:

```
C:\Users\Abhinav\Tools\jdk17\jdk-17.0.20.1+1
```

### Point Gradle at JDK 17

* **Command line (per shell):**
  ```powershell
  $env:JAVA_HOME = 'C:\Users\Abhinav\Tools\jdk17\jdk-17.0.20.1+1'
  ```
* **Machine-wide for all Gradle builds** — add to `C:\Users\Abhinav\.gradle\gradle.properties`:
  ```properties
  org.gradle.java.home=C:/Users/Abhinav/Tools/jdk17/jdk-17.0.20.1+1
  ```
* **Android Studio:** *Settings → Build, Execution, Deployment → Build Tools → Gradle → Gradle JDK → 17*.

### Local SDK path (`android_app/local.properties`, git-ignored)

```properties
sdk.dir=C:/Users/Abhinav/AppData/Local/Android/Sdk
```

If Android SDK Platform 34 / Build-Tools 34.0.0 are missing, the Android Gradle Plugin
downloads them automatically on the first build (the SDK licences are already accepted here).

---

## 2. Build the debug APK

```powershell
cd C:\Users\Abhinav\TrueTrack\android_app
$env:JAVA_HOME = 'C:\Users\Abhinav\Tools\jdk17\jdk-17.0.20.1+1'
.\gradlew.bat :app:assembleDebug
```

Output: `android_app\app\build\outputs\apk\debug\app-debug.apk` (~56 MB — bundles the
`onnxruntime-android` native libraries, `truetrack_model.onnx`, `norm_stats.json`,
`npu_model_spec.json`, the 1300-frame `corridor_telemetry.json` and the offline OSM tiles).

---

## 3. Install on the phone

1. **Enable Developer options:** *Settings → About phone → tap "Build number" 7 times*.
2. **Enable USB debugging:** *Settings → System → Developer options → USB debugging → ON*.
3. Connect the phone by USB and set the USB mode to **File transfer / MTP** (not "Charging only"),
   otherwise `adb` cannot see the device.
4. Accept the **"Allow USB debugging?"** dialog on the phone and tick *Always allow from this computer*.
5. Verify the connection:

   ```powershell
   adb devices -l
   # 10BFAU1572000XR   device        -> ready
   # 10BFAU1572000XR   unauthorized  -> tap "Allow" on the phone, or: adb kill-server
   ```
6. Install (or use `.\gradlew.bat :app:installDebug`):

   ```powershell
   adb install -r C:\Users\Abhinav\TrueTrack\android_app\app\build\outputs\apk\debug\app-debug.apk
   ```
7. Launch from the app drawer (**TrueTrack**) or:

   ```powershell
   adb shell am start -n com.truetrack.navigation/.MainActivity
   ```

**Android Studio alternative:** *Open* `C:\Users\Abhinav\TrueTrack\android_app`, set the Gradle JDK
to 17 (step 1), pick the phone in the device dropdown and press **Run ▶**.

**Wireless (optional):** *Developer options → Wireless debugging → Pair device with pairing code*, then
`adb pair <ip>:<port> <code>` followed by `adb connect <ip>:5555`.

---

## 4. First-run verification checklist

On first launch the app asks for **location permission** (`ACCESS_FINE_LOCATION` /
`ACCESS_COARSE_LOCATION`) — tap **Allow** (it is also what lets Android 13+ expose the Wi-Fi IP used
by the telemetry bridge). Everything else it needs (`HIGH_SAMPLING_RATE_SENSORS`, `WAKE_LOCK`,
`VIBRATE`) is granted automatically.

1. Launch **TrueTrack** — the cockpit should render immediately with:
   * header `TRUETRACK` + badge `HEXAGON NPU 1.4ms`,
   * GNSS status banner, speed hero card, lean-angle card,
   * a 2x2 grid: `YAW RATE (wz)`, `LATERAL CONSTRAINT`, `DRIFT RATE`, `DUAL-SCREEN BRIDGE`,
   * a dark OpenStreetMap card showing the HITEC City corridor, the road centreline, the tunnel
     segment, the vehicle marker and the TrueTrack-vs-naive track polylines (with a recenter button),
   * bottom controls `KILL GPS FIX` / `MODE: SIMULATION` / `AUDIO: ON` / `LEAN NHC: ON`.
2. Confirm the assets and model loaded:

   ```powershell
   adb logcat -s MainActivity:V
   # "Preloaded offline corridor map tiles into local cache"
   # "Loaded 1300 simulation telemetry frames"
   # "Successfully loaded normalization stats"
   # "TrueTrack ONNX model successfully initialized"
   # "Configured Qualcomm Hexagon NNAPI hardware acceleration"   (or the CPU fallback warning)
   ```
3. **Simulation mode is the default:** the map replays the 1300-frame corridor run, starting 3 s
   before the underpass blackout. Confirm the marker moves, the polylines accumulate and the
   blackout banner + timer engage on their own at the tunnel segment.
4. Tap **MODE: SIMULATION** -> it becomes `MODE: LIVE SENSOR`: the readouts now come from the phone's
   real 50 Hz IMU + on-device ONNX inference. Move/rotate the phone and check that speed, lean angle
   and yaw rate react; hold it still and the ZUPT standstill lock should drop speed to `0.0`.
5. Tap **KILL GPS FIX (SIMULATE BLACKOUT)** -> banner turns red, the timer counts up, `DRIFT RATE`
   switches to the dead-reckoning value and the drift polyline starts diverging. Tap again to restore.
6. Use the map **recenter button** to re-enable auto-follow, and toggle **AUDIO: ON/OFF** and
   **LEAN NHC: ON/OFF** to confirm those controls respond.
7. The app does **not** hold the screen awake: enable *Developer options → Stay awake* (or keep
   tapping the screen), otherwise it will sleep during a drive.

---

## 5. Optional: laptop cockpit (dual-screen bridge)

* On launch the app starts a WebSocket telemetry server on port **8765**; the `DUAL-SCREEN BRIDGE`
  card shows `:8765` (or `:8765 (off)` if the server could not bind the port).
* Find the phone's Wi-Fi address and point the laptop cockpit at it:

  ```powershell
  adb shell ip -4 addr show wlan0
  # then on the laptop open web_app/index.html and connect to: ws://<phone-ip>:8765
  ```
* Put the phone and the laptop on the same Wi-Fi / hotspot and keep the app in the foreground while
  the cockpit is connected (the app only broadcasts while it is running).

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `Unsupported class file major version 69` | Gradle is running on Java 25 (Android Studio JBR). Set the Gradle JDK to **17** (step 1). |
| `device unauthorized` | Accept *Allow USB debugging* on the phone, then `adb kill-server` and `adb devices`. |
| `no devices/emulators found` | Use a data cable, switch USB mode to *File transfer*, install the OEM USB driver, try another port. |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | `adb uninstall com.truetrack.navigation`, then install again. |
| `Failed to find Build Tools revision 34.0.0` | Install *SDK Platform 34* + *Build-Tools 34.0.0* in the SDK Manager (or let AGP auto-download on a networked build). |
| Installs but shows a blank/dark screen | `adb logcat -s AndroidRuntime:E MainActivity:V` and look for inflation or asset errors. |
