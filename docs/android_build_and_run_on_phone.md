# TrueTrack Android App — Build & Run on a Phone

Verified working instructions for deploying the on-device TrueTrack cockpit
(`com.truetrack.navigation`) to a physical Android phone from this Windows machine.

> Build status: `BUILD SUCCESSFUL in 3m 7s` (37 tasks) with Gradle 8.12 + AGP 8.2.2 +
> Kotlin 1.9.22 on JDK 17, producing `android_app/app/build/outputs/apk/debug/app-debug.apk`.

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

Output: `android_app\app\build\outputs\apk\debug\app-debug.apk` (~53 MB — bundles the
`onnxruntime-android` native libraries, `truetrack_model.onnx`, `norm_stats.json` and
`npu_model_spec.json`).

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

The app needs **no runtime permission dialogs** (only normal permissions are used:
`HIGH_SAMPLING_RATE_SENSORS`, `WAKE_LOCK`, `VIBRATE`, plus the declared location/Wi-Fi permissions).

1. Launch **TrueTrack** — the cockpit should render immediately with:
   * header `TRUETRACK` + badge `HEXAGON NPU 1.4ms`,
   * GNSS status banner, speed hero card, lean-angle card,
   * a 2x2 grid: `YAW RATE (wz)`, `LATERAL CONSTRAINT`, `DRIFT RATE`, `DUAL-SCREEN BRIDGE`,
   * the raw diagnostics line and the red `KILL GPS FIX (SIMULATE BLACKOUT)` button.
2. Confirm the model + NPU delegate loaded:

   ```powershell
   adb logcat -s MainActivity:V
   # "Successfully loaded normalization stats"
   # "ONNX model loaded successfully (<nnn> KB)"
   # "Qualcomm Hexagon NPU acceleration provider initialized"  (or "... fallback to CPU")
   ```
3. Move/rotate the phone -> speed, lean angle and yaw-rate readouts must react (50 Hz IMU).
4. Tap **KILL GPS FIX (SIMULATE BLACKOUT)** -> banner turns red, blackout timer counts up,
   `DRIFT RATE` switches to `0.008 m/s (DR)` and the diagnostics line shows live dead-reckoned
   `LAT/LON`. Tap again to restore the fix.
5. Toggle **AUDIO: ON/OFF** and **LEAN NHC: ACTIVE/OFF** to confirm the buttons respond.
6. The app does **not** hold the screen awake: enable *Developer options → Stay awake* (or keep
   tapping the screen), otherwise it will sleep during a drive.

---

## 5. Optional: laptop cockpit (dual-screen bridge)

* On launch the app starts a WebSocket telemetry server on port **8765** and prints the URL in the
  `DUAL-SCREEN BRIDGE` card (`ws://<phone-ip>:8765`).
* Put the phone and the laptop on the same Wi-Fi / hotspot, open `web_app/index.html` on the laptop
  and point it at that `ws://` URL.
* If the card shows `ws://0.0.0.0:8765` or `ws://*:8765 (off)`: join a Wi-Fi network and allow
  **Location** for the app (Android 13+ restricts Wi-Fi info to apps with location permission),
  then relaunch the app.

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
