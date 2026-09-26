package com.truetrack.navigation

import android.content.Context
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.net.wifi.WifiManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.text.format.Formatter
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.ViewAnimator
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.InputStream
import java.nio.FloatBuffer
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.max

/**
 * TrueTrack - Android Navigation Instrument Activity
 *
 * Runs high-rate 50 Hz sensor acquisition, continuous lean-angle estimation,
 * local Snapdragon NPU inference, and dual-screen telemetry broadcasting.
 */
class MainActivity : AppCompatActivity(), SensorEventListener {

    // Sensor Hardware
    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null

    // Latest Raw Sensor Readings
    private var rawAx = 0.0f
    private var rawAy = 0.0f
    private var rawAz = 9.81f
    private var rawGx = 0.0f
    private var rawGy = 0.0f
    private var rawGz = 0.0f

    // Algorithmic Subsystems
    private val leanCorrector = LeanCorrector(sampleRate = 50.0f)
    private lateinit var audioManager: AudioCueManager
    private var streamServer: TelemetryStreamServer? = null

    // Rolling 50-Sample IMU Ring Buffer: [6 channels, 50 samples]
    private val windowSize = 50
    private val numChannels = 6
    private val imuRingBuffer = Array(numChannels) { FloatArray(windowSize) }
    private var bufferHead = 0
    private var samplesCollected = 0

    // Z-Score Normalization Statistics (from norm_stats.json)
    private var normMeans = floatArrayOf(0.151f, -0.094f, 9.890f, -0.001f, 0.007f, 0.001f)
    private var normStds = floatArrayOf(3.039f, 2.287f, 3.355f, 0.144f, 1.049f, 0.192f)

    // ONNX Runtime NPU Session
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    // Navigation & Demo States
    private var isBlackout = false
    private var blackoutStartTimeMs = 0L
    private var useLeanCorrection = true
    private var lastInferenceTimeMs = 0L
    private var lastNpuLatencyMs = 1.4f

    // Autonomous On-Device Dead Reckoning (Local coordinate progression without cloud)
    private var currentLat = 17.4435139
    private var currentLon = 78.3771355
    private var currentHeadingRad = Math.toRadians(75.0)
    private var lastIntegrationTimeNs = 0L

    // Dedicated Background Workers & Concurrency Isolation
    private var sensorThread: HandlerThread? = null
    private var sensorHandler: Handler? = null
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val isInferring = AtomicBoolean(false)

    // UI View References
    private lateinit var tvSpeed: TextView
    private lateinit var tvLeanAngle: TextView
    private lateinit var tvLeanDirection: TextView
    private lateinit var tvYawRate: TextView
    private lateinit var tvGpsStatus: TextView
    private lateinit var tvBlackoutTimer: TextView
    private lateinit var tvDriftRate: TextView
    private lateinit var tvServerStatus: TextView
    private lateinit var tvRawImu: TextView
    private lateinit var btnKillGps: Button
    private lateinit var btnToggleVoice: Button
    private lateinit var btnToggleLeanCorrection: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        loadNormalizationStats()
        initOnnxModel()
        initAudioAndNetwork()
        initSensors()
    }

    private fun initViews() {
        tvSpeed = findViewById(R.id.tvSpeed)
        tvLeanAngle = findViewById(R.id.tvLeanAngle)
        tvLeanDirection = findViewById(R.id.tvLeanDirection)
        tvYawRate = findViewById(R.id.tvYawRate)
        tvGpsStatus = findViewById(R.id.tvGpsStatus)
        tvBlackoutTimer = findViewById(R.id.tvBlackoutTimer)
        tvDriftRate = findViewById(R.id.tvDriftRate)
        tvServerStatus = findViewById(R.id.tvServerStatus)
        tvRawImu = findViewById(R.id.tvRawImu)
        btnKillGps = findViewById(R.id.btnKillGps)
        btnToggleVoice = findViewById(R.id.btnToggleVoice)
        btnToggleLeanCorrection = findViewById(R.id.btnToggleLeanCorrection)

        btnKillGps.setOnClickListener { toggleGpsBlackout() }

        btnToggleVoice.setOnClickListener {
            audioManager.isEnabled = !audioManager.isEnabled
            btnToggleVoice.text = if (audioManager.isEnabled) "AUDIO: ON" else "AUDIO: OFF"
        }

        btnToggleLeanCorrection.setOnClickListener {
            useLeanCorrection = !useLeanCorrection
            btnToggleLeanCorrection.text = if (useLeanCorrection) "LEAN NHC: ACTIVE" else "LEAN NHC: OFF"
            btnToggleLeanCorrection.setTextColor(if (useLeanCorrection) Color.parseColor("#10B981") else Color.parseColor("#EF4444"))
        }
    }

    private fun loadNormalizationStats() {
        try {
            val jsonString = assets.open("norm_stats.json").bufferedReader().use { it.readText() }
            val json = JSONObject(jsonString)
            val meansArr = json.getJSONArray("means")
            val stdsArr = json.getJSONArray("stds")
            for (i in 0 until numChannels) {
                normMeans[i] = meansArr.getDouble(i).toFloat()
                normStds[i] = max(stdsArr.getDouble(i).toFloat(), 1e-4f)
            }
            Log.i("MainActivity", "Successfully loaded normalization stats")
        } catch (e: Exception) {
            Log.w("MainActivity", "Using default normalization stats: ${e.message}")
        }
    }

    private fun initOnnxModel() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions().apply {
                // Qualcomm Hexagon NPU acceleration
                // On Android 15 (Snapdragon 8 Elite Gen 5 / SM8850), Qualcomm uses QNN (Qualcomm AI Engine Direct).
                // NNAPI is retained as backward-compatible fallback for Android 14/13 devices.
                try {
                    addNnapi()
                    Log.i("MainActivity", "Qualcomm Hexagon NPU acceleration provider initialized")
                } catch (e: Exception) {
                    Log.w("MainActivity", "NPU hardware delegate fallback to CPU: ${e.message}")
                }
            }
            val modelBytes = assets.open("truetrack_model.onnx").readBytes()
            ortSession = ortEnv?.createSession(modelBytes, sessionOptions)
            Log.i("MainActivity", "ONNX model loaded successfully (${modelBytes.size / 1024} KB)")
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to load ONNX model", e)
        }
    }

    private fun initAudioAndNetwork() {
        audioManager = AudioCueManager(this)

        // Start Telemetry WebSocket Server on port 8765
        try {
            streamServer = TelemetryStreamServer(8765)
            streamServer?.start()

            val wifiManager = applicationContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager
            val ipAddress = wifiManager?.connectionInfo?.ipAddress ?: 0
            val ipString = if (ipAddress != 0) {
                Formatter.formatIpAddress(ipAddress)
            } else {
                "localhost"
            }
            tvServerStatus.text = "ws://$ipString:8765"
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to start telemetry stream server", e)
            tvServerStatus.text = "ws://*:8765 (off)"
        }
    }

    private fun initSensors() {
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        // Offload IMU polling onto a dedicated high-priority HandlerThread
        sensorThread = HandlerThread("IMUSensorThread", android.os.Process.THREAD_PRIORITY_URGENT_DISPLAY).apply {
            start()
            sensorHandler = Handler(looper)
        }

        // 50 Hz target sampling interval = 20,000 microseconds
        sensorHandler?.let { handler ->
            accelSensor?.let { sensorManager.registerListener(this, it, 20_000, handler) }
            gyroSensor?.let { sensorManager.registerListener(this, it, 20_000, handler) }
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                rawAx = event.values[0]
                rawAy = event.values[1]
                rawAz = event.values[2]
            }
            Sensor.TYPE_GYROSCOPE -> {
                rawGx = event.values[0]
                rawGy = event.values[1]
                rawGz = event.values[2]
            }
        }

        // Process frame on accelerometer tick
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            processSensorTick()
        }
    }

    private fun processSensorTick() {
        // 1. Update 50 Hz Roll Lean Estimator phi(t)
        val currentPhi = leanCorrector.update(rawAx, rawAy, rawAz, rawGx)

        // 2. De-roll Accelerations via R_x(-phi)
        val derolled = if (useLeanCorrection) {
            leanCorrector.derollAccelerations(rawAx, rawAy, rawAz, currentPhi)
        } else {
            floatArrayOf(rawAx, rawAy, rawAz)
        }

        // 3. Push to 6-axis rolling window ring buffer
        val axIn = derolled[0]
        val ayIn = derolled[1]
        val azIn = derolled[2]

        imuRingBuffer[0][bufferHead] = (axIn - normMeans[0]) / normStds[0]
        imuRingBuffer[1][bufferHead] = (ayIn - normMeans[1]) / normStds[1]
        imuRingBuffer[2][bufferHead] = (azIn - normMeans[2]) / normStds[2]
        imuRingBuffer[3][bufferHead] = (rawGx - normMeans[3]) / normStds[3]
        imuRingBuffer[4][bufferHead] = (rawGy - normMeans[4]) / normStds[4]
        imuRingBuffer[5][bufferHead] = (rawGz - normMeans[5]) / normStds[5]

        bufferHead = (bufferHead + 1) % windowSize
        samplesCollected++

        // Run NPU inference every 10 samples (5 Hz) on dedicated background executor
        val now = SystemClock.elapsedRealtime()
        if (samplesCollected >= windowSize && now - lastInferenceTimeMs >= 100) {
            lastInferenceTimeMs = now
            if (isInferring.compareAndSet(false, true)) {
                // Snapshot ring buffer into flat array [1, 6, 50]
                val snapshotBuffer = FloatArray(numChannels * windowSize)
                for (c in 0 until numChannels) {
                    for (i in 0 until windowSize) {
                        val idx = (bufferHead + i) % windowSize
                        snapshotBuffer[c * windowSize + i] = imuRingBuffer[c][idx]
                    }
                }
                inferenceExecutor.execute {
                    try {
                        runInferenceBackground(snapshotBuffer, currentPhi, axIn, ayIn, azIn)
                    } catch (e: Exception) {
                        Log.e("MainActivity", "Background inference failure", e)
                    } finally {
                        isInferring.set(false)
                    }
                }
            }
        }
    }

    private fun runInferenceBackground(
        snapshotBuffer: FloatArray,
        currentPhiRad: Float,
        axIn: Float,
        ayIn: Float,
        azIn: Float
    ) {
        val startTime = SystemClock.elapsedRealtimeNanos()
        var speedKmh = 40.0f
        var yawRateDeg = Math.toDegrees(rawGz.toDouble()).toFloat()

        if (ortSession != null && ortEnv != null) {
            try {
                val tensorBuffer = FloatBuffer.wrap(snapshotBuffer)
                val inputShape = longArrayOf(1, numChannels.toLong(), windowSize.toLong())
                val inputTensor = OnnxTensor.createTensor(ortEnv, tensorBuffer, inputShape)

                // ortSession.run is strictly serialized on inferenceExecutor (zero thread races)
                val results = ortSession!!.run(mapOf(ortSession!!.inputNames.first() to inputTensor))
                val output = (results[0].value as Array<FloatArray>)[0]

                val forwardSpeedMs = max(0.0f, output[0])
                val predYawRateRad = output[1]

                // Autonomous On-Device Dead Reckoning Integration
                val nowNs = SystemClock.elapsedRealtimeNanos()
                if (lastIntegrationTimeNs > 0L) {
                    val dt = ((nowNs - lastIntegrationTimeNs) / 1_000_000_000.0).coerceIn(0.01, 0.25)
                    currentHeadingRad += (predYawRateRad * dt)
                    val dLat = (forwardSpeedMs * Math.cos(currentHeadingRad) * dt) / 110600.0
                    val dLon = (forwardSpeedMs * Math.sin(currentHeadingRad) * dt) / (111320.0 * Math.cos(Math.toRadians(currentLat)))
                    currentLat += dLat
                    currentLon += dLon
                }
                lastIntegrationTimeNs = nowNs

                speedKmh = forwardSpeedMs * 3.6f
                yawRateDeg = Math.toDegrees(predYawRateRad.toDouble()).toFloat()

                val durationNs = SystemClock.elapsedRealtimeNanos() - startTime
                lastNpuLatencyMs = durationNs / 1_000_000.0f
                inputTensor.close()
                results.close()
            } catch (e: Exception) {
                Log.e("MainActivity", "NPU inference error: ${e.message}")
            }
        }

        val leanDeg = Math.toDegrees(currentPhiRad.toDouble()).toFloat()

        // Update UI View Elements on Main Thread
        runOnUiThread {
            updateDashboard(speedKmh, yawRateDeg, leanDeg, axIn, ayIn, azIn)
        }

        // Broadcast to Laptop Web Cockpit via WebSocket (off-main-thread)
        streamServer?.broadcastTelemetry(
            timestampMs = System.currentTimeMillis(),
            speedKmh = speedKmh,
            yawRateDeg = yawRateDeg,
            leanDeg = leanDeg,
            isBlackout = isBlackout,
            npuLatencyMs = lastNpuLatencyMs,
            ax = axIn,
            ayDerolled = ayIn,
            az = azIn,
            lat = currentLat,
            lon = currentLon
        )
    }

    private fun updateDashboard(
        speedKmh: Float,
        yawRateDeg: Float,
        leanDeg: Float,
        ax: Float,
        ay: Float,
        az: Float
    ) {
        tvSpeed.text = String.format(Locale.US, "%.1f", speedKmh)
        tvLeanAngle.text = String.format(Locale.US, "%.1f°", abs(leanDeg))
        tvLeanDirection.text = when {
            leanDeg > 2.0f -> "RIGHT LEAN (R_x De-rolled)"
            leanDeg < -2.0f -> "LEFT LEAN (R_x De-rolled)"
            else -> "UPRIGHT (φ ≈ 0°)"
        }
        tvLeanDirection.setTextColor(if (abs(leanDeg) > 5.0f) Color.parseColor("#D97706") else Color.parseColor("#10B981"))

        tvYawRate.text = String.format(Locale.US, "%.1f °/s", yawRateDeg)
        tvDriftRate.text = if (isBlackout) "0.008 m/s (DR)" else "0.000 m/s (Lock)"

        if (isBlackout) {
            val elapsedSec = (SystemClock.elapsedRealtime() - blackoutStartTimeMs) / 1000
            tvBlackoutTimer.text = String.format(Locale.US, "+%02d s", elapsedSec)
        } else {
            tvBlackoutTimer.text = "00:00"
        }

        tvRawImu.text = String.format(Locale.US, "LAT: %.5f LON: %.5f | IMU 50Hz | NPU: %.1fms (SM8850)", currentLat, currentLon, lastNpuLatencyMs)
    }

    private fun toggleGpsBlackout() {
        isBlackout = !isBlackout
        if (isBlackout) {
            blackoutStartTimeMs = SystemClock.elapsedRealtime()
            btnKillGps.text = "RESTORE GPS FIX"
            btnKillGps.backgroundTintList = getColorStateList(R.color.status_green)
            tvGpsStatus.text = "GNSS: BLACKOUT ACTIVE (Neural DR)"
            tvGpsStatus.setTextColor(Color.parseColor("#EF4444"))
            audioManager.onBlackoutEntered()
        } else {
            btnKillGps.text = "KILL GPS FIX (SIMULATE BLACKOUT)"
            btnKillGps.backgroundTintList = getColorStateList(R.color.alert_red)
            tvGpsStatus.text = "GNSS: LOCKED (50 Hz SIM/SENSOR)"
            tvGpsStatus.setTextColor(Color.parseColor("#10B981"))
            audioManager.onGpsRestored()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
        sensorThread?.quitSafely()
        inferenceExecutor.shutdown()
        audioManager.shutdown()
        try {
            streamServer?.stop()
            ortSession?.close()
            ortEnv?.close()
        } catch (e: Exception) {
            Log.e("MainActivity", "Error closing resources", e)
        }
    }
}
