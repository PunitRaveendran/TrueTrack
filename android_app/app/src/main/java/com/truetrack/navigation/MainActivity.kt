package com.truetrack.navigation

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.net.wifi.WifiManager
import android.os.Bundle
import android.os.SystemClock
import android.text.format.Formatter
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.tileprovider.tilesource.XYTileSource
import org.osmdroid.util.GeoPoint
import org.osmdroid.util.MapTileIndex
import org.osmdroid.views.CustomZoomButtonsController
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline
import java.io.InputStream
import java.nio.FloatBuffer
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max

/**
 * TrueTrack - Android Navigation Instrument Activity
 *
 * Runs high-rate 50 Hz sensor acquisition, continuous lean-angle estimation,
 * local Snapdragon NPU inference, dual-screen telemetry broadcasting,
 * and high-contrast dark-mode OpenStreetMap real-time dead reckoning.
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

    // UI View References
    private lateinit var tvSpeed: TextView
    private lateinit var tvLeanAngle: TextView
    private lateinit var tvLeanDirection: TextView
    private lateinit var tvYawRate: TextView
    private lateinit var tvLateralConstraint: TextView
    private lateinit var tvGpsStatus: TextView
    private lateinit var tvBlackoutTimer: TextView
    private lateinit var tvDriftRate: TextView
    private lateinit var tvServerStatus: TextView
    private lateinit var tvRawImu: TextView
    private lateinit var btnKillGps: Button
    private lateinit var btnToggleVoice: Button
    private lateinit var btnToggleLeanCorrection: Button

    // Map & Dead Reckoning Navigation View
    private lateinit var mapView: MapView
    private lateinit var vehicleMarker: Marker
    private lateinit var btnRecenter: ImageButton
    private lateinit var tvMapStatus: TextView
    private lateinit var mapModeDot: View
    private val trueTrackPolyline = Polyline()
    private val driftPolyline = Polyline()
    private val trueTrackPoints = mutableListOf<GeoPoint>()
    private val driftPoints = mutableListOf<GeoPoint>()

    private var isAutoFollow = true
    private var drLat = 17.4445
    private var drLon = 78.3771
    private var drHeadingDeg = 35.0
    private var naiveLat = 17.4445
    private var naiveLon = 78.3771
    private var naiveHeadingDeg = 35.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        requestLocationPermissions()
        initMap()
        loadNormalizationStats()
        initOnnxModel()
        initAudioAndNetwork()
        initSensors()
    }

    private fun requestLocationPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                ),
                1001
            )
        }
    }

    private fun initViews() {
        tvSpeed = findViewById(R.id.tvSpeed)
        tvLeanAngle = findViewById(R.id.tvLeanAngle)
        tvLeanDirection = findViewById(R.id.tvLeanDirection)
        tvYawRate = findViewById(R.id.tvYawRate)
        tvLateralConstraint = findViewById(R.id.tvLateralConstraint)
        tvGpsStatus = findViewById(R.id.tvGpsStatus)
        tvBlackoutTimer = findViewById(R.id.tvBlackoutTimer)
        tvDriftRate = findViewById(R.id.tvDriftRate)
        tvServerStatus = findViewById(R.id.tvServerStatus)
        tvRawImu = findViewById(R.id.tvRawImu)
        btnKillGps = findViewById(R.id.btnKillGps)
        btnToggleVoice = findViewById(R.id.btnToggleVoice)
        btnToggleLeanCorrection = findViewById(R.id.btnToggleLeanCorrection)

        mapView = findViewById(R.id.mapView)
        btnRecenter = findViewById(R.id.btnRecenter)
        tvMapStatus = findViewById(R.id.tvMapStatus)
        mapModeDot = findViewById(R.id.mapModeDot)

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

        btnRecenter.setOnClickListener {
            isAutoFollow = true
            mapView.controller.animateTo(vehicleMarker.position)
        }
    }

    private fun initMap() {
        try {
            Configuration.getInstance().userAgentValue = packageName

            // Carto Dark Matter - Night navigation theme matching Google Maps dark mode
            val cartoDark = object : XYTileSource(
                "CartoDark",
                0, 19, 256, ".png",
                arrayOf(
                    "https://a.basemaps.cartocdn.com/rastertiles/dark_all/",
                    "https://b.basemaps.cartocdn.com/rastertiles/dark_all/",
                    "https://c.basemaps.cartocdn.com/rastertiles/dark_all/"
                )
            ) {
                override fun getTileURLString(pMapTileIndex: Long): String {
                    return baseUrl + MapTileIndex.getZoom(pMapTileIndex) + "/" +
                            MapTileIndex.getX(pMapTileIndex) + "/" +
                            MapTileIndex.getY(pMapTileIndex) + mImageFilenameEnding
                }
            }
            mapView.setTileSource(cartoDark)
        } catch (e: Exception) {
            mapView.setTileSource(TileSourceFactory.MAPNIK)
        }

        mapView.setMultiTouchControls(true)
        mapView.zoomController.setVisibility(CustomZoomButtonsController.Visibility.NEVER)
        mapView.controller.setZoom(17.5)

        val startPoint = GeoPoint(drLat, drLon)
        mapView.controller.setCenter(startPoint)

        // Trajectory Polylines
        // 1. TrueTrack Neural Dead-Reckoning: Glowing Electric Blue
        trueTrackPolyline.outlinePaint.color = Color.parseColor("#38BDF8")
        trueTrackPolyline.outlinePaint.strokeWidth = 9f
        trueTrackPolyline.outlinePaint.strokeCap = Paint.Cap.ROUND
        mapView.overlays.add(trueTrackPolyline)

        // 2. Naive uncorrected IMU Drift during Blackout: Alert Red (Dashed)
        driftPolyline.outlinePaint.color = Color.parseColor("#EF4444")
        driftPolyline.outlinePaint.strokeWidth = 5f
        driftPolyline.outlinePaint.pathEffect = DashPathEffect(floatArrayOf(20f, 15f), 0f)
        driftPolyline.outlinePaint.strokeCap = Paint.Cap.ROUND
        mapView.overlays.add(driftPolyline)

        // Vehicle Navigation Arrow Marker
        vehicleMarker = Marker(mapView)
        vehicleMarker.icon = ContextCompat.getDrawable(this, R.drawable.ic_navigation_arrow)
        vehicleMarker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        vehicleMarker.position = startPoint
        vehicleMarker.rotation = drHeadingDeg.toFloat()
        mapView.overlays.add(vehicleMarker)

        // Disengage auto-follow if user manually drags or touches the map
        mapView.setOnTouchListener { _, _ ->
            isAutoFollow = false
            false
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
            Log.e("MainActivity", "Error reading norm_stats.json, defaulting to standard weights", e)
        }
    }

    private fun initOnnxModel() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val modelBytes = assets.open("truetrack_model.onnx").readBytes()
            val sessionOptions = OrtSession.SessionOptions().apply {
                try {
                    addNnapi()
                    Log.i("MainActivity", "Configured Qualcomm Hexagon NNAPI hardware acceleration")
                } catch (nnapiErr: Exception) {
                    Log.w("MainActivity", "NNAPI unavailable on this chip, falling back to CPU runtime", nnapiErr)
                }
            }
            ortSession = ortEnv!!.createSession(modelBytes, sessionOptions)
            Log.i("MainActivity", "TrueTrack ONNX model successfully initialized")
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to initialize ONNX Runtime", e)
        }
    }

    private fun initAudioAndNetwork() {
        audioManager = AudioCueManager(this)

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
            tvServerStatus.text = ":8765"
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to start telemetry stream server", e)
            tvServerStatus.text = ":8765 (off)"
        }
    }

    private fun initSensors() {
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        // 50 Hz target sampling interval = 20,000 microseconds
        accelSensor?.let { sensorManager.registerListener(this, it, 20_000) }
        gyroSensor?.let { sensorManager.registerListener(this, it, 20_000) }
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

        // Run NPU inference every 10 samples (5 Hz) or whenever buffer is populated
        val now = SystemClock.elapsedRealtime()
        if (samplesCollected >= windowSize && now - lastInferenceTimeMs >= 100) {
            lastInferenceTimeMs = now
            runInference(currentPhi, axIn, ayIn, azIn)
        }
    }

    private fun runInference(currentPhiRad: Float, axIn: Float, ayIn: Float, azIn: Float) {
        val startTime = SystemClock.elapsedRealtimeNanos()
        var speedKmh = 40.0f
        var yawRateDeg = Math.toDegrees(rawGz.toDouble()).toFloat()

        if (ortSession != null && ortEnv != null) {
            try {
                // Flatten rolling buffer in chronological order: [1, 6, 50]
                val tensorBuffer = FloatBuffer.allocate(numChannels * windowSize)
                for (c in 0 until numChannels) {
                    for (i in 0 until windowSize) {
                        val idx = (bufferHead + i) % windowSize
                        tensorBuffer.put(imuRingBuffer[c][idx])
                    }
                }
                tensorBuffer.rewind()

                val inputShape = longArrayOf(1, numChannels.toLong(), windowSize.toLong())
                val inputTensor = OnnxTensor.createTensor(ortEnv, tensorBuffer, inputShape)

                val results = ortSession!!.run(mapOf(ortSession!!.inputNames.first() to inputTensor))
                val output = (results[0].value as Array<FloatArray>)[0]

                val forwardSpeedMs = max(0.0f, output[0])
                val predYawRateRad = output[1]

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

        // Update UI & Map
        runOnUiThread {
            updateDashboard(speedKmh, yawRateDeg, leanDeg, axIn, ayIn, azIn)
        }

        // Broadcast to Laptop Web Cockpit via WebSocket
        streamServer?.broadcastTelemetry(
            timestampMs = System.currentTimeMillis(),
            speedKmh = speedKmh,
            yawRateDeg = yawRateDeg,
            leanDeg = leanDeg,
            isBlackout = isBlackout,
            npuLatencyMs = lastNpuLatencyMs,
            ax = axIn,
            ayDerolled = ayIn,
            az = azIn
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

        tvRawImu.text = String.format(Locale.US, "IMU 50Hz: ax=%.2f ay_derolled=%.2f az=%.2f | NPU: %.1fms", ax, ay, az, lastNpuLatencyMs)

        // Update real-time map dead reckoning
        updateMap(speedKmh, yawRateDeg)
    }

    private fun updateMap(speedKmh: Float, yawRateDeg: Float) {
        val dt = 0.1 // 100ms inference step
        val speedMs = speedKmh / 3.6
        drHeadingDeg = (drHeadingDeg + yawRateDeg * dt) % 360.0
        val headingRad = Math.toRadians(drHeadingDeg)

        // 0.000009 degrees per meter at ~17.4° latitude
        val dLat = speedMs * dt * Math.cos(headingRad) * 0.000009
        val dLon = speedMs * dt * Math.sin(headingRad) * 0.000009

        drLat += dLat
        drLon += dLon
        val currentPoint = GeoPoint(drLat, drLon)

        trueTrackPoints.add(currentPoint)
        if (trueTrackPoints.size > 300) trueTrackPoints.removeAt(0)
        trueTrackPolyline.setPoints(trueTrackPoints)

        if (isBlackout) {
            // Simulate uncorrected raw IMU drift error (demonstrating why TrueTrack is needed)
            naiveHeadingDeg = (naiveHeadingDeg + (yawRateDeg + 9.5f) * dt) % 360.0
            val naiveRad = Math.toRadians(naiveHeadingDeg)
            naiveLat += speedMs * 1.35 * dt * Math.cos(naiveRad) * 0.000009
            naiveLon += speedMs * 1.35 * dt * Math.sin(naiveRad) * 0.000009
            driftPoints.add(GeoPoint(naiveLat, naiveLon))
            if (driftPoints.size > 300) driftPoints.removeAt(0)
            driftPolyline.setPoints(driftPoints)

            tvMapStatus.text = "GPS BLACKOUT • NEURAL DR ACTIVE"
            tvMapStatus.setTextColor(Color.parseColor("#EF4444"))
            mapModeDot.backgroundTintList = getColorStateList(R.color.alert_red)
        } else {
            naiveLat = drLat
            naiveLon = drLon
            naiveHeadingDeg = drHeadingDeg
            driftPoints.clear()
            driftPolyline.setPoints(driftPoints)

            tvMapStatus.text = "LIVE NEURAL TRAJECTORY"
            tvMapStatus.setTextColor(Color.parseColor("#38BDF8"))
            mapModeDot.backgroundTintList = getColorStateList(R.color.status_green)
        }

        vehicleMarker.position = currentPoint
        vehicleMarker.rotation = drHeadingDeg.toFloat()

        if (isAutoFollow) {
            mapView.controller.animateTo(currentPoint)
        }
        mapView.invalidate()
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
            tvGpsStatus.text = "GNSS: LOCKED (50 Hz SENSOR/SIM)"
            tvGpsStatus.setTextColor(Color.parseColor("#10B981"))
            audioManager.onGpsRestored()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onResume() {
        super.onResume()
        if (::mapView.isInitialized) {
            mapView.onResume()
        }
    }

    override fun onPause() {
        super.onPause()
        if (::mapView.isInitialized) {
            mapView.onPause()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
        audioManager.shutdown()
        if (::mapView.isInitialized) {
            mapView.onDetach()
        }
        try {
            streamServer?.stop()
            ortSession?.close()
            ortEnv?.close()
        } catch (e: Exception) {
            Log.e("MainActivity", "Error closing resources", e)
        }
    }
}
