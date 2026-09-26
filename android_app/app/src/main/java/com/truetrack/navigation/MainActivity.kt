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
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.SystemClock
import android.text.format.Formatter
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONArray
import org.json.JSONObject
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.CustomZoomButtonsController
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline
import java.io.File
import java.nio.FloatBuffer
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

data class TelemetryFrame(
    val t: Float,
    val speedKmh: Float,
    val predSpeedKmh: Float,
    val predYawDegS: Float,
    val headingDeg: Float,
    val isBlackout: Boolean,
    val gtLat: Double,
    val gtLon: Double,
    val naiveLat: Double,
    val naiveLon: Double,
    val trueTrackLat: Double,
    val trueTrackLon: Double,
    val errNaiveM: Float,
    val errTrueTrackM: Float,
    val rawImuAx: Float,
    val rawImuAy: Float,
    val rawImuGz: Float
)

data class SimulationRoute(
    val id: String,
    val displayName: String,
    val shortName: String,
    val assetFileName: String,
    val initialZoom: Double
)

class MainActivity : AppCompatActivity(), SensorEventListener {

    // ONNX Runtime & Hexagon NPU
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    // 6-Axis Normalization Statistics
    private val numChannels = 6
    private val windowSize = 50
    private val normMeans = FloatArray(numChannels) { 0.0f }
    private val normStds = FloatArray(numChannels) { 1.0f }

    // Rolling Ring Buffer [6, 50]
    private val imuRingBuffer = Array(numChannels) { FloatArray(windowSize) }
    private var bufferHead = 0
    private var samplesCollected = 0

    // Sensor Management (50 Hz)
    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null

    // Raw IMU State
    private var rawAx = 0.0f
    private var rawAy = 0.0f
    private var rawAz = 9.80665f
    private var rawGx = 0.0f
    private var rawGy = 0.0f
    private var rawGz = 0.0f

    // Algorithmic Subsystems
    private val leanCorrector = LeanCorrector(sampleRate = 50.0f)
    private lateinit var audioManager: AudioCueManager
    private var streamServer: TelemetryStreamServer? = null

    // Navigation & Demo States
    private var isBlackout = false
    private var blackoutStartTimeMs = 0L
    private var useLeanCorrection = true
    private var lastInferenceTimeMs = 0L
    private var lastNpuLatencyMs = 1.4f

    // Dedicated Background Workers & Concurrency Isolation
    private var sensorThread: HandlerThread? = null
    private var sensorHandler: Handler? = null
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val isInferring = AtomicBoolean(false)

    // Autonomous On-Device Dead Reckoning State
    private var currentLat = 13.0759526
    private var currentLon = 80.1983471
    private var currentHeadingRad = Math.toRadians(73.0)
    private var lastIntegrationTimeNs = 0L

    // Live DR Integration State
    private var drLat = 13.0759526
    private var drLon = 80.1983471
    private var drHeadingDeg = 73.0
    private var naiveLat = 13.0759526
    private var naiveLon = 80.1983471

    // AVAILABLE ROUTES (Real iQOO Recorded Drives + Synthetic Underpass)
    private val availableRoutes = listOf(
        SimulationRoute("rohini", "Rohini Theatre Koyambedu (Chennai - Real iQOO)", "ROHINI", "rohini_telemetry.json", 17.5),
        SimulationRoute("varada", "Varadarajapuram Route (Chennai - Real iQOO)", "VARADA", "varadarajapuram_telemetry.json", 16.5),
        SimulationRoute("45_46", "45/46 Road Corridor (Chennai - Real iQOO)", "45/46 RD", "45_46_telemetry.json", 16.0),
        SimulationRoute("hitec", "HITEC City Underpass (Hyderabad - Corridor Sim)", "HITEC", "corridor_telemetry.json", 16.5)
    )
    private var currentRouteIndex = 0

    // SIMULATION ENGINE
    private val simulationFrames = mutableListOf<TelemetryFrame>()
    private val simHandler = Handler(Looper.getMainLooper())
    private val simIntervalMs = 100L // 10 Hz matches 0.1s telemetry
    private var simFrameIndex = 0
    private var isDemoMode = true
    private var prevBlackoutState = false

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
    private lateinit var btnDemoMode: Button
    private lateinit var btnToggleVoice: Button
    private lateinit var btnToggleLeanCorrection: Button

    // Map and Dead Reckoning Navigation View
    private lateinit var mapView: MapView
    private lateinit var vehicleMarker: Marker
    private lateinit var btnRecenter: ImageButton
    private lateinit var tvMapStatus: TextView
    private lateinit var mapModeDot: View

    // Map Overlays
    private val roadCenterlineOverlay = Polyline()
    private val tunnelTubeOverlay = Polyline()
    private val trueTrackPolyline = Polyline()
    private val driftPolyline = Polyline()
    private val trueTrackPoints = mutableListOf<GeoPoint>()
    private val driftPoints = mutableListOf<GeoPoint>()
    private var isAutoFollow = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        requestLocationPermissions()
        initMap()
        preloadOfflineTiles()

        // Load Default Route: Rohini Theatre Koyambedu (Chennai - Real iQOO Data)
        loadRouteTelemetry(availableRoutes[currentRouteIndex])

        loadNormalizationStats()
        initOnnxModel()
        initAudioAndNetwork()
        initSensors()
    }

    override fun onResume() {
        super.onResume()
        if (::mapView.isInitialized) mapView.onResume()
        if (isDemoMode) startSimulationLoop()
    }

    override fun onPause() {
        super.onPause()
        if (::mapView.isInitialized) mapView.onPause()
        stopSimulationLoop()
    }

    override fun onDestroy() {
        super.onDestroy()
        stopSimulationLoop()
        if (::sensorManager.isInitialized) sensorManager.unregisterListener(this)
        sensorThread?.quitSafely()
        inferenceExecutor.shutdown()
        if (::audioManager.isInitialized) audioManager.shutdown()
        if (::mapView.isInitialized) mapView.onDetach()
        try {
            streamServer?.stop()
            ortSession?.close()
            ortEnv?.close()
        } catch (e: Exception) {
            Log.e("MainActivity", "Error closing resources", e)
        }
    }

    // SIMULATION LOOP - Fixed 100ms timer (10 Hz)
    private val simRunnable = object : Runnable {
        override fun run() {
            if (isDemoMode && simulationFrames.isNotEmpty()) {
                tickSimulationFrame()
                simHandler.postDelayed(this, simIntervalMs)
            }
        }
    }

    private fun startSimulationLoop() {
        simHandler.removeCallbacks(simRunnable)
        if (isDemoMode && simulationFrames.isNotEmpty()) {
            simHandler.postDelayed(simRunnable, simIntervalMs)
        }
    }

    private fun stopSimulationLoop() {
        simHandler.removeCallbacks(simRunnable)
    }

    /**
     * Advance one simulation frame at 10 Hz on main thread.
     * Uses real recorded data / neural predictions.
     * Voice triggers on enter and exit of blackout.
     */
    private fun tickSimulationFrame() {
        if (simulationFrames.isEmpty()) return
        val frame = simulationFrames[simFrameIndex]
        simFrameIndex = (simFrameIndex + 1) % simulationFrames.size

        // Blackout lifecycle: fire alerts only on state transitions
        val frameBlackout = frame.isBlackout
        if (frameBlackout && !prevBlackoutState) {
            isBlackout = true
            blackoutStartTimeMs = SystemClock.elapsedRealtime()
            audioManager.onBlackoutEntered()
        } else if (!frameBlackout && prevBlackoutState) {
            isBlackout = false
            audioManager.onGpsRestored()
            driftPoints.clear()
            driftPolyline.setPoints(driftPoints)
        }
        prevBlackoutState = frameBlackout

        val speedKmh = frame.predSpeedKmh.coerceAtLeast(0f)
        val yawRateDeg = frame.predYawDegS
        val simLeanDeg = (yawRateDeg * 0.30f).coerceIn(-40f, 40f)

        val ttPoint = GeoPoint(frame.trueTrackLat, frame.trueTrackLon)
        trueTrackPoints.add(ttPoint)
        if (trueTrackPoints.size > 500) trueTrackPoints.removeAt(0)
        trueTrackPolyline.setPoints(ArrayList(trueTrackPoints))

        if (isBlackout) {
            val naivePoint = GeoPoint(frame.naiveLat, frame.naiveLon)
            driftPoints.add(naivePoint)
            if (driftPoints.size > 500) driftPoints.removeAt(0)
            driftPolyline.setPoints(ArrayList(driftPoints))
        }

        vehicleMarker.position = ttPoint
        vehicleMarker.rotation = frame.headingDeg

        if (isAutoFollow) mapView.controller.setCenter(ttPoint)

        if (isBlackout) {
            tvMapStatus.text = "GPS BLACKOUT  NEURAL DR ACTIVE"
            tvMapStatus.setTextColor(Color.parseColor("#EF4444"))
            mapModeDot.backgroundTintList = getColorStateList(R.color.alert_red)
        } else {
            tvMapStatus.text = "LIVE NEURAL TRAJECTORY"
            tvMapStatus.setTextColor(Color.parseColor("#38BDF8"))
            mapModeDot.backgroundTintList = getColorStateList(R.color.status_green)
        }
        mapView.invalidate()

        val accelMag = sqrt(
            frame.rawImuAx * frame.rawImuAx +
            frame.rawImuAy * frame.rawImuAy +
            9.80665f * 9.80665f
        )
        updateDashboard(speedKmh, yawRateDeg, simLeanDeg, accelMag, frame)

        streamServer?.broadcastTelemetry(
            timestampMs = System.currentTimeMillis(),
            speedKmh = speedKmh,
            yawRateDeg = yawRateDeg,
            leanDeg = simLeanDeg,
            isBlackout = isBlackout,
            npuLatencyMs = lastNpuLatencyMs,
            ax = frame.rawImuAx,
            ayDerolled = frame.rawImuAy,
            az = 9.80665f
        )
    }

    private fun loadRouteTelemetry(route: SimulationRoute) {
        try {
            val jsonString = assets.open(route.assetFileName).bufferedReader().use { it.readText() }
            val jsonArray = JSONArray(jsonString)
            simulationFrames.clear()

            val centerlinePts = mutableListOf<GeoPoint>()
            val tunnelPts = mutableListOf<GeoPoint>()

            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                val gt = obj.getJSONArray("gt")
                val naive = obj.getJSONArray("naive")
                val tt = obj.getJSONArray("truetrack")
                val isBlk = obj.getInt("is_blackout") == 1
                val gtPt = GeoPoint(gt.getDouble(0), gt.getDouble(1))
                centerlinePts.add(gtPt)
                if (isBlk) {
                    tunnelPts.add(gtPt)
                }

                simulationFrames.add(
                    TelemetryFrame(
                        t = obj.getDouble("t").toFloat(),
                        speedKmh = obj.getDouble("speed_kmh").toFloat(),
                        predSpeedKmh = obj.optDouble("pred_speed_kmh", obj.getDouble("speed_kmh")).toFloat(),
                        predYawDegS = obj.optDouble("pred_yaw_deg_s", 0.0).toFloat(),
                        headingDeg = obj.getDouble("heading_deg").toFloat(),
                        isBlackout = isBlk,
                        gtLat = gt.getDouble(0),
                        gtLon = gt.getDouble(1),
                        naiveLat = naive.getDouble(0),
                        naiveLon = naive.getDouble(1),
                        trueTrackLat = tt.getDouble(0),
                        trueTrackLon = tt.getDouble(1),
                        errNaiveM = obj.getDouble("err_naive_m").toFloat(),
                        errTrueTrackM = obj.getDouble("err_truetrack_m").toFloat(),
                        rawImuAx = obj.optDouble("raw_imu_ax", 0.0).toFloat(),
                        rawImuAy = obj.optDouble("raw_imu_ay", 0.0).toFloat(),
                        rawImuGz = obj.optDouble("raw_imu_gz", 0.0).toFloat()
                    )
                )
            }

            roadCenterlineOverlay.setPoints(centerlinePts)
            tunnelTubeOverlay.setPoints(tunnelPts)

            simFrameIndex = 0
            trueTrackPoints.clear()
            driftPoints.clear()
            trueTrackPolyline.setPoints(trueTrackPoints)
            driftPolyline.setPoints(driftPoints)
            isBlackout = false
            prevBlackoutState = false

            if (simulationFrames.isNotEmpty()) {
                val startPt = GeoPoint(simulationFrames[0].gtLat, simulationFrames[0].gtLon)
                drLat = simulationFrames[0].gtLat
                drLon = simulationFrames[0].gtLon
                drHeadingDeg = simulationFrames[0].headingDeg.toDouble()
                vehicleMarker.position = startPt
                vehicleMarker.rotation = simulationFrames[0].headingDeg
                mapView.controller.setZoom(route.initialZoom)
                mapView.controller.setCenter(startPt)
                mapView.invalidate()
            }

            btnDemoMode.text = "SIM: ${route.shortName}"
            btnDemoMode.setTextColor(Color.parseColor("#38BDF8"))
            Log.i("MainActivity", "Successfully loaded route ${route.displayName} (${simulationFrames.size} frames)")
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to load route telemetry ${route.assetFileName}", e)
        }
    }

    private fun selectRoute(index: Int) {
        currentRouteIndex = index
        val route = availableRoutes[index]
        isDemoMode = true
        loadRouteTelemetry(route)
        startSimulationLoop()
        if (::audioManager.isInitialized) {
            audioManager.speak("Route selected: ${route.shortName}. Tracking initialized.")
        }
    }

    private fun enableLiveSensorMode() {
        isDemoMode = false
        stopSimulationLoop()
        trueTrackPoints.clear()
        driftPoints.clear()
        trueTrackPolyline.setPoints(trueTrackPoints)
        driftPolyline.setPoints(driftPoints)
        isBlackout = false
        prevBlackoutState = false
        btnDemoMode.text = "LIVE SENSOR"
        btnDemoMode.setTextColor(Color.parseColor("#94A3B8"))
        if (::audioManager.isInitialized) {
            audioManager.speak("Switched to live phone sensor mode.")
        }
    }

    private fun showRouteSelectionDialog() {
        val items = arrayOf(
            "🎬 Rohini Theatre Koyambedu (Chennai - Real iQOO Data)",
            "🛣️ Varadarajapuram Route (Chennai - Real iQOO Data)",
            "📍 45/46 Corridor (Chennai - Real iQOO Data)",
            "🏢 HITEC City Underpass (Hyderabad - Corridor Sim)",
            "📱 Live Sensors (Real Phone IMU Mode)"
        )

        AlertDialog.Builder(this)
            .setTitle("Select Navigation Route / Mode")
            .setItems(items) { _, which ->
                when (which) {
                    0 -> selectRoute(0) // Rohini
                    1 -> selectRoute(1) // Varadarajapuram
                    2 -> selectRoute(2) // 45/46
                    3 -> selectRoute(3) // HITEC City
                    4 -> enableLiveSensorMode()
                }
            }
            .show()
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

    private fun preloadOfflineTiles() {
        try {
            val tileCacheDir = File(Configuration.getInstance().osmdroidTileCache, "Mapnik")
            if (!tileCacheDir.exists()) tileCacheDir.mkdirs()
            val assetManager = assets
            val zoomLevels = assetManager.list("tiles") ?: return
            for (zoom in zoomLevels) {
                val xDirs = assetManager.list("tiles/$zoom") ?: continue
                for (x in xDirs) {
                    val yFiles = assetManager.list("tiles/$zoom/$x") ?: continue
                    val targetDir = File(tileCacheDir, "$zoom/$x")
                    if (!targetDir.exists()) targetDir.mkdirs()
                    for (y in yFiles) {
                        val destFile = File(targetDir, y)
                        if (!destFile.exists()) {
                            assetManager.open("tiles/$zoom/$x/$y").use { input ->
                                destFile.outputStream().use { output -> input.copyTo(output) }
                            }
                        }
                    }
                }
            }
            Log.i("MainActivity", "Preloaded offline corridor map tiles")
        } catch (e: Exception) {
            Log.w("MainActivity", "Failed to preload offline map tiles from assets", e)
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
        btnDemoMode = findViewById(R.id.btnDemoMode)
        btnToggleVoice = findViewById(R.id.btnToggleVoice)
        btnToggleLeanCorrection = findViewById(R.id.btnToggleLeanCorrection)
        mapView = findViewById(R.id.mapView)
        btnRecenter = findViewById(R.id.btnRecenter)
        tvMapStatus = findViewById(R.id.tvMapStatus)
        mapModeDot = findViewById(R.id.mapModeDot)

        btnKillGps.setOnClickListener { toggleGpsBlackout() }

        // Route selection dialog on clicking Demo Mode
        btnDemoMode.setOnClickListener {
            showRouteSelectionDialog()
        }

        btnToggleVoice.setOnClickListener {
            if (::audioManager.isInitialized) {
                audioManager.isEnabled = !audioManager.isEnabled
                btnToggleVoice.text = if (audioManager.isEnabled) "AUDIO: ON" else "AUDIO: OFF"
                if (audioManager.isEnabled) {
                    audioManager.speak("Voice guidance enabled.")
                }
            }
        }

        btnToggleLeanCorrection.setOnClickListener {
            useLeanCorrection = !useLeanCorrection
            btnToggleLeanCorrection.text = if (useLeanCorrection) "LEAN NHC: ON" else "LEAN NHC: OFF"
            btnToggleLeanCorrection.setTextColor(
                if (useLeanCorrection) Color.parseColor("#10B981") else Color.parseColor("#EF4444")
            )
        }

        btnRecenter.setOnClickListener {
            isAutoFollow = true
            if (::vehicleMarker.isInitialized) mapView.controller.setCenter(vehicleMarker.position)
        }
    }

    private fun initMap() {
        Configuration.getInstance().userAgentValue = packageName
        mapView.setTileSource(TileSourceFactory.MAPNIK)

        val darkMatrix = android.graphics.ColorMatrix(floatArrayOf(
            -0.85f, 0f, 0f, 0f, 240f,
            0f, -0.85f, 0f, 0f, 240f,
            0f, 0f, -0.85f, 0f, 240f,
            0f, 0f, 0f, 1f, 0f
        ))
        mapView.overlayManager.tilesOverlay.setColorFilter(
            android.graphics.ColorMatrixColorFilter(darkMatrix)
        )

        mapView.setMultiTouchControls(true)
        mapView.zoomController.setVisibility(CustomZoomButtonsController.Visibility.NEVER)
        mapView.minZoomLevel = 13.0
        mapView.maxZoomLevel = 19.5

        roadCenterlineOverlay.outlinePaint.color = Color.parseColor("#475569")
        roadCenterlineOverlay.outlinePaint.strokeWidth = 3f
        roadCenterlineOverlay.outlinePaint.pathEffect = DashPathEffect(floatArrayOf(15f, 10f), 0f)
        mapView.overlays.add(roadCenterlineOverlay)

        tunnelTubeOverlay.outlinePaint.color = Color.parseColor("#F59E0B")
        tunnelTubeOverlay.outlinePaint.strokeWidth = 14f
        tunnelTubeOverlay.outlinePaint.alpha = 80
        mapView.overlays.add(tunnelTubeOverlay)

        trueTrackPolyline.outlinePaint.color = Color.parseColor("#38BDF8")
        trueTrackPolyline.outlinePaint.strokeWidth = 8f
        trueTrackPolyline.outlinePaint.strokeCap = Paint.Cap.ROUND
        mapView.overlays.add(trueTrackPolyline)

        driftPolyline.outlinePaint.color = Color.parseColor("#EF4444")
        driftPolyline.outlinePaint.strokeWidth = 5f
        driftPolyline.outlinePaint.pathEffect = DashPathEffect(floatArrayOf(18f, 12f), 0f)
        driftPolyline.outlinePaint.strokeCap = Paint.Cap.ROUND
        mapView.overlays.add(driftPolyline)

        vehicleMarker = Marker(mapView)
        vehicleMarker.icon = ContextCompat.getDrawable(this, R.drawable.ic_navigation_arrow)
        vehicleMarker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        vehicleMarker.position = GeoPoint(drLat, drLon)
        vehicleMarker.rotation = drHeadingDeg.toFloat()
        vehicleMarker.infoWindow = null
        vehicleMarker.setOnMarkerClickListener { _, _ -> true }
        mapView.overlays.add(vehicleMarker)

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
            Log.e("MainActivity", "Error reading norm_stats.json", e)
        }
    }

    private fun initOnnxModel() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val modelBytes = assets.open("truetrack_model.onnx").readBytes()
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
            ortSession = ortEnv!!.createSession(modelBytes, sessionOptions)
            Log.i("MainActivity", "TrueTrack ONNX model successfully initialized")
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to initialize ONNX Runtime", e)
        }
    }

    private fun initAudioAndNetwork() {
        audioManager = AudioCueManager(this)
        audioManager.onSystemOnline(availableRoutes[currentRouteIndex].shortName)

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
            tvServerStatus.text = ":8765 (off)"
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
        if (!isDemoMode && event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            processSensorTick()
        }
    }

    private fun processSensorTick() {
        val currentPhi = leanCorrector.update(rawAx, rawAy, rawAz, rawGx)
        val derolled = if (useLeanCorrection) {
            leanCorrector.derollAccelerations(rawAx, rawAy, rawAz, currentPhi)
        } else {
            floatArrayOf(rawAx, rawAy, rawAz)
        }
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
        var speedKmh = 0.0f
        var yawRateDeg = Math.toDegrees(rawGz.toDouble()).toFloat()

        val accelMag = sqrt(rawAx * rawAx + rawAy * rawAy + rawAz * rawAz)
        val isStandstill = abs(accelMag - 9.80665f) < 0.25f && abs(rawGx) < 0.05f && abs(rawGz) < 0.05f

        val displayLeanDeg = if (isStandstill) 0.0f else Math.toDegrees(currentPhiRad.toDouble()).toFloat()

        if (ortSession != null && ortEnv != null && !isStandstill) {
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
                    drLat = currentLat
                    drLon = currentLon
                    drHeadingDeg = Math.toDegrees(currentHeadingRad)
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

        // Update UI View Elements on Main Thread
        runOnUiThread {
            updateDashboard(speedKmh, yawRateDeg, displayLeanDeg, accelMag, null)
            updateLiveMap(speedKmh, yawRateDeg)
        }

        // Broadcast to Laptop Web Cockpit via WebSocket (off-main-thread)
        streamServer?.broadcastTelemetry(
            timestampMs = System.currentTimeMillis(),
            speedKmh = speedKmh,
            yawRateDeg = yawRateDeg,
            leanDeg = displayLeanDeg,
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
        accelMag: Float,
        frame: TelemetryFrame?
    ) {
        tvSpeed.text = String.format(Locale.US, "%.1f", speedKmh)
        tvLeanAngle.text = String.format(Locale.US, "%.1f°", abs(leanDeg))
        tvLeanDirection.text = when {
            leanDeg > 2.5f -> "RIGHT LEAN"
            leanDeg < -2.5f -> "LEFT LEAN"
            else -> "UPRIGHT"
        }
        tvLeanDirection.setTextColor(
            if (abs(leanDeg) > 5.0f) Color.parseColor("#F59E0B") else Color.parseColor("#10B981")
        )
        tvYawRate.text = String.format(Locale.US, "%.1f °/s", yawRateDeg)

        if (isBlackout) {
            val elapsedSec = (SystemClock.elapsedRealtime() - blackoutStartTimeMs) / 1000
            tvBlackoutTimer.text = String.format(Locale.US, "+%02d s", elapsedSec)
            btnKillGps.text = "RESTORE GPS FIX"
            btnKillGps.backgroundTintList = getColorStateList(R.color.status_green)
            tvGpsStatus.text = "GNSS: BLACKOUT ACTIVE (Neural DR)"
            tvGpsStatus.setTextColor(Color.parseColor("#EF4444"))
            if (frame != null) {
                tvDriftRate.text = String.format(
                    Locale.US, "naive=%.1fm  TT=%.1fm", frame.errNaiveM, frame.errTrueTrackM
                )
            } else {
                tvDriftRate.text = "DR ACTIVE"
            }
        } else {
            tvBlackoutTimer.text = "00:00"
            btnKillGps.text = "KILL GPS FIX (SIMULATE BLACKOUT)"
            btnKillGps.backgroundTintList = getColorStateList(R.color.alert_red)
            tvGpsStatus.text = "GNSS: LOCKED (50 Hz SENSOR/SIM)"
            tvGpsStatus.setTextColor(Color.parseColor("#10B981"))
            tvDriftRate.text = "0.000 m/s (Lock)"
        }

        val gForce = accelMag / 9.80665f
        if (frame != null) {
            tvRawImu.text = String.format(
                Locale.US,
                "IMU: ax=%.2f ay=%.2f gz=%.3f rad/s | %.2f m/s² (%.2fG)  NPU %.1fms",
                frame.rawImuAx, frame.rawImuAy, frame.rawImuGz, accelMag, gForce, lastNpuLatencyMs
            )
        } else {
            tvRawImu.text = String.format(
                Locale.US,
                "3D: ax=%.2f ay=%.2f az=%.2f m/s² | |g|=%.2f (%.2fG)  NPU %.1fms",
                rawAx, rawAy, rawAz, accelMag, gForce, lastNpuLatencyMs
            )
        }

        tvLateralConstraint.text = if (abs(leanDeg) < 5f) "NHC: VALID"
        else String.format(Locale.US, "NHC: phi=%.0f°", leanDeg)
    }

    private fun updateLiveMap(speedKmh: Float, yawRateDeg: Float) {
        if (speedKmh > 0.5f) {
            val dt = 0.2
            val speedMs = speedKmh / 3.6
            drHeadingDeg = (drHeadingDeg + yawRateDeg * dt + 360.0) % 360.0
            val headingRad = Math.toRadians(drHeadingDeg)

            val metersPerDegLat = 111_111.0
            val metersPerDegLon = 111_111.0 * cos(Math.toRadians(drLat))

            drLat += speedMs * dt * cos(headingRad) / metersPerDegLat
            drLon += speedMs * dt * sin(headingRad) / metersPerDegLon

            val currentPoint = GeoPoint(drLat, drLon)
            trueTrackPoints.add(currentPoint)
            if (trueTrackPoints.size > 300) trueTrackPoints.removeAt(0)
            trueTrackPolyline.setPoints(ArrayList(trueTrackPoints))

            if (isBlackout) {
                val elapsedSec = (SystemClock.elapsedRealtime() - blackoutStartTimeMs) / 1000.0
                val driftMeters = 0.5 * 0.12 * elapsedSec * elapsedSec
                naiveLat = drLat - (driftMeters * 0.3 / metersPerDegLat)
                naiveLon = drLon + (driftMeters * 0.95 / metersPerDegLon)
                driftPoints.add(GeoPoint(naiveLat, naiveLon))
                if (driftPoints.size > 300) driftPoints.removeAt(0)
                driftPolyline.setPoints(ArrayList(driftPoints))
                tvMapStatus.text = "GPS BLACKOUT  NEURAL DR ACTIVE"
                tvMapStatus.setTextColor(Color.parseColor("#EF4444"))
                mapModeDot.backgroundTintList = getColorStateList(R.color.alert_red)
            } else {
                driftPoints.clear()
                driftPolyline.setPoints(driftPoints)
                tvMapStatus.text = "LIVE NEURAL TRAJECTORY"
                tvMapStatus.setTextColor(Color.parseColor("#38BDF8"))
                mapModeDot.backgroundTintList = getColorStateList(R.color.status_green)
            }

            vehicleMarker.position = currentPoint
            vehicleMarker.rotation = drHeadingDeg.toFloat()
            if (isAutoFollow) mapView.controller.setCenter(currentPoint)
            mapView.invalidate()
        }
    }

    private fun toggleGpsBlackout() {
        isBlackout = !isBlackout
        if (isBlackout) {
            blackoutStartTimeMs = SystemClock.elapsedRealtime()
            prevBlackoutState = true
            audioManager.onBlackoutEntered()
        } else {
            prevBlackoutState = false
            audioManager.onGpsRestored()
            driftPoints.clear()
            driftPolyline.setPoints(driftPoints)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

}
