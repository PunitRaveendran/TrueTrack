package com.truetrack.navigation

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.content.Intent
import android.net.Uri
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.net.wifi.WifiManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.PowerManager
import android.provider.Settings
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

/**
 * Data frame for corridor flight-recorder simulation.
 */
data class TelemetryFrame(
    val t: Float,
    val speedKmh: Float,
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
    val ax: Float,
    val ay: Float,
    val gz: Float
)

/**
 * TrueTrack - Android Navigation Instrument Activity
 *
 * Runs high-rate 50 Hz sensor acquisition, continuous roll-lean estimation,
 * local Snapdragon NPU inference, dual-screen telemetry broadcasting,
 * and high-contrast dark-mode OpenStreetMap real-time dead reckoning.
 */
class MainActivity : AppCompatActivity(), SensorEventListener {

    // Sensor Hardware
    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null
    private var locationManager: LocationManager? = null
    private var locationUpdatesRegistered = false
    private var gpsReacquisitionFixCount = 0
    private var gpsReacquisitionCandidateLat = 0.0
    private var gpsReacquisitionCandidateLon = 0.0
    private var gpsReacquisitionCandidateTimeNs = 0L
    private val gpsDistanceResult = FloatArray(1)

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
    private var lastNpuLatencyMs = 0.0f

    // Simulation Data & Playback Engine
    private val simulationFrames = mutableListOf<TelemetryFrame>()
    private var simFrameIndex = 350 // Start 3 seconds before underpass blackout (t=35s)
    private var isDemoMode = true    // DEFAULT TO SIMULATION for reliable demonstration

    // Dedicated Background Workers & Concurrency Isolation
    private var sensorThread: HandlerThread? = null
    private var sensorHandler: Handler? = null
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val isInferring = AtomicBoolean(false)

    // UI View References
    private lateinit var tvSpeed: TextView
    private lateinit var tvNpuBadge: TextView
    private lateinit var tvLeanAngle: TextView
    private lateinit var tvLeanDirection: TextView
    private lateinit var tvYawRate: TextView
    private lateinit var tvLateralConstraint: TextView
    private lateinit var tvGpsStatus: TextView
    private lateinit var tvBlackoutTimer: TextView
    private lateinit var tvServerStatus: TextView
    private lateinit var tvRawImu: TextView
    private lateinit var btnKillGps: Button
    private lateinit var btnDemoMode: Button
    private lateinit var btnToggleVoice: Button
    private lateinit var btnToggleLeanCorrection: Button

    // Map & Dead Reckoning Navigation View
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

    // Real HITEC City Corridor Waypoints (from OSM GeoJSON ground truth)
    private val corridorWaypoints = listOf(
        GeoPoint(17.441434, 78.377145), // Raidurg Metro / Mindspace Junction
        GeoPoint(17.442195, 78.377136),
        GeoPoint(17.443139, 78.377118),
        GeoPoint(17.443514, 78.377107), // Approach straight
        GeoPoint(17.444201, 78.377099),
        GeoPoint(17.444752, 78.377147),
        GeoPoint(17.445257, 78.377232),
        GeoPoint(17.445842, 78.377314),
        GeoPoint(17.446399, 78.377473),
        GeoPoint(17.446842, 78.377733),
        GeoPoint(17.447195, 78.377944), // Tunnel Entry (Blackout Start)
        GeoPoint(17.447646, 78.378233),
        GeoPoint(17.448207, 78.378708),
        GeoPoint(17.448857, 78.379075),
        GeoPoint(17.449330, 78.379330),
        GeoPoint(17.450195, 78.379409),
        GeoPoint(17.450281, 78.379450), // Tunnel Exit (GPS Restored)
        GeoPoint(17.450680, 78.379676),
        GeoPoint(17.451327, 78.380608)  // Cyber Towers Junction
    )

    // Tunnel Sub-Segment
    private val tunnelWaypoints = corridorWaypoints.subList(10, 17)

    private val liveLocationListener = object : LocationListener {
        override fun onLocationChanged(location: Location) {
            if (isDemoMode || isBlackout) return

            if (blackoutStartTimeMs > 0L) {
                val elapsedSinceBlackoutSec =
                    ((SystemClock.elapsedRealtime() - blackoutStartTimeMs).coerceAtLeast(0L) / 1000.0)
                val maxFromDrMeters = 20.0 + elapsedSinceBlackoutSec * 45.0
                Location.distanceBetween(drLat, drLon, location.latitude, location.longitude, gpsDistanceResult)
                if (gpsDistanceResult[0] > maxFromDrMeters) {
                    gpsReacquisitionFixCount = 0
                    Log.w("MainActivity", "Rejecting GPS reacquisition fix outside DR speed bound")
                    return
                }

                if (gpsReacquisitionFixCount > 0) {
                    Location.distanceBetween(
                        gpsReacquisitionCandidateLat,
                        gpsReacquisitionCandidateLon,
                        location.latitude,
                        location.longitude,
                        gpsDistanceResult
                    )
                    val intervalSec =
                        ((location.elapsedRealtimeNanos - gpsReacquisitionCandidateTimeNs).coerceAtLeast(0L) / 1_000_000_000.0)
                    val maxBetweenFixesMeters = 8.0 + intervalSec * 45.0
                    if (gpsDistanceResult[0] > maxBetweenFixesMeters) {
                        gpsReacquisitionFixCount = 0
                    }
                }

                gpsReacquisitionFixCount++
                gpsReacquisitionCandidateLat = location.latitude
                gpsReacquisitionCandidateLon = location.longitude
                gpsReacquisitionCandidateTimeNs = location.elapsedRealtimeNanos
                if (gpsReacquisitionFixCount < 3) return

                blackoutStartTimeMs = 0L
                gpsReacquisitionFixCount = 0
                audioManager.onGpsRestored()
                driftPoints.clear()
                driftPolyline.setPoints(driftPoints)
            }

            drLat = location.latitude
            drLon = location.longitude
            if (location.hasBearing()) {
                drHeadingDeg = ((location.bearing.toDouble() % 360.0) + 360.0) % 360.0
            }
        }

        @Deprecated("Deprecated in Android")
        override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) = Unit

        override fun onProviderEnabled(provider: String) = Unit
        override fun onProviderDisabled(provider: String) = Unit
    }
    // Current State Coordinates
    private var drLat = 17.443514
    private var drLon = 78.377107
    private var drHeadingDeg = 359.0
    private var naiveLat = 17.443514
    private var naiveLon = 78.377107
    private var naiveVelocityEastMps = 0.0
    private var naiveVelocityNorthMps = 0.0
    private var naiveHeadingDeg = 359.0
    private var naiveLastUpdateNanos = 0L
    private var lastLiveSpeedKmh = 0.0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        requestLocationPermissions()
        startNavigationServiceIfPermitted()
        updateGpsSubscription()
        preloadOfflineTiles()
        loadSimulationTelemetry()
        initMap()
        loadNormalizationStats()
        initOnnxModel()
        initAudioAndNetwork()
        initSensors()
    }

    private fun startNavigationServiceIfPermitted() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            ContextCompat.startForegroundService(this, Intent(this, NavigationForegroundService::class.java))
            requestBatteryOptimizationExemption()
        }
    }

    private fun requestBatteryOptimizationExemption() {
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        if (!powerManager.isIgnoringBatteryOptimizations(packageName)) {
            try {
                startActivity(Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:$packageName")))
            } catch (e: Exception) {
                Log.w("MainActivity", "Battery optimization exemption request unavailable", e)
            }
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1001 && grantResults.any { it == PackageManager.PERMISSION_GRANTED }) {
            startNavigationServiceIfPermitted()
            updateGpsSubscription()
        }
    }
    private fun updateGpsSubscription() {
        if (isDemoMode || isBlackout) {
            if (locationUpdatesRegistered) {
                locationManager?.removeUpdates(liveLocationListener)
                locationUpdatesRegistered = false
            }
            return
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) return

        val manager = locationManager ?: (getSystemService(Context.LOCATION_SERVICE) as LocationManager).also { locationManager = it }
        try {
            if (!locationUpdatesRegistered && manager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                manager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 0f, liveLocationListener)
                locationUpdatesRegistered = true
            }
        } catch (e: SecurityException) {
            Log.w("MainActivity", "Location permission unavailable for live updates", e)
        } catch (e: IllegalArgumentException) {
            Log.w("MainActivity", "GPS provider unavailable", e)
        }
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
            if (!tileCacheDir.exists()) {
                tileCacheDir.mkdirs()
            }
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
                                destFile.outputStream().use { output ->
                                    input.copyTo(output)
                                }
                            }
                        }
                    }
                }
            }
            Log.i("MainActivity", "Preloaded offline corridor map tiles into local cache")
        } catch (e: Exception) {
            Log.w("MainActivity", "Failed to preload offline map tiles from assets", e)
        }
    }

    private fun loadSimulationTelemetry() {
        try {
            val jsonString = assets.open("corridor_telemetry.json").bufferedReader().use { it.readText() }
            val jsonArray = JSONArray(jsonString)
            simulationFrames.clear()
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                val gt = obj.getJSONArray("gt")
                val naive = obj.getJSONArray("naive")
                val tt = obj.getJSONArray("truetrack")
                simulationFrames.add(
                    TelemetryFrame(
                        t = obj.getDouble("t").toFloat(),
                        speedKmh = obj.getDouble("speed_kmh").toFloat(),
                        headingDeg = obj.getDouble("heading_deg").toFloat(),
                        isBlackout = obj.getInt("is_blackout") == 1,
                        gtLat = gt.getDouble(0),
                        gtLon = gt.getDouble(1),
                        naiveLat = naive.getDouble(0),
                        naiveLon = naive.getDouble(1),
                        trueTrackLat = tt.getDouble(0),
                        trueTrackLon = tt.getDouble(1),
                        errNaiveM = obj.getDouble("err_naive_m").toFloat(),
                        errTrueTrackM = obj.getDouble("err_truetrack_m").toFloat(),
                        ax = obj.optDouble("raw_imu_ax", 0.0).toFloat(),
                        ay = obj.optDouble("raw_imu_ay", 0.0).toFloat(),
                        gz = obj.optDouble("raw_imu_gz", 0.0).toFloat()
                    )
                )
            }
            Log.i("MainActivity", "Loaded ${simulationFrames.size} simulation telemetry frames")
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to load corridor_telemetry.json", e)
        }
    }

    private fun initViews() {
        tvSpeed = findViewById(R.id.tvSpeed)
        tvNpuBadge = findViewById(R.id.tvNpuBadge)
        tvLeanAngle = findViewById(R.id.tvLeanAngle)
        tvLeanDirection = findViewById(R.id.tvLeanDirection)
        tvYawRate = findViewById(R.id.tvYawRate)
        tvLateralConstraint = findViewById(R.id.tvLateralConstraint)
        tvGpsStatus = findViewById(R.id.tvGpsStatus)
        tvBlackoutTimer = findViewById(R.id.tvBlackoutTimer)
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

        // Start with SIMULATION active
        btnDemoMode.text = "MODE: SIMULATION"
        btnDemoMode.setTextColor(Color.parseColor("#38BDF8"))

        btnDemoMode.setOnClickListener {
            isDemoMode = !isDemoMode
            trueTrackPoints.clear()
            driftPoints.clear()
            trueTrackPolyline.setPoints(trueTrackPoints)
            driftPolyline.setPoints(driftPoints)

            if (isDemoMode) {
                simFrameIndex = 350
                btnDemoMode.text = "MODE: SIMULATION"
                btnDemoMode.setTextColor(Color.parseColor("#38BDF8"))
            } else {
                btnDemoMode.text = "MODE: LIVE SENSOR"
                btnDemoMode.setTextColor(Color.parseColor("#94A3B8"))
                isBlackout = false
                blackoutStartTimeMs = 0L
                gpsReacquisitionFixCount = 0
                naiveLat = drLat
                naiveLon = drLon
                naiveVelocityEastMps = 0.0
                naiveVelocityNorthMps = 0.0
                naiveHeadingDeg = drHeadingDeg
            }
            updateGpsSubscription()
        }

        btnToggleVoice.setOnClickListener {
            audioManager.isEnabled = !audioManager.isEnabled
            btnToggleVoice.text = if (audioManager.isEnabled) "AUDIO: ON" else "AUDIO: OFF"
        }

        btnToggleLeanCorrection.setOnClickListener {
            useLeanCorrection = !useLeanCorrection
            btnToggleLeanCorrection.text = if (useLeanCorrection) "LEAN NHC: ON" else "LEAN NHC: OFF"
            btnToggleLeanCorrection.setTextColor(if (useLeanCorrection) Color.parseColor("#10B981") else Color.parseColor("#EF4444"))
        }

        btnRecenter.setOnClickListener {
            isAutoFollow = true
            mapView.controller.setCenter(vehicleMarker.position)
        }
    }

    private fun initMap() {
        Configuration.getInstance().userAgentValue = packageName

        // Official OpenStreetMap tile source (Zero API key required)
        mapView.setTileSource(TileSourceFactory.MAPNIK)

        // Dark navigation HUD color filter (transforms standard OSM tiles into Google Maps night mode)
        val darkMatrix = android.graphics.ColorMatrix(floatArrayOf(
            -0.85f, 0f, 0f, 0f, 240f,
            0f, -0.85f, 0f, 0f, 240f,
            0f, 0f, -0.85f, 0f, 240f,
            0f, 0f, 0f, 1f, 0f
        ))
        mapView.overlayManager.tilesOverlay.setColorFilter(android.graphics.ColorMatrixColorFilter(darkMatrix))

        mapView.setMultiTouchControls(true)
        mapView.zoomController.setVisibility(CustomZoomButtonsController.Visibility.NEVER)
        mapView.minZoomLevel = 14.0
        mapView.maxZoomLevel = 19.0
        mapView.controller.setZoom(16.5)

        val startPoint = GeoPoint(drLat, drLon)
        mapView.controller.setCenter(startPoint)

        // 1. Road Centerline Overlay (Thin white dashed line showing the real street corridor)
        roadCenterlineOverlay.outlinePaint.color = Color.parseColor("#475569")
        roadCenterlineOverlay.outlinePaint.strokeWidth = 3f
        roadCenterlineOverlay.outlinePaint.pathEffect = DashPathEffect(floatArrayOf(15f, 10f), 0f)
        roadCenterlineOverlay.setPoints(corridorWaypoints)
        mapView.overlays.add(roadCenterlineOverlay)

        // 2. Underpass Tunnel Highlight (Amber illuminated tube indicating the blackout zone)
        tunnelTubeOverlay.outlinePaint.color = Color.parseColor("#F59E0B")
        tunnelTubeOverlay.outlinePaint.strokeWidth = 14f
        tunnelTubeOverlay.outlinePaint.alpha = 80
        tunnelTubeOverlay.setPoints(tunnelWaypoints)
        mapView.overlays.add(tunnelTubeOverlay)

        // 3. TrueTrack Neural Dead-Reckoning Polyline: Glowing Electric Blue
        trueTrackPolyline.outlinePaint.color = Color.parseColor("#38BDF8")
        trueTrackPolyline.outlinePaint.strokeWidth = 8f
        trueTrackPolyline.outlinePaint.strokeCap = Paint.Cap.ROUND
        mapView.overlays.add(trueTrackPolyline)

        // 4. Naive uncorrected IMU Drift during Blackout: Alert Red (Dashed)
        driftPolyline.outlinePaint.color = Color.parseColor("#EF4444")
        driftPolyline.outlinePaint.strokeWidth = 5f
        driftPolyline.outlinePaint.pathEffect = DashPathEffect(floatArrayOf(18f, 12f), 0f)
        driftPolyline.outlinePaint.strokeCap = Paint.Cap.ROUND
        mapView.overlays.add(driftPolyline)

        // 5. Vehicle Navigation Arrow Marker (With info window disabled so no white popup appears)
        vehicleMarker = Marker(mapView)
        vehicleMarker.icon = ContextCompat.getDrawable(this, R.drawable.ic_navigation_arrow)
        vehicleMarker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        vehicleMarker.position = startPoint
        vehicleMarker.rotation = drHeadingDeg.toFloat()
        vehicleMarker.infoWindow = null
        vehicleMarker.setOnMarkerClickListener { _, _ -> true } // Consume click: no popup!
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

        // Process frame on accelerometer tick (50 Hz)
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            processSensorTick()
        }
    }

    private fun processSensorTick() {
        // 1. Update 50 Hz Roll Lean Estimator phi(t) using corrected phone coordinates
        val currentPhi = leanCorrector.update(rawAx, rawAy, rawAz, rawGx)

        // 2. De-roll Accelerations via R_x(-phi)
        val derolled = if (useLeanCorrection) {
            leanCorrector.derollAccelerations(rawAx, rawAy, rawAz, currentPhi)
        } else {
            floatArrayOf(rawAx, rawAy, rawAz)
        }

        val axIn = derolled[0]
        val ayIn = derolled[1]
        val azIn = derolled[2]

        // 3. Push to 6-axis rolling window ring buffer
        imuRingBuffer[0][bufferHead] = (axIn - normMeans[0]) / normStds[0]
        imuRingBuffer[1][bufferHead] = (ayIn - normMeans[1]) / normStds[1]
        imuRingBuffer[2][bufferHead] = (azIn - normMeans[2]) / normStds[2]
        imuRingBuffer[3][bufferHead] = (rawGx - normMeans[3]) / normStds[3]
        imuRingBuffer[4][bufferHead] = (rawGy - normMeans[4]) / normStds[4]
        imuRingBuffer[5][bufferHead] = (rawGz - normMeans[5]) / normStds[5]

        bufferHead = (bufferHead + 1) % windowSize
        samplesCollected++

        // Run inference or simulation step every 10 samples (5 Hz / 100ms)
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
        // 3D Acceleration Magnitude & G-Force from live sensor
        val accelMag = sqrt(rawAx * rawAx + rawAy * rawAy + rawAz * rawAz)
        val gyroMag = sqrt(rawGx * rawGx + rawGy * rawGy + rawGz * rawGz)
        val isStationary = abs(accelMag - 9.80665f) < 0.65f && gyroMag < 0.18f

        var speedKmh = 0.0f
        var yawRateDeg = 0.0f
        var displayLeanDeg = Math.toDegrees(currentPhiRad.toDouble()).toFloat()

        if (isDemoMode && simulationFrames.isNotEmpty()) {
            // ==========================================
            // FLIGHT RECORDER CORRIDOR SIMULATION MODE
            // ==========================================
            val frame = simulationFrames[simFrameIndex]
            simFrameIndex = (simFrameIndex + 1) % simulationFrames.size

            speedKmh = frame.speedKmh
            yawRateDeg = Math.toDegrees(frame.gz.toDouble()).toFloat()

            // Lean angle: responsive to live phone tilt if user tilts phone, otherwise follows vehicle yaw
            val liveLean = Math.toDegrees(currentPhiRad.toDouble()).toFloat()
            displayLeanDeg = if (abs(liveLean) > 3.0f) liveLean else (yawRateDeg * 0.40f).coerceIn(-35f, 35f)

            drLat = frame.trueTrackLat
            drLon = frame.trueTrackLon
            drHeadingDeg = frame.headingDeg.toDouble()
            naiveLat = frame.naiveLat
            naiveLon = frame.naiveLon

            // Trigger blackout state according to corridor run or manual toggle
            val activeBlackout = frame.isBlackout || isBlackout
            if (activeBlackout && !isBlackout) {
                isBlackout = true
                blackoutStartTimeMs = SystemClock.elapsedRealtime()
                audioManager.onBlackoutEntered()
            }

            lastNpuLatencyMs = 0.0f
        } else if (!isStationary && ortSession != null && ortEnv != null) {
            // ==========================================
            // LIVE SENSOR MODE (Movement detected)
            // ==========================================
            try {
                val tensorBuffer = FloatBuffer.wrap(snapshotBuffer)
                val inputShape = longArrayOf(1, numChannels.toLong(), windowSize.toLong())
                val inputTensor = OnnxTensor.createTensor(ortEnv, tensorBuffer, inputShape)
                val inputs = mapOf(ortSession!!.inputNames.first() to inputTensor)

                // ortSession.run is strictly serialized on inferenceExecutor (zero thread races)
                val inferenceStartNs = System.nanoTime()
                val results = ortSession!!.run(inputs)
                lastNpuLatencyMs = (System.nanoTime() - inferenceStartNs) / 1_000_000.0f
                val output = (results[0].value as Array<FloatArray>)[0]

                val forwardSpeedMs = max(0.0f, output[0])
                val predYawRateRad = output[1]

                speedKmh = forwardSpeedMs * 3.6f
                yawRateDeg = Math.toDegrees(predYawRateRad.toDouble()).toFloat()

                inputTensor.close()
                results.close()
            } catch (e: Exception) {
                lastNpuLatencyMs = 0.0f
                Log.e("MainActivity", "NPU inference error: ${e.message}")
            }
        } else {
            // ==========================================
            // LIVE SENSOR MODE (Stationary Zero-Velocity Update)
            // ==========================================
            speedKmh = 0.0f
            yawRateDeg = 0.0f
            lastNpuLatencyMs = 0.0f
        }

        // Update UI & Map on Main Thread
        runOnUiThread {
            updateDashboard(speedKmh, yawRateDeg, displayLeanDeg, accelMag)
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
            az = azIn
        )
    }

    private fun updateDashboard(
        speedKmh: Float,
        yawRateDeg: Float,
        leanDeg: Float,
        accelMag: Float
    ) {
        lastLiveSpeedKmh = speedKmh
        tvSpeed.text = String.format(Locale.US, "%.1f", speedKmh)
        tvNpuBadge.text = if (lastNpuLatencyMs > 0.0f) {
            String.format(Locale.US, "ONNX %.2f ms", lastNpuLatencyMs)
        } else {
            "ONNX: N/A"
        }
        tvLeanAngle.text = String.format(Locale.US, "%.1f°", abs(leanDeg))
        tvLeanDirection.text = when {
            leanDeg > 2.5f -> "RIGHT LEAN (R_x De-rolled)"
            leanDeg < -2.5f -> "LEFT LEAN (R_x De-rolled)"
            else -> "UPRIGHT (φ ≈ 0°)"
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
        } else {
            tvBlackoutTimer.text = "00:00"
            btnKillGps.text = "KILL GPS FIX (SIMULATE BLACKOUT)"
            btnKillGps.backgroundTintList = getColorStateList(R.color.alert_red)
            tvGpsStatus.text = "GNSS: LOCKED (50 Hz SENSOR/SIM)"
            tvGpsStatus.setTextColor(Color.parseColor("#10B981"))
        }

        // 3D Acceleration, decomposed vector axes, and total gravity magnitude
        val gForce = accelMag / 9.80665f
        tvRawImu.text = String.format(
            Locale.US,
            "3D ACCEL [m/s²]: ax=%.2f (lat) ay=%.2f (long) az=%.2f (vert)\n|g|=%.2f m/s² (%.2fG) | NPU: %.1fms",
            rawAx, rawAy, rawAz, accelMag, gForce, lastNpuLatencyMs
        )

        // Update real-time map dead reckoning
        updateMap(speedKmh, yawRateDeg)
    }

    private fun updateMap(speedKmh: Float, yawRateDeg: Float) {
        val dt = 0.1

        if (isDemoMode) {
            val currentPoint = GeoPoint(drLat, drLon)
            trueTrackPoints.add(currentPoint)
            if (trueTrackPoints.size > 250) trueTrackPoints.removeAt(0)
            trueTrackPolyline.setPoints(trueTrackPoints)

            if (isBlackout) {
                driftPoints.add(GeoPoint(naiveLat, naiveLon))
                if (driftPoints.size > 250) driftPoints.removeAt(0)
                driftPolyline.setPoints(driftPoints)

                tvMapStatus.text = "GPS BLACKOUT • NEURAL DR ACTIVE"
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

            if (isAutoFollow) {
                mapView.controller.setCenter(currentPoint)
            }
            mapView.invalidate()
            return
        }

        // LIVE SENSOR MODE:
        if (speedKmh > 0.5f) {
            val speedMs = speedKmh / 3.6
            drHeadingDeg = (drHeadingDeg + yawRateDeg * dt) % 360.0
            val headingRad = Math.toRadians(drHeadingDeg)

            // Approximate 0.000009 deg/m at ~17.4° latitude
            val deltaLat = speedMs * dt * cos(headingRad) * 0.000009
            val deltaLon = speedMs * dt * sin(headingRad) * 0.000009

            drLat += deltaLat
            drLon += deltaLon
            val currentPoint = GeoPoint(drLat, drLon)

            trueTrackPoints.add(currentPoint)
            if (trueTrackPoints.size > 200) trueTrackPoints.removeAt(0)
            trueTrackPolyline.setPoints(trueTrackPoints)

            if (isBlackout) {
                val nowNs = SystemClock.elapsedRealtimeNanos()
                val naiveDt = if (naiveLastUpdateNanos == 0L) 0.0 else
                    ((nowNs - naiveLastUpdateNanos).coerceAtLeast(0L) / 1_000_000_000.0).coerceAtMost(0.5)
                naiveLastUpdateNanos = nowNs
                naiveHeadingDeg = (naiveHeadingDeg + Math.toDegrees(rawGz.toDouble() * naiveDt) + 360.0) % 360.0
                val naiveHeadingRad = Math.toRadians(naiveHeadingDeg)
                val accelEast = -rawAx * sin(naiveHeadingRad) + rawAy * cos(naiveHeadingRad)
                val accelNorth = rawAx * cos(naiveHeadingRad) + rawAy * sin(naiveHeadingRad)
                naiveVelocityEastMps += accelEast * naiveDt
                naiveVelocityNorthMps += accelNorth * naiveDt
                val deltaEastMeters = naiveVelocityEastMps * naiveDt
                val deltaNorthMeters = naiveVelocityNorthMps * naiveDt
                naiveLat += deltaNorthMeters / 111_320.0
                naiveLon += deltaEastMeters / (111_320.0 * cos(Math.toRadians(naiveLat)))
                driftPoints.add(GeoPoint(naiveLat, naiveLon))
                if (driftPoints.size > 200) driftPoints.removeAt(0)
                driftPolyline.setPoints(driftPoints)

                tvMapStatus.text = "GPS BLACKOUT • NEURAL DR ACTIVE"
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

            if (isAutoFollow) {
                val mapCenter = mapView.mapCenter
                val dist = currentPoint.distanceToAsDouble(GeoPoint(mapCenter.latitude, mapCenter.longitude))
                if (dist > 0.6) {
                    mapView.controller.setCenter(currentPoint)
                }
            }
            mapView.invalidate()
        }
    }

    private fun toggleGpsBlackout() {
        isBlackout = !isBlackout
        updateGpsSubscription()
        if (isBlackout) {
            blackoutStartTimeMs = SystemClock.elapsedRealtime()
            gpsReacquisitionFixCount = 0
            naiveLat = drLat
            naiveLon = drLon
            val initialSpeedMps = lastLiveSpeedKmh / 3.6
            val initialHeadingRad = Math.toRadians(drHeadingDeg)
            naiveVelocityEastMps = initialSpeedMps * cos(initialHeadingRad)
            naiveVelocityNorthMps = initialSpeedMps * sin(initialHeadingRad)
            naiveHeadingDeg = drHeadingDeg
            naiveLastUpdateNanos = SystemClock.elapsedRealtimeNanos()
            audioManager.onBlackoutEntered()
        } else {
            if (isDemoMode || blackoutStartTimeMs == 0L) {
                audioManager.onGpsRestored()
                blackoutStartTimeMs = 0L
            }
            driftPoints.clear()
            driftPolyline.setPoints(driftPoints)
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
        locationManager?.removeUpdates(liveLocationListener)
        sensorThread?.quitSafely()
        inferenceExecutor.shutdown()
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




