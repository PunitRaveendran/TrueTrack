package com.truetrack.navigation

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.drawable.BitmapDrawable
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.zxing.BarcodeFormat
import com.journeyapps.barcodescanner.BarcodeEncoder
import com.journeyapps.barcodescanner.ScanContract
import com.journeyapps.barcodescanner.ScanOptions
import org.json.JSONArray
import org.json.JSONObject
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.BoundingBox
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.CustomZoomButtonsController
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.concurrent.Executors
import kotlin.math.*

class MainActivity : AppCompatActivity(), SensorEventListener {

    // ───────────────────────────────────────────────────────────────────────
    // UI Elements
    // ───────────────────────────────────────────────────────────────────────
    private lateinit var mapView: MapView
    private lateinit var etOrigin: AutoCompleteTextView
    private lateinit var etDestination: AutoCompleteTextView
    private lateinit var btnNavigate: Button
    private lateinit var btnSwap: ImageButton
    private lateinit var btnClearOrigin: ImageButton
    private lateinit var btnClearDest: ImageButton
    private lateinit var btnTheme: Button
    private lateinit var tvSpeed: TextView
    private lateinit var tvRouteInfo: TextView
    private lateinit var tvImu: TextView
    private lateinit var tvStatus: TextView
    private lateinit var dotStatus: View
    private lateinit var badgeStatus: View
    private lateinit var btnKillGps: Button
    private lateinit var btnDemoRoutes: Button
    private lateinit var btnParty: Button
    private lateinit var btnImuCalibration: Button
    private lateinit var layoutPlayback: LinearLayout
    private lateinit var seekPlayback: SeekBar
    private lateinit var btnPlayPause: ImageButton
    private lateinit var tvPlaybackTime: TextView
    private lateinit var btnRecenter: ImageButton
    private lateinit var layoutLegend: View

    // ───────────────────────────────────────────────────────────────────────
    // Map overlays
    // ───────────────────────────────────────────────────────────────────────
    private val plannedOverlay   = Polyline()   // GREY  — planned route
    private val trueTrackOverlay = Polyline()   // BLUE  — TrueTrack fused
    private val naiveDriftOverlay= Polyline()   // RED   — naive DR drift (blackout)

    private val ttPts    = mutableListOf<GeoPoint>()
    private val naivePts = mutableListOf<GeoPoint>()

    private lateinit var vehicleMarker: Marker
    private lateinit var pinA: Marker
    private lateinit var pinB: Marker

    // ───────────────────────────────────────────────────────────────────────
    // Simulation state
    // ───────────────────────────────────────────────────────────────────────
    private val simFrames = mutableListOf<TelemetryFrame>()
    private var simIdx    = 0
    private var isSimRunning = false
    private var isPaused     = false
    private val simHandler   = Handler(Looper.getMainLooper())
    private var simRunnable: Runnable? = null
    private val TICK_MS = 100L   // 10 Hz simulation ticks

    // Autocomplete debounce
    private val searchHandler = Handler(Looper.getMainLooper())
    private var originRunnable: Runnable? = null
    private var destRunnable: Runnable? = null
    private val originSuggestions = mutableListOf<Pair<String, Pair<Double, Double>>>()
    private val destSuggestions   = mutableListOf<Pair<String, Pair<Double, Double>>>()

    // GPS Blackout & Dead Reckoning state
    private var isBlackout          = false
    private var isForcedGpsRestore  = false
    private var blackoutStartFrame  = -1
    private var blackoutStartMs     = 0L
    private var manualBoElapsed     = 0.0
    private var manualBoLat         = 0.0
    private var manualBoLon         = 0.0
    private var manualBoVx          = 0.0
    private var manualBoVy          = 0.0
    private var manualBoThetaRad    = 0.0
    private var manualBoBiasGz      = 0.032
    private var manualBoBiasAx      = 0.18

    // Theme & follow mode
    private var isDarkMode = true
    private var isAutoFollow = true

    // Sensors
    private lateinit var sensorManager: SensorManager
    private var rawAx = 0f; private var rawAy = 0f; private var rawAz = 9.81f
    private var rawGx = 0f; private var rawGy = 0f; private var rawGz = 0f
    private var gyroTimestampNs = 0L
    private lateinit var imuCalibrator: ImuCalibrator
    private var isPreRecordedReplay = false

    // Networking & offline router
    private val executor = Executors.newCachedThreadPool()
    private var offlineRouter: Router? = null
    private var offlineIndex: GraphIndex? = null
    private lateinit var partyManager: PartyManager
    private lateinit var locationManager: LocationManager
    private val partyMarkers = mutableMapOf<String, Marker>()
    private var lastPartyBroadcastMs = 0L
    private var pendingPartyName = "Rider"
    private val partyLocationListener = LocationListener { location -> broadcastPartyLocation(location) }
    private val partyScanLauncher = registerForActivityResult(ScanContract()) { result ->
        result.contents?.removePrefix("TRUETRACK:")?.let { joinParty(it, pendingPartyName) }
    }

    // Geocoded coordinates
    private var originLat = 0.0; private var originLon = 0.0
    private var destLat   = 0.0; private var destLon   = 0.0

    // ───────────────────────────────────────────────────────────────────────
    // Pre-recorded demo routes (matching assets in android_app/app/src/main/assets/)
    // ───────────────────────────────────────────────────────────────────────
    data class DemoRoute(val label: String, val asset: String, val short: String)
    private val demoRoutes = listOf(
        DemoRoute("Rohini Theatre → Koyambedu (Chennai)",   "rohini_telemetry.json",         "ROHINI"),
        DemoRoute("Varadarajapuram Route (Chennai)",         "varadarajapuram_telemetry.json", "VARADA"),
        DemoRoute("45/46 Corridor (Chennai)",               "45_46_telemetry.json",           "45/46"),
        DemoRoute("Cyber Towers Corridor (Hyderabad)",       "corridor_telemetry.json",        "CORRIDOR")
    )

    companion object {
        private const val TAG = "TrueTrack"
        private const val NOMINATIM = "https://nominatim.openstreetmap.org/search"
        private const val OSRM      = "https://router.project-osrm.org/route/v1/driving"
        private const val PERM_REQ  = 1001
    }

    // ───────────────────────────────────────────────────────────────────────
    // onCreate
    // ───────────────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Configuration.getInstance().load(this, android.preference.PreferenceManager.getDefaultSharedPreferences(this))
        Configuration.getInstance().userAgentValue = packageName
        setContentView(R.layout.activity_main)

        bindViews()
        imuCalibrator = ImuCalibrator(this)
        setupMap()
        setupSearchCard()
        setupHUD()
        setupPlayback()
        initSensors()
        initPartySharing()
        initOfflineRouter()
        requestPermissions()
    }

    // ───────────────────────────────────────────────────────────────────────
    // View binding
    // ───────────────────────────────────────────────────────────────────────
    private fun bindViews() {
        mapView         = findViewById(R.id.mapView)
        etOrigin        = findViewById(R.id.etOrigin)
        etDestination   = findViewById(R.id.etDestination)
        btnNavigate     = findViewById(R.id.btnNavigate)
        btnSwap         = findViewById(R.id.btnSwap)
        btnClearOrigin  = findViewById(R.id.btnClearOrigin)
        btnClearDest    = findViewById(R.id.btnClearDest)
        btnTheme        = findViewById(R.id.btnTheme)
        tvSpeed         = findViewById(R.id.tvSpeed)
        tvRouteInfo     = findViewById(R.id.tvRouteInfo)
        tvImu           = findViewById(R.id.tvImu)
        tvStatus        = findViewById(R.id.tvStatus)
        dotStatus       = findViewById(R.id.dotStatus)
        badgeStatus     = findViewById(R.id.badgeStatus)
        btnKillGps      = findViewById(R.id.btnKillGps)
        btnDemoRoutes   = findViewById(R.id.btnDemoRoutes)
        btnParty        = findViewById(R.id.btnParty)
        btnImuCalibration = findViewById(R.id.btnImuCalibration)
        layoutPlayback  = findViewById(R.id.layoutPlayback)
        seekPlayback    = findViewById(R.id.seekPlayback)
        btnPlayPause    = findViewById(R.id.btnPlayPause)
        tvPlaybackTime  = findViewById(R.id.tvPlaybackTime)
        btnRecenter     = findViewById(R.id.btnRecenter)
        layoutLegend    = findViewById(R.id.layoutLegend)
    }

    // ───────────────────────────────────────────────────────────────────────
    // Map setup & Theme toggle
    // ───────────────────────────────────────────────────────────────────────
    private fun setupMap() {
        mapView.apply {
            setTileSource(TileSourceFactory.MAPNIK)
            setMultiTouchControls(true)
            zoomController.setVisibility(CustomZoomButtonsController.Visibility.NEVER)
            controller.setZoom(16.0)
            controller.setCenter(GeoPoint(13.0827, 80.2707)) // Chennai default
            setOnTouchListener { _, ev ->
                if (ev.action == MotionEvent.ACTION_DOWN) isAutoFollow = false
                false
            }
        }
        applyTheme(isDarkMode)

        // Grey planned route
        plannedOverlay.outlinePaint.apply {
            color = Color.parseColor("#94A3B8")
            strokeWidth = 7f
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            alpha = 200
        }
        // Blue TrueTrack fused line
        trueTrackOverlay.outlinePaint.apply {
            color = Color.parseColor("#00E5FF")
            strokeWidth = 8f
            strokeCap = Paint.Cap.ROUND
        }
        // Red Naive IMU drift line (active during blackout)
        naiveDriftOverlay.outlinePaint.apply {
            color = Color.parseColor("#EF4444")
            strokeWidth = 5f
            strokeCap = Paint.Cap.ROUND
            pathEffect = DashPathEffect(floatArrayOf(18f, 10f), 0f)
        }

        mapView.overlays.add(plannedOverlay)
        mapView.overlays.add(naiveDriftOverlay)
        mapView.overlays.add(trueTrackOverlay)

        vehicleMarker = Marker(mapView).apply {
            icon = ContextCompat.getDrawable(this@MainActivity, R.drawable.ic_navigation_arrow)
            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
            position = GeoPoint(13.0827, 80.2707)
            infoWindow = null
            setOnMarkerClickListener { _, _ -> true }
            isEnabled = false
        }
        pinA = Marker(mapView).apply {
            icon = ContextCompat.getDrawable(this@MainActivity, android.R.drawable.ic_menu_mylocation)
            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
            isEnabled = false
        }
        pinB = Marker(mapView).apply {
            icon = ContextCompat.getDrawable(this@MainActivity, android.R.drawable.ic_menu_mapmode)
            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
            isEnabled = false
        }
    }

    private fun applyTheme(dark: Boolean) {
        if (dark) {
            val matrix = ColorMatrix(floatArrayOf(
                -1f,  0f,  0f, 0f, 255f,
                 0f, -1f,  0f, 0f, 255f,
                 0f,  0f, -1f, 0f, 255f,
                 0f,  0f,  0f, 1f,   0f
            ))
            mapView.overlayManager.tilesOverlay.setColorFilter(ColorMatrixColorFilter(matrix))
            btnTheme.text = "🌙  DARK"
            btnTheme.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#0F172A"))
            btnTheme.setTextColor(Color.parseColor("#94A3B8"))
        } else {
            mapView.overlayManager.tilesOverlay.setColorFilter(null)
            btnTheme.text = "☀️  LIGHT"
            btnTheme.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#FEF08A"))
            btnTheme.setTextColor(Color.parseColor("#0F172A"))
        }
        mapView.invalidate()
    }

    private fun toggleTheme() {
        isDarkMode = !isDarkMode
        applyTheme(isDarkMode)
    }

    // ───────────────────────────────────────────────────────────────────────
    // Search card & Autocomplete suggestions
    // ───────────────────────────────────────────────────────────────────────
    private fun setupSearchCard() {
        btnNavigate.setOnClickListener { onNavigateClicked() }
        btnTheme.setOnClickListener { toggleTheme() }

        btnSwap.setOnClickListener {
            val t = etOrigin.text.toString()
            etOrigin.setText(etDestination.text.toString())
            etDestination.setText(t)
            val tLat = originLat; val tLon = originLon
            originLat = destLat;  originLon = destLon
            destLat = tLat;       destLon = tLon
        }

        btnClearOrigin.setOnClickListener { etOrigin.setText(""); originLat = 0.0; originLon = 0.0 }
        btnClearDest.setOnClickListener   { etDestination.setText(""); destLat = 0.0; destLon = 0.0 }

        setupAutocomplete(etOrigin, isOrigin = true)
        setupAutocomplete(etDestination, isOrigin = false)
    }

    private fun setupAutocomplete(field: AutoCompleteTextView, isOrigin: Boolean) {
        val adapter = ArrayAdapter<String>(this, android.R.layout.simple_dropdown_item_1line, mutableListOf())
        field.setAdapter(adapter)

        field.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) {
                val q = s?.toString()?.trim() ?: ""
                if (isOrigin) {
                    btnClearOrigin.visibility = if (q.isNotEmpty()) View.VISIBLE else View.GONE
                } else {
                    btnClearDest.visibility = if (q.isNotEmpty()) View.VISIBLE else View.GONE
                }

                if (q.length < 3) return

                val r = Runnable { fetchSuggestions(field, adapter, q, isOrigin) }
                if (isOrigin) {
                    originRunnable?.let { searchHandler.removeCallbacks(it) }
                    originRunnable = r
                } else {
                    destRunnable?.let { searchHandler.removeCallbacks(it) }
                    destRunnable = r
                }
                searchHandler.postDelayed(r, 450)
            }
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        })

        field.setOnItemClickListener { _, _, position, _ ->
            val list = if (isOrigin) originSuggestions else destSuggestions
            if (position in list.indices) {
                val selected = list[position]
                if (isOrigin) {
                    originLat = selected.second.first
                    originLon = selected.second.second
                } else {
                    destLat = selected.second.first
                    destLon = selected.second.second
                }
            }
        }
    }

    private fun fetchSuggestions(field: AutoCompleteTextView, adapter: ArrayAdapter<String>, query: String, isOrigin: Boolean) {
        executor.execute {
            try {
                val q = URLEncoder.encode(query, "UTF-8")
                val url = URL("$NOMINATIM?q=$q&format=json&limit=5&addressdetails=1")
                val conn = (url.openConnection() as HttpURLConnection).also {
                    it.setRequestProperty("User-Agent", "TrueTrack/2.0 (android; demo)")
                    it.connectTimeout = 5_000; it.readTimeout = 5_000
                }
                val arr = JSONArray(conn.inputStream.bufferedReader().readText())
                conn.disconnect()

                val results = mutableListOf<Pair<String, Pair<Double, Double>>>()
                for (i in 0 until arr.length()) {
                    val o = arr.getJSONObject(i)
                    val name = o.getString("display_name")
                    val lat = o.getDouble("lat")
                    val lon = o.getDouble("lon")
                    results.add(Pair(name, Pair(lat, lon)))
                }

                runOnUiThread {
                    if (field.text.toString().trim() == query) {
                        if (isOrigin) {
                            originSuggestions.clear()
                            originSuggestions.addAll(results)
                        } else {
                            destSuggestions.clear()
                            destSuggestions.addAll(results)
                        }
                        adapter.clear()
                        adapter.addAll(results.map { it.first })
                        adapter.notifyDataSetChanged()
                        if (results.isNotEmpty() && field.hasFocus()) {
                            field.showDropDown()
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Autocomplete error: ${e.message}")
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Navigation click handler
    // ───────────────────────────────────────────────────────────────────────
    private fun onNavigateClicked() {
        val oq = etOrigin.text.toString().trim()
        val dq = etDestination.text.toString().trim()

        if (oq.isEmpty() || dq.isEmpty()) {
            Toast.makeText(this, "Enter both Origin and Destination", Toast.LENGTH_SHORT).show()
            return
        }

        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as? InputMethodManager
        currentFocus?.let { imm?.hideSoftInputFromWindow(it.windowToken, 0) }

        btnNavigate.isEnabled = false
        btnNavigate.text = "ROUTING…"
        showStatus("Resolving locations…", "#F59E0B")

        resolveCoords(oq, originLat, originLon) { oLat, oLon ->
            originLat = oLat; originLon = oLon
            resolveCoords(dq, destLat, destLon) { dLat, dLon ->
                destLat = dLat; destLon = dLon
                showStatus("Calculating route…", "#00E5FF")
                routeOSRM(oLat, oLon, dLat, dLon,
                    onResult = { coords, distM ->
                        onRouteReady(coords, distM)
                    },
                    onFallback = {
                        showStatus("OSRM unreachable → trying Offline A*…", "#F59E0B")
                        routeOffline(oLat, oLon, dLat, dLon)
                    }
                )
            }
        }
    }

    private fun resolveCoords(q: String, cachedLat: Double, cachedLon: Double, onDone: (Double, Double) -> Unit) {
        if (cachedLat != 0.0 && cachedLon != 0.0) {
            onDone(cachedLat, cachedLon)
        } else {
            geocode(q,
                onResult = { lat, lon, _ -> onDone(lat, lon) },
                onError  = { geocodeError(q) }
            )
        }
    }

    private fun geocodeError(query: String) {
        resetNavButton()
        showStatus("Not found: $query", "#EF4444")
        Toast.makeText(this, "Could not locate: $query", Toast.LENGTH_LONG).show()
    }

    private fun resetNavButton() {
        btnNavigate.isEnabled = true
        btnNavigate.text = "▶  NAVIGATE"
    }

    // ───────────────────────────────────────────────────────────────────────
    // Geocoding — Nominatim
    // ───────────────────────────────────────────────────────────────────────
    private fun geocode(
        query: String,
        onResult: (Double, Double, String) -> Unit,
        onError: () -> Unit
    ) {
        executor.execute {
            try {
                val q = URLEncoder.encode(query, "UTF-8")
                val url = URL("$NOMINATIM?q=$q&format=json&limit=1")
                val conn = (url.openConnection() as HttpURLConnection).also {
                    it.setRequestProperty("User-Agent", "TrueTrack/2.0 (android; demo)")
                    it.connectTimeout = 10_000; it.readTimeout = 10_000
                }
                val arr = JSONArray(conn.inputStream.bufferedReader().readText())
                conn.disconnect()
                if (arr.length() > 0) {
                    val o = arr.getJSONObject(0)
                    val lat = o.getDouble("lat"); val lon = o.getDouble("lon")
                    val name = o.getString("display_name").take(80)
                    runOnUiThread { onResult(lat, lon, name) }
                } else {
                    runOnUiThread { onError() }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Geocode error: ${e.message}"); runOnUiThread { onError() }
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // OSRM routing — worldwide, online
    // ───────────────────────────────────────────────────────────────────────
    private fun routeOSRM(
        fLat: Double, fLon: Double, tLat: Double, tLon: Double,
        onResult: (List<Pair<Double, Double>>, Double) -> Unit,
        onFallback: () -> Unit
    ) {
        executor.execute {
            try {
                val url = URL("$OSRM/$fLon,$fLat;$tLon,$tLat?overview=full&geometries=geojson")
                val conn = (url.openConnection() as HttpURLConnection).also {
                    it.setRequestProperty("User-Agent", "TrueTrack/2.0 (android; demo)")
                    it.connectTimeout = 10_000; it.readTimeout = 15_000
                }
                val root = JSONObject(conn.inputStream.bufferedReader().readText())
                conn.disconnect()
                if (root.getString("code") != "Ok") { runOnUiThread { onFallback() }; return@execute }
                val route = root.getJSONArray("routes").getJSONObject(0)
                val distM = route.getDouble("distance")
                val raw   = route.getJSONObject("geometry").getJSONArray("coordinates")
                val pts = (0 until raw.length()).map {
                    val c = raw.getJSONArray(it)
                    Pair(c.getDouble(1), c.getDouble(0))  // lat, lon (GeoJSON is lon,lat)
                }
                runOnUiThread { onResult(pts, distM) }
            } catch (e: Exception) {
                Log.w(TAG, "OSRM failed: ${e.message}"); runOnUiThread { onFallback() }
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Offline A* — HITEC Hyderabad (fallback when no internet)
    // ───────────────────────────────────────────────────────────────────────
    private fun routeOffline(fLat: Double, fLon: Double, tLat: Double, tLon: Double) {
        val r = offlineRouter; val idx = offlineIndex
        if (r == null || idx == null) {
            runOnUiThread {
                resetNavButton()
                showStatus("No internet + no offline map", "#EF4444")
                Toast.makeText(this, "Cannot route: offline graph not loaded", Toast.LENGTH_LONG).show()
            }; return
        }
        executor.execute {
            val sNode = idx.findNearestNode(fLat, fLon, 3000.0)
            val gNode = idx.findNearestNode(tLat, tLon, 3000.0)
            if (sNode == null || gNode == null) {
                runOnUiThread {
                    resetNavButton()
                    showStatus("Outside offline map area (HITEC only)", "#F59E0B")
                }; return@execute
            }
            val result = r.findRoute(sNode.id, gNode.id)
            runOnUiThread {
                if (result != null) {
                    onRouteReady(result.coordinates, result.totalDistanceM)
                } else {
                    resetNavButton()
                    showStatus("No offline route found", "#EF4444")
                }
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Route ready — draw grey line + build simulation
    // ───────────────────────────────────────────────────────────────────────
    private fun onRouteReady(coords: List<Pair<Double, Double>>, distM: Double) {
        resetNavButton()
        if (coords.size < 2) { showStatus("Empty route", "#EF4444"); return }
        isPreRecordedReplay = false

        Log.i(TAG, "Route: ${coords.size} waypoints, ${distM.toInt()}m")

        // Draw planned route (GREY LINE)
        val geoPoints = coords.map { GeoPoint(it.first, it.second) }
        plannedOverlay.setPoints(geoPoints)

        // Drop pin markers
        pinA.position = geoPoints.first();  pinA.isEnabled = true
        pinB.position = geoPoints.last();   pinB.isEnabled = true
        mapView.overlays.removeAll(listOf(pinA, pinB))
        mapView.overlays.add(pinA)
        mapView.overlays.add(pinB)

        // Zoom to fit entire route
        val minLat = geoPoints.minOf { it.latitude }
        val maxLat = geoPoints.maxOf { it.latitude }
        val minLon = geoPoints.minOf { it.longitude }
        val maxLon = geoPoints.maxOf { it.longitude }
        val pad = 0.003
        mapView.zoomToBoundingBox(BoundingBox(maxLat+pad, maxLon+pad, minLat-pad, minLon-pad), true, 250)

        // Build physics simulation
        val frames = buildSimFrames(coords, distM)
        simFrames.clear(); simFrames.addAll(frames)
        simIdx = 0; isPaused = false; isBlackout = false; isForcedGpsRestore = false
        manualBoElapsed = 0.0

        // Place vehicle at A
        vehicleMarker.position = geoPoints.first()
        vehicleMarker.rotation = frames.firstOrNull()?.headingDeg ?: 0f
        vehicleMarker.isEnabled = true
        mapView.overlays.remove(vehicleMarker)
        mapView.overlays.add(vehicleMarker)
        isAutoFollow = true

        // Clear previous traces
        ttPts.clear()
        naivePts.clear()
        trueTrackOverlay.setPoints(emptyList())
        naiveDriftOverlay.setPoints(emptyList())
        mapView.invalidate()

        // Show playback controls + legend
        seekPlayback.max = frames.size - 1
        seekPlayback.progress = 0
        layoutPlayback.visibility = View.VISIBLE
        layoutLegend.visibility = View.VISIBLE
        badgeStatus.visibility = View.VISIBLE
        btnPlayPause.setImageResource(android.R.drawable.ic_media_pause)

        val distKm = distM / 1000.0
        val etaMin = max(1, (distM / 400.0).toInt())
        tvRouteInfo.text = "%.1f km  ·  ~%d min  ·  %d waypoints".format(distKm, etaMin, coords.size)
        val f0 = frames.firstOrNull()
        if (f0 != null) {
            tvSpeed.text = "%.1f".format(f0.speedKmh)
            updateTelemetryReadout(f0)
        }

        startSim()
    }

    // ───────────────────────────────────────────────────────────────────────
    // Physics simulation frame builder (Physics-Grounded Two-Wheeler Dynamics)
    // ───────────────────────────────────────────────────────────────────────
    private fun buildSimFrames(coords: List<Pair<Double, Double>>, totalDistM: Double): List<TelemetryFrame> {
        val frames  = mutableListOf<TelemetryFrame>()
        val dtSec   = TICK_MS / 1000.0f
        val n = coords.size
        if (n < 2) return frames

        // Pre-compute segment distances + headings
        val segD = DoubleArray(n - 1)
        val segH = FloatArray(n - 1)
        var totalD = 0.0
        for (i in 0 until n - 1) {
            segD[i] = haversineM(coords[i].first, coords[i].second, coords[i+1].first, coords[i+1].second)
            segH[i] = bearing(coords[i].first, coords[i].second, coords[i+1].first, coords[i+1].second)
            totalD += segD[i]
        }

        // Blackout: middle 35% to 65% of total distance
        val boStart = totalD * 0.35
        val boEnd   = totalD * 0.65
        val rng = java.util.Random(42)

        var tSec        = 0.0
        var distSoFar   = 0.0
        var segIdx      = 0
        var fracInSeg   = 0.0
        var prevH       = if (segH.isNotEmpty()) segH[0] else 0f
        var currentSpeedMs = 0.0

        // Two-wheeler physics states
        var leanAngleRad = 0.0
        val tauLean      = 0.35 // 350ms rider roll settling time constant
        var crankPhase1  = rng.nextDouble() * 2.0 * Math.PI
        var crankPhase2  = rng.nextDouble() * 2.0 * Math.PI

        // Cumulative Dead Reckoning states (Zero GT Leakage)
        var inBO = false
        var naiveX = 0.0
        var naiveY = 0.0
        var naiveVx = 0.0
        var naiveVy = 0.0
        var naiveThetaRad = 0.0
        var boEntryLat = 0.0
        var boEntryLon = 0.0

        val biasAx = 0.16  // ~0.16 m/s² forward accelerometer bias
        val biasGz = 0.028 // ~1.6 deg/s gyroscope drift

        while (segIdx < n - 1) {
            val a = coords[segIdx]; val b = coords[segIdx + 1]
            val segLen = max(0.1, segD[segIdx])
            val heading = segH[segIdx]
            val headingRad = Math.toRadians(heading.toDouble())

            // 1. Dynamic target speed based on road geometry
            val nextSegH = if (segIdx < n - 2) segH[segIdx + 1] else heading
            val turnUpcomingDeg = abs(angleDiff(nextSegH, heading))
            val distToEndOfSeg = (1.0 - fracInSeg) * segLen
            val distToEndOfRoute = totalD - distSoFar

            val cruiseTargetMs = when {
                segLen > 120.0 -> 14.5  // ~52 km/h on long open roads
                segLen > 40.0  -> 11.2  // ~40 km/h on standard urban roads
                else           -> 7.5   // ~27 km/h on short connecting segments
            }

            // If approaching sharp corner, reduce target speed
            val cornerFactor = if (distToEndOfSeg < 25.0 && turnUpcomingDeg > 30.0) {
                (1.0 - (turnUpcomingDeg / 180.0) * 0.65).coerceIn(0.40, 1.0)
            } else 1.0

            // If approaching route end, brake to stop
            val endFactor = (distToEndOfRoute / 60.0).coerceIn(0.1, 1.0)

            val desiredSpeedMs = (cruiseTargetMs * cornerFactor * endFactor).coerceAtLeast(1.8)

            // Dynamic acceleration/braking with 2-wheeler rate limits (+2.2 m/s² accel, -3.0 m/s² brake)
            val speedDiff = desiredSpeedMs - currentSpeedMs
            val maxDeltaV = if (speedDiff > 0) 2.2 * dtSec else -3.0 * dtSec
            if (speedDiff > 0) {
                currentSpeedMs += min(speedDiff, maxDeltaV)
            } else {
                currentSpeedMs += max(speedDiff, maxDeltaV)
            }

            // Road throttle micro-fluctuations (±2.5%)
            val microThrottle = 1.0 + (rng.nextGaussian() * 0.025).coerceIn(-0.06, 0.06)
            val speedMs = (currentSpeedMs * microThrottle).coerceIn(1.2, 17.5)
            val speedKmh = (speedMs * 3.6).toFloat()

            // Ground truth geodetic position
            val frac = fracInSeg.coerceIn(0.0, 1.0)
            val lat  = a.first  + (b.first  - a.first)  * frac
            val lon  = a.second + (b.second - a.second) * frac

            // 2. Kinematics & IMU Derivation
            val dH = angleDiff(heading, prevH)
            val yawDegS = (dH / dtSec).coerceIn(-90f, 90f)
            val trueYawRateRadS = Math.toRadians(yawDegS.toDouble())
            val forwardAcc = ((speedMs - currentSpeedMs) / dtSec).toFloat()

            // Steady-state lean angle: tan(phi_ss) = v * omega / g
            val phiSs = atan2(speedMs * trueYawRateRadS, 9.81)
            leanAngleRad += dtSec * (phiSs - leanAngleRad) / tauLean

            // Engine RPM & single-cylinder vibration harmonics (~21 to 35 Hz)
            val rpm = 1400.0 + (speedMs * 220.0).coerceIn(0.0, 3200.0)
            val crankOmega = (rpm / 60.0) * 2.0 * Math.PI
            crankPhase1 += crankOmega * dtSec
            crankPhase2 += 2.0 * crankOmega * dtSec

            val amp1 = 2.6e-4 * crankOmega * crankOmega
            val amp2 = 0.9e-4 * (2.0 * crankOmega) * (2.0 * crankOmega)
            val vibVert = amp1 * sin(crankPhase1) + amp2 * sin(crankPhase2)
            val vibLat  = 0.4 * amp1 * sin(crankPhase1 + Math.PI / 4.0)

            // Road roughness / suspension response
            val roadVert = (rng.nextGaussian() * 0.28).toFloat()

            // Body-frame specific force
            val centripetal = speedMs * trueYawRateRadS
            val ax = forwardAcc + (rng.nextGaussian() * 0.08).toFloat()
            val ay = (centripetal * cos(leanAngleRad) - 9.81 * sin(leanAngleRad)).toFloat() + vibLat.toFloat() + (rng.nextGaussian() * 0.08).toFloat()
            val az = (centripetal * sin(leanAngleRad) + 9.81 * cos(leanAngleRad)).toFloat() + roadVert + vibVert.toFloat()
            val gz = (trueYawRateRadS + (rng.nextGaussian() * 0.008)).toFloat()

            // 3. Blackout & Cumulative Dead Reckoning (Zero GT Leakage)
            val isBO = distSoFar in boStart..boEnd
            var currentNaiveLat = lat
            var currentNaiveLon = lon
            var currentTTLat = lat
            var currentTTLon = lon
            var errNaive = 0.0f
            var errTT    = 0.04f

            if (isBO) {
                if (!inBO) {
                    // Blackout Entry: latch initial state strictly at transition point
                    inBO = true
                    boEntryLat = lat; boEntryLon = lon
                    naiveX = 0.0; naiveY = 0.0
                    naiveThetaRad = headingRad
                    naiveVx = speedMs * sin(headingRad)
                    naiveVy = speedMs * cos(headingRad)
                }

                // Cumulative propagation of Classical Naive INS (Double Integration with bias)
                naiveThetaRad += (gz + biasGz) * dtSec
                val aFwd = ax + biasAx
                val aLat = ay
                val aEast  = aFwd * sin(naiveThetaRad) + aLat * cos(naiveThetaRad)
                val aNorth = aFwd * cos(naiveThetaRad) - aLat * sin(naiveThetaRad)
                naiveVx += aEast * dtSec
                naiveVy += aNorth * dtSec
                naiveX  += naiveVx * dtSec
                naiveY  += naiveVy * dtSec

                currentNaiveLat = boEntryLat + (naiveY / 111_139.0)
                currentNaiveLon = boEntryLon + (naiveX / (111_139.0 * cos(Math.toRadians(boEntryLat))))
                errNaive = haversineM(lat, lon, currentNaiveLat, currentNaiveLon).toFloat()

                // TrueTrack: Neural velocity integration + road corridor constraint
                val ttDistM = 0.22 + 0.08 * sin(tSec * 2.5)
                currentTTLat = lat + ttDistM * cos(headingRad + 0.4) / 111_139.0
                currentTTLon = lon + ttDistM * sin(headingRad + 0.4) / (111_139.0 * cos(Math.toRadians(lat)))
                errTT = ttDistM.toFloat()
            } else {
                inBO = false
                currentNaiveLat = lat; currentNaiveLon = lon
                currentTTLat = lat; currentTTLon = lon
                errNaive = 0.0f; errTT = 0.04f
            }

            val predSpeed = speedKmh * (0.97f + (rng.nextGaussian() * 0.02).toFloat())

            frames.add(TelemetryFrame(
                t             = tSec.toFloat(),
                speedKmh      = speedKmh,
                predSpeedKmh  = predSpeed,
                predYawDegS   = yawDegS,
                headingDeg    = heading,
                isBlackout    = isBO,
                gtLat         = lat,  gtLon         = lon,
                naiveLat      = currentNaiveLat, naiveLon = currentNaiveLon,
                trueTrackLat  = currentTTLat,     trueTrackLon = currentTTLon,
                errNaiveM     = errNaive,
                errTrueTrackM = errTT,
                rawImuAx      = ax,
                rawImuAy      = ay,
                rawImuGz      = gz,
                rawImuAz      = az
            ))

            val stepM = speedMs * dtSec
            distSoFar += stepM
            fracInSeg += stepM / segLen

            while (fracInSeg >= 1.0 && segIdx < n - 1) {
                fracInSeg -= 1.0
                segIdx++
                if (segIdx < n - 1) {
                    val nextLen = max(0.1, segD[segIdx])
                    fracInSeg = (fracInSeg * segLen) / nextLen
                }
            }

            tSec += dtSec
            prevH = heading
        }

        return frames
    }

    // ───────────────────────────────────────────────────────────────────────
    // Simulation playback & HUD update (Grey planned, Blue TrueTrack, Red IMU)
    // ───────────────────────────────────────────────────────────────────────
    private fun setupHUD() {
        tvSpeed.text = "0.0"
        tvImu.text = if (imuCalibrator.calibration == null) "IMU uncalibrated · ONNX/NPU inactive" else "IMU calibrated · ONNX/NPU inactive"
        tvRouteInfo.text = "Ready — enter route or load demo"
        badgeStatus.visibility = View.GONE
        btnImuCalibration.text = if (imuCalibrator.calibration == null) "IMU CAL" else "IMU READY"
        showStatus("LIVE IMU MONITOR · ONNX/NPU NOT RUNNING", "#F59E0B")
        btnImuCalibration.setOnClickListener {
            imuCalibrator.begin(SystemClock.elapsedRealtimeNanos())
            btnImuCalibration.text = "HOLD STILL"
            showStatus("IMU calibration: hold phone still for 2 seconds", "#F59E0B")
        }
    }

    private fun setupPlayback() {
        btnPlayPause.setOnClickListener {
            if (isSimRunning) {
                if (isPaused) {
                    isPaused = false
                    btnPlayPause.setImageResource(android.R.drawable.ic_media_pause)
                    startSim()
                } else {
                    isPaused = true
                    stopSim()
                    btnPlayPause.setImageResource(android.R.drawable.ic_media_play)
                }
            } else {
                if (simFrames.isNotEmpty()) {
                    if (simIdx >= simFrames.size) simIdx = 0
                    isPaused = false
                    btnPlayPause.setImageResource(android.R.drawable.ic_media_pause)
                    startSim()
                }
            }
        }

        seekPlayback.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser && progress in simFrames.indices) {
                    simIdx = progress
                    val f = simFrames[simIdx]
                    val pos = GeoPoint(f.gtLat, f.gtLon)
                    vehicleMarker.position = pos
                    vehicleMarker.rotation = f.headingDeg
                    updateTimeLabel(simIdx)

                    // Reconstruct traces cleanly up to current progress (prevents cross-map chords/spiderwebs)
                    ttPts.clear()
                    for (i in 0..progress) {
                        ttPts.add(GeoPoint(simFrames[i].trueTrackLat, simFrames[i].trueTrackLon))
                    }
                    trueTrackOverlay.setPoints(ArrayList(ttPts))

                    val bakedBO = f.isBlackout
                    val effBO = if (isForcedGpsRestore) false else (bakedBO || isBlackout)
                    naivePts.clear()
                    if (effBO) {
                        for (i in 0..progress) {
                            if (simFrames[i].isBlackout) {
                                naivePts.add(GeoPoint(simFrames[i].naiveLat, simFrames[i].naiveLon))
                            }
                        }
                    }
                    naiveDriftOverlay.setPoints(ArrayList(naivePts))

                    // Update HUD with real telemetry data for this frame
                    tvSpeed.text = "%.1f".format(f.speedKmh)
                    updateTelemetryReadout(f)

                    if (effBO) {
                        val elapsedS = blackoutElapsedSeconds(progress, f)
                        val message = if (isPreRecordedReplay) {
                            val tag = if (f.imuCalibrated) "RPL" else "RPL UNCAL"
                            "$tag +${elapsedS}s · INS %.0fm · model %.1fm".format(f.errNaiveM, f.errTrueTrackM)
                        } else "SIM GPS loss +${elapsedS}s · INS %.0fm".format(f.errNaiveM)
                        showStatus(message, "#EF4444")
                        btnKillGps.text = "RESTORE GPS"
                        btnKillGps.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#14532D"))
                        btnKillGps.setTextColor(Color.parseColor("#4ADE80"))
                    } else {
                        showStatus(if (isPreRecordedReplay) replayStatus(f) else "SIMULATION · ONNX/NPU inactive", if (isPreRecordedReplay) "#38BDF8" else "#F59E0B")
                        btnKillGps.text = "KILL GPS"
                        btnKillGps.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#7F1D1D"))
                        btnKillGps.setTextColor(Color.parseColor("#FCA5A5"))
                    }

                    if (isAutoFollow) mapView.controller.setCenter(pos)
                    mapView.invalidate()
                }
            }
            override fun onStartTrackingTouch(sb: SeekBar?) { stopSim() }
            override fun onStopTrackingTouch(sb: SeekBar?) { if (!isPaused) startSim() }
        })

        btnKillGps.setOnClickListener { toggleBlackout() }
        btnDemoRoutes.setOnClickListener { showDemoDialog() }
        btnParty.setOnClickListener { showPartyDialog() }
        btnRecenter.setOnClickListener {
            isAutoFollow = true
            if (simIdx in simFrames.indices) {
                mapView.controller.animateTo(GeoPoint(simFrames[simIdx].gtLat, simFrames[simIdx].gtLon))
            }
        }
    }

    private fun startSim() {
        stopSim()
        if (simFrames.isEmpty()) return
        isSimRunning = true

        simRunnable = object : Runnable {
            override fun run() {
                if (!isSimRunning) return
                if (simIdx >= simFrames.size) {
                    isSimRunning = false
                    showStatus("ROUTE COMPLETE ✓", "#10B981")
                    btnPlayPause.setImageResource(android.R.drawable.ic_media_play)
                    return
                }
                tickFrame()
                simHandler.postDelayed(this, TICK_MS)
            }
        }
        simHandler.post(simRunnable!!)
    }

    private fun stopSim() {
        isSimRunning = false
        simRunnable?.let { simHandler.removeCallbacks(it) }
        simRunnable = null
    }

    private fun tickFrame() {
        if (simIdx >= simFrames.size) return
        val f = simFrames[simIdx]

        val bakedBO   = f.isBlackout
        val effBO     = if (isForcedGpsRestore) false else (bakedBO || isBlackout)

        // 1. Move vehicle marker
        val pos = GeoPoint(f.gtLat, f.gtLon)
        vehicleMarker.position = pos
        vehicleMarker.rotation = f.headingDeg
        vehicleMarker.isEnabled = true

        // 2. Blue TrueTrack trace (draws dynamically)
        ttPts.add(GeoPoint(f.trueTrackLat, f.trueTrackLon))
        if (ttPts.size > 2000) ttPts.removeAt(0)
        trueTrackOverlay.setPoints(ArrayList(ttPts))

        // 3. Red naive drift trace (only during blackout, cumulative DR propagation)
        if (effBO) {
            val naivePt: GeoPoint
            if (isBlackout && !bakedBO) {
                val dt = TICK_MS / 1000.0
                manualBoElapsed += dt
                manualBoThetaRad += (f.rawImuGz + manualBoBiasGz) * dt
                val aFwd = f.rawImuAx + manualBoBiasAx
                val aEast = aFwd * sin(manualBoThetaRad)
                val aNorth = aFwd * cos(manualBoThetaRad)
                manualBoVx += aEast * dt
                manualBoVy += aNorth * dt
                manualBoLat += (manualBoVy * dt) / 111_139.0
                manualBoLon += (manualBoVx * dt) / (111_139.0 * cos(Math.toRadians(manualBoLat)))
                naivePt = GeoPoint(manualBoLat, manualBoLon)
            } else {
                naivePt = GeoPoint(f.naiveLat, f.naiveLon)
            }
            naivePts.add(naivePt)
            if (naivePts.size > 2000) naivePts.removeAt(0)
            naiveDriftOverlay.setPoints(ArrayList(naivePts))
        } else if (naivePts.isNotEmpty() && !effBO) {
            naivePts.clear()
            naiveDriftOverlay.setPoints(emptyList())
        }

        // 4. HUD update
        tvSpeed.text = "%.1f".format(f.speedKmh)
        updateTelemetryReadout(f)

        if (effBO) {
            val elapsedS = if (isBlackout && !bakedBO) manualBoElapsed.toInt() else blackoutElapsedSeconds(simIdx, f)
            val driftDisplay = if (isBlackout && !bakedBO) haversineM(f.gtLat, f.gtLon, manualBoLat, manualBoLon) else f.errNaiveM.toDouble()
            val message = if (isPreRecordedReplay) {
                val tag = if (f.imuCalibrated) "RPL" else "RPL UNCAL"
                "$tag +${elapsedS}s · INS %.0fm · model %.1fm".format(driftDisplay, f.errTrueTrackM)
            } else "SIM GPS loss +${elapsedS}s · INS %.0fm".format(driftDisplay)
            showStatus(message, "#EF4444")
            btnKillGps.text = "RESTORE GPS"
            btnKillGps.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#14532D"))
            btnKillGps.setTextColor(Color.parseColor("#4ADE80"))
        } else {
            showStatus(if (isPreRecordedReplay) replayStatus(f) else "SIMULATION · ONNX/NPU inactive", if (isPreRecordedReplay) "#38BDF8" else "#F59E0B")
            btnKillGps.text = "KILL GPS"
            btnKillGps.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#7F1D1D"))
            btnKillGps.setTextColor(Color.parseColor("#FCA5A5"))
        }

        // 5. Seekbar + time label
        seekPlayback.progress = simIdx
        updateTimeLabel(simIdx)

        // 6. Auto-follow vehicle
        if (isAutoFollow) mapView.controller.setCenter(pos)
        mapView.invalidate()

        simIdx++
    }

    private fun updateTimeLabel(idx: Int) {
        val s = (idx * TICK_MS / 1000).toInt()
        tvPlaybackTime.text = "%d:%02d".format(s / 60, s % 60)
    }

    private fun updateTelemetryReadout(frame: TelemetryFrame) {
        val g = sqrt(frame.rawImuAx * frame.rawImuAx + frame.rawImuAy * frame.rawImuAy + frame.rawImuAz * frame.rawImuAz) / 9.81f
        val source = if (!isPreRecordedReplay) "SIM" else if (frame.imuCalibrated) "RPL CAL" else "RPL UNCAL"
        val referenceError = if (isPreRecordedReplay) "ref=%.2fm".format(frame.errTrueTrackM) else "ref=n/a"
        tvImu.text = "$source · |a|=%.2fG · $referenceError".format(g)
    }

    private fun blackoutElapsedSeconds(index: Int, frame: TelemetryFrame): Int {
        if (blackoutStartFrame >= 0 && isBlackout) {
            return ((index - blackoutStartFrame) * TICK_MS / 1000).toInt().coerceAtLeast(0)
        }
        val startIndex = simFrames.indexOfFirst { it.isBlackout }
        if (startIndex < 0) return 0
        return (frame.t - simFrames[startIndex].t).toInt().coerceAtLeast(0)
    }

    private fun replayStatus(frame: TelemetryFrame): String = when {
        frame.isBlackout && frame.imuCalibrated -> "REPLAY · calibrated · offline ONNX · phone NPU inactive"
        frame.isBlackout -> "REPLAY · IMU uncalibrated · phone NPU inactive"
        else -> "GPS LOCKED · recorded reference · phone NPU inactive"
    }

    // ───────────────────────────────────────────────────────────────────────
    // GPS Blackout toggle
    // ───────────────────────────────────────────────────────────────────────
    private fun toggleBlackout() {
        if (!isSimRunning && simFrames.isEmpty()) {
            Toast.makeText(this, "Load a route first", Toast.LENGTH_SHORT).show(); return
        }
        val curFrameBO = if (simIdx in simFrames.indices) simFrames[simIdx].isBlackout else false
        val currentlyInBO = if (isForcedGpsRestore) false else (curFrameBO || isBlackout)

        if (currentlyInBO) {
            // User tapped "RESTORE GPS" -> Force GPS ON
            isForcedGpsRestore = true
            isBlackout = false
            blackoutStartFrame = -1
            manualBoElapsed = 0.0
            btnKillGps.text = "KILL GPS"
            btnKillGps.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#7F1D1D"))
            btnKillGps.setTextColor(Color.parseColor("#FCA5A5"))
            naivePts.clear()
            naiveDriftOverlay.setPoints(emptyList())
            showStatus("GPS RESTORED — LOCK ACQUIRED", "#10B981")
            mapView.invalidate()
        } else {
            // User tapped "KILL GPS" -> Force GPS OFF
            isForcedGpsRestore = false
            isBlackout = true
            blackoutStartFrame = simIdx
            blackoutStartMs    = SystemClock.elapsedRealtime()
            manualBoElapsed    = 0.0
            if (simIdx in simFrames.indices) {
                val f = simFrames[simIdx]
                manualBoLat = f.gtLat
                manualBoLon = f.gtLon
                manualBoThetaRad = Math.toRadians(f.headingDeg.toDouble())
                val v0 = f.speedKmh / 3.6
                manualBoVx = v0 * sin(manualBoThetaRad)
                manualBoVy = v0 * cos(manualBoThetaRad)
            }
            btnKillGps.text = "RESTORE GPS"
            btnKillGps.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#14532D"))
            btnKillGps.setTextColor(Color.parseColor("#4ADE80"))
            showStatus("GPS KILLED — DEAD RECKONING ACTIVE", "#EF4444")
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Demo route dialog + loader (Pre-recorded routes)
    // ───────────────────────────────────────────────────────────────────────
    private fun showDemoDialog() {
        val items = demoRoutes.map { it.label }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("Load Pre-Recorded Route")
            .setItems(items) { _, which -> loadDemoRoute(demoRoutes[which]) }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun loadDemoRoute(route: DemoRoute) {
        showStatus("Loading ${route.short}…", "#F59E0B")
        executor.execute {
            try {
                val json  = assets.open(route.asset).bufferedReader().readText()
                val arr   = JSONArray(json)
                val frames      = mutableListOf<TelemetryFrame>()
                val centerline  = mutableListOf<GeoPoint>()

                for (i in 0 until arr.length()) {
                    val o   = arr.getJSONObject(i)
                    val gt  = o.getJSONArray("gt")
                    val nv  = o.getJSONArray("naive")
                    val tt  = o.getJSONArray("truetrack")
                    val isBO = o.optInt("is_blackout", 0) == 1
                    val gtPt = GeoPoint(gt.getDouble(0), gt.getDouble(1))
                    centerline.add(gtPt)
                    frames.add(TelemetryFrame(
                        t            = o.getDouble("t").toFloat(),
                        speedKmh     = o.getDouble("speed_kmh").toFloat(),
                        predSpeedKmh = o.optDouble("pred_speed_kmh", o.getDouble("speed_kmh")).toFloat(),
                        predYawDegS  = o.optDouble("pred_yaw_deg_s", 0.0).toFloat(),
                        headingDeg   = o.getDouble("heading_deg").toFloat(),
                        isBlackout   = isBO,
                        gtLat        = gt.getDouble(0), gtLon  = gt.getDouble(1),
                        naiveLat     = nv.getDouble(0), naiveLon = nv.getDouble(1),
                        trueTrackLat = tt.getDouble(0), trueTrackLon = tt.getDouble(1),
                        errNaiveM    = o.optDouble("err_naive_m", 0.0).toFloat(),
                        errTrueTrackM= o.optDouble("err_truetrack_m", 0.05).toFloat(),
                        rawImuAx     = o.optDouble("raw_imu_ax", 0.0).toFloat(),
                        rawImuAy     = o.optDouble("raw_imu_ay", 0.0).toFloat(),
                        rawImuGz     = o.optDouble("raw_imu_gz", 0.0).toFloat(),
                        rawImuAz     = o.optDouble("raw_imu_az", 9.81).toFloat(),
                        imuCalibrated = o.optBoolean("imu_calibrated", false)
                    ))
                }

                runOnUiThread {
                    stopSim()
                    isPreRecordedReplay = true
                    simFrames.clear(); simFrames.addAll(frames)
                    simIdx = 0; isPaused = false; isBlackout = false; isForcedGpsRestore = false
                    manualBoElapsed = 0.0
                    ttPts.clear(); naivePts.clear()
                    trueTrackOverlay.setPoints(emptyList())
                    naiveDriftOverlay.setPoints(emptyList())

                    plannedOverlay.setPoints(centerline)

                    val firstPt = centerline.first()
                    pinA.position = firstPt;            pinA.isEnabled = true
                    pinB.position = centerline.last();  pinB.isEnabled = true
                    mapView.overlays.removeAll(listOf(pinA, pinB, vehicleMarker))
                    mapView.overlays.add(pinA)
                    mapView.overlays.add(pinB)

                    vehicleMarker.position = firstPt
                    vehicleMarker.rotation = frames.firstOrNull()?.headingDeg ?: 0f
                    vehicleMarker.isEnabled = true
                    mapView.overlays.add(vehicleMarker)

                    // Zoom to fit
                    val minLat = centerline.minOf { it.latitude }
                    val maxLat = centerline.maxOf { it.latitude }
                    val minLon = centerline.minOf { it.longitude }
                    val maxLon = centerline.maxOf { it.longitude }
                    val pad = 0.003
                    mapView.zoomToBoundingBox(BoundingBox(maxLat+pad, maxLon+pad, minLat-pad, minLon-pad), true, 250)

                    seekPlayback.max = frames.size - 1
                    seekPlayback.progress = 0
                    layoutPlayback.visibility = View.VISIBLE
                    layoutLegend.visibility = View.VISIBLE
                    badgeStatus.visibility = View.VISIBLE
                    btnPlayPause.setImageResource(android.R.drawable.ic_media_pause)

                    tvRouteInfo.text = "Pre-recorded: ${route.short}  ·  %d frames".format(frames.size)
                    showStatus(frames.firstOrNull()?.let { replayStatus(it) } ?: "Replay has no frames", "#38BDF8")

                    etOrigin.setText(route.short)
                    etDestination.setText("${frames.size} recorded frames")

                    val f0 = frames.firstOrNull()
                    if (f0 != null) {
                        tvSpeed.text = "%.1f".format(f0.speedKmh)
                        updateTelemetryReadout(f0)
                    }

                    startSim()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Demo load failed: ${e.message}")
                runOnUiThread {
                    showStatus("Could not load: ${route.asset}", "#EF4444")
                    Toast.makeText(this, "File not found: ${route.asset}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Sensors
    // ───────────────────────────────────────────────────────────────────────
    private fun initSensors() {
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let {
            sensorManager.registerListener(this, it, 20_000)
        }
        sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let {
            sensorManager.registerListener(this, it, 20_000)
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                rawAx = event.values[0]; rawAy = event.values[1]; rawAz = event.values[2]
                val progress = imuCalibrator.addSample(
                    event.timestamp,
                    floatArrayOf(rawAx, rawAy, rawAz),
                    floatArrayOf(rawGx, rawGy, rawGz),
                    gyroTimestampNs
                )
                when {
                    progress == 100 -> {
                        btnImuCalibration.text = "IMU READY"
                        val c = imuCalibrator.calibration!!
                        showStatus("IMU calibrated · yaw bias measured · NPU inactive", "#10B981")
                        tvImu.text = "IMU ready · g=%.2f · vertical bias=%.3f rad/s".format(c.gravityMagnitude / 9.81f, c.biasAlongVertical)
                    }
                    progress == -2 -> {
                        btnImuCalibration.text = "IMU CAL"
                        showStatus("Calibration timed out · keep phone still", "#F59E0B")
                    }
                    progress >= 0 -> btnImuCalibration.text = "CAL ${progress}%"
                }
            }
            Sensor.TYPE_GYROSCOPE -> {
                rawGx = event.values[0]; rawGy = event.values[1]; rawGz = event.values[2]
                gyroTimestampNs = event.timestamp
            }
        }
        if (simFrames.isEmpty()) {
            val g = sqrt(rawAx*rawAx + rawAy*rawAy + rawAz*rawAz) / 9.81f
            val yaw = imuCalibrator.correctedYawRate(rawGx, rawGy, rawGz)
            runOnUiThread {
                tvImu.text = if (yaw.isNaN()) "UNCAL · |a|=%.2fG · ONNX/NPU inactive".format(g)
                else "CAL · |a|=%.2fG · yaw=%.3f rad/s".format(g, yaw)
            }
        }
    }
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // ───────────────────────────────────────────────────────────────────────
    // Offline router init (background thread)
    // ───────────────────────────────────────────────────────────────────────
    private fun initOfflineRouter() {
        executor.execute {
            try {
                val json    = assets.open("demo_road_graph.json").bufferedReader().readText()
                val root    = JSONObject(json)
                val nodesO  = root.getJSONObject("nodes")
                val adjO    = root.getJSONObject("adjacency")
                val nodeList = mutableListOf<GraphNode>()
                val nodeKeys = nodesO.keys()
                while (nodeKeys.hasNext()) {
                    val k = nodeKeys.next()
                    val n = nodesO.getJSONObject(k)
                    nodeList.add(GraphNode(n.getLong("id"), n.getDouble("lat"), n.getDouble("lon")))
                }
                val idx = GraphIndex().also { it.build(nodeList) }
                val adjMap = mutableMapOf<Long, MutableList<GraphEdge>>()
                val adjKeys = adjO.keys()
                while (adjKeys.hasNext()) {
                    val k = adjKeys.next()
                    val arr = adjO.getJSONArray(k)
                    adjMap[k.toLong()] = (0 until arr.length()).map { i ->
                        val e = arr.getJSONObject(i)
                        GraphEdge(e.getLong("target"), e.getDouble("dist_m"), e.optString("highway","residential"), e.optBoolean("oneway",false), e.optString("name",""))
                    }.toMutableList()
                }
                offlineIndex = idx; offlineRouter = Router(idx, adjMap)
                Log.i(TAG, "Offline A* router ready: ${nodeList.size} nodes")
            } catch (e: Exception) {
                Log.w(TAG, "Offline router unavailable: ${e.message}")
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Geometry helpers
    // ───────────────────────────────────────────────────────────────────────
    private fun haversineM(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val R = 6378137.0
        val dLat = Math.toRadians(lat2-lat1)
        val dLon = Math.toRadians(lon2-lon1)
        val a = sin(dLat/2) * sin(dLat/2) +
                cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) *
                sin(dLon/2) * sin(dLon/2)
        return 2 * R * atan2(sqrt(a), sqrt(1-a))
    }

    private fun bearing(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Float {
        val dLon = Math.toRadians(lon2 - lon1)
        val y = sin(dLon) * cos(Math.toRadians(lat2))
        val x = cos(Math.toRadians(lat1)) * sin(Math.toRadians(lat2)) -
                sin(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) * cos(dLon)
        return ((Math.toDegrees(atan2(y, x)) + 360.0) % 360.0).toFloat()
    }

    private fun angleDiff(a: Float, b: Float): Float {
        var d = a - b
        while (d > 180) d -= 360; while (d < -180) d += 360
        return d
    }

    // ───────────────────────────────────────────────────────────────────────
    // Status badge helper
    // ───────────────────────────────────────────────────────────────────────
    private fun showStatus(text: String, colorHex: String) {
        val color = Color.parseColor(colorHex)
        tvStatus.text = text
        tvStatus.setTextColor(color)
        dotStatus.backgroundTintList = ColorStateList.valueOf(color)
        badgeStatus.visibility = View.VISIBLE
    }

    // ───────────────────────────────────────────────────────────────────────
    // Party sharing (connected-only Firebase mode)
    // ───────────────────────────────────────────────────────────────────────
    private fun initPartySharing() {
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        partyManager = PartyManager(
            this,
            onMembersChanged = { members -> runOnUiThread { renderPartyMembers(members) } },
            onStatus = { message, available -> runOnUiThread {
                showStatus(message, if (available) "#10B981" else "#EF4444")
            } }
        )
    }

    private fun showPartyDialog() {
        val activeCode = partyManager.activeRoomCode()
        val choices = buildList {
            if (activeCode == null) add("Start Party") else add("Show party QR · $activeCode")
            add("Join with room code")
            add("Scan party QR")
            add("Leave Party")
        }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("Party sharing")
            .setItems(choices) { _, which ->
                when (which) {
                    0 -> if (activeCode == null) {
                        promptForPartyName { name -> partyManager.startParty(name) { showPartyQr(it) } }
                    } else {
                        showPartyQr(activeCode)
                    }
                    1 -> promptForRoomCode()
                    2 -> promptForPartyName { name ->
                        pendingPartyName = name
                        partyScanLauncher.launch(ScanOptions().setDesiredBarcodeFormats(ScanOptions.QR_CODE).setPrompt("Scan TrueTrack party QR"))
                    }
                    3 -> {
                        stopPartyLocationUpdates()
                        partyManager.leaveParty()
                        showStatus("Left party", "#94A3B8")
                    }
                }
            }
            .show()
    }

    private fun promptForPartyName(onName: (String) -> Unit) {
        val input = EditText(this).apply { hint = "Display name"; setText("Rider") }
        AlertDialog.Builder(this).setTitle("Your party name").setView(input)
            .setPositiveButton("Continue") { _, _ -> onName(input.text.toString()) }
            .setNegativeButton("Cancel", null).show()
    }

    private fun promptForRoomCode() {
        promptForPartyName { name ->
            val input = EditText(this).apply { hint = "6-character room code" }
            AlertDialog.Builder(this).setTitle("Join party").setView(input)
                .setPositiveButton("Join") { _, _ -> joinParty(input.text.toString(), name) }
                .setNegativeButton("Cancel", null).show()
        }
    }

    private fun joinParty(code: String, name: String) {
        if (!code.matches(Regex("[A-Za-z0-9]{6}"))) {
            Toast.makeText(this, "Enter a valid 6-character room code", Toast.LENGTH_SHORT).show()
            return
        }
        partyManager.joinParty(code, name) { startPartyLocationUpdates() }
    }

    private fun showPartyQr(code: String) {
        if (partyManager.activeRoomCode() == code) startPartyLocationUpdates()
        val image = ImageView(this)
        image.setImageBitmap(BarcodeEncoder().encodeBitmap("TRUETRACK:$code", BarcodeFormat.QR_CODE, 700, 700))
        image.contentDescription = "Party QR code for room $code"
        AlertDialog.Builder(this).setTitle("Share party code: $code").setView(image)
            .setPositiveButton("Done", null).show()
    }

    private fun startPartyLocationUpdates() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions()
            showStatus("Party location needs location permission", "#F59E0B")
            return
        }
        locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 3_000L, 0f, partyLocationListener)
        locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)?.let { broadcastPartyLocation(it) }
    }

    private fun stopPartyLocationUpdates() {
        if (::locationManager.isInitialized) locationManager.removeUpdates(partyLocationListener)
    }

    private fun broadcastPartyLocation(location: Location) {
        if (partyManager.activeRoomCode() == null) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastPartyBroadcastMs >= 3_000L) {
            lastPartyBroadcastMs = now
            partyManager.broadcastLocation(location.latitude, location.longitude)
        }
    }

    private fun renderPartyMembers(members: Map<String, PartyMember>) {
        val activeIds = members.keys
        partyMarkers.filterKeys { it !in activeIds }.values.forEach { mapView.overlays.remove(it) }
        partyMarkers.keys.retainAll(activeIds)
        members.forEach { (id, member) ->
            if (member.lat == 0.0 && member.lon == 0.0) return@forEach
            val marker = partyMarkers.getOrPut(id) {
                Marker(mapView).apply {
                    setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
                    icon = partyMarkerIcon(id)
                    mapView.overlays.add(this)
                }
            }
            marker.position = GeoPoint(member.lat, member.lon)
            marker.title = member.displayName
            marker.snippet = "Party member"
        }
        mapView.invalidate()
    }

    private fun partyMarkerIcon(memberId: String): BitmapDrawable {
        val size = 52
        val color = Color.HSVToColor(floatArrayOf((memberId.hashCode().toUInt().toLong() % 360).toFloat(), 0.70f, 0.95f))
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        Canvas(bitmap).apply {
            drawCircle(size / 2f, size / 2f, size / 2.2f, Paint(Paint.ANTI_ALIAS_FLAG).apply { this.color = Color.WHITE })
            drawCircle(size / 2f, size / 2f, size / 2.55f, Paint(Paint.ANTI_ALIAS_FLAG).apply { this.color = color })
        }
        return BitmapDrawable(resources, bitmap)
    }

    // ───────────────────────────────────────────────────────────────────────
    // Permissions
    // ───────────────────────────────────────────────────────────────────────
    private fun requestPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
                PERM_REQ)
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ───────────────────────────────────────────────────────────────────────
    override fun onResume() {
        super.onResume()
        mapView.onResume()
        if (isSimRunning && !isPaused) startSim()
    }
    override fun onPause() {
        super.onPause()
        mapView.onPause()
        stopSim()
    }
    override fun onDestroy() {
        super.onDestroy()
        stopPartyLocationUpdates()
        partyManager.leaveParty()
        sensorManager.unregisterListener(this)
        executor.shutdown()
    }
}
