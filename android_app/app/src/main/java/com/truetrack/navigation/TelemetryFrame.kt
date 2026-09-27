package com.truetrack.navigation

/**
 * A single simulation / telemetry frame produced by either:
 *  • buildSimFrames() in MainActivity (A→B synthesis)
 *  • Loading a pre-recorded JSON asset (demo routes)
 *
 * Ground-truth position:  gtLat / gtLon
 * TrueTrack fused:        trueTrackLat / trueTrackLon   (BLUE on map)
 * Naive DR drift:         naiveLat / naiveLon            (RED on map, only during blackout)
 */
data class TelemetryFrame(
    // ── Time ────────────────────────────────────────────────────────
    val t: Float,                   // seconds from route start

    // ── Kinematics ──────────────────────────────────────────────────
    val speedKmh: Float,            // ground-truth speed km/h
    val predSpeedKmh: Float,        // NPU-predicted speed km/h
    val predYawDegS: Float,         // NPU-predicted yaw rate °/s
    val headingDeg: Float,          // bearing to next waypoint (0=North)

    // ── GPS state ───────────────────────────────────────────────────
    val isBlackout: Boolean,        // true = GPS unavailable at this frame

    // ── Positions ───────────────────────────────────────────────────
    val gtLat: Double,              // ground-truth latitude
    val gtLon: Double,              // ground-truth longitude

    val naiveLat: Double,           // naive dead-reckoning (accumulates drift)
    val naiveLon: Double,

    val trueTrackLat: Double,       // TrueTrack neural-inertial fused
    val trueTrackLon: Double,

    // ── Error metrics ───────────────────────────────────────────────
    val errNaiveM: Float,           // metres of naive DR error from GT
    val errTrueTrackM: Float,       // metres of TrueTrack error from GT

    // ── Raw IMU values ──────────────────────────────────────────────
    val rawImuAx: Float,            // forward acceleration m/s²
    val rawImuAy: Float,            // lateral acceleration m/s²
    val rawImuGz: Float,            // yaw rate rad/s (z-axis gyroscope)
    val rawImuAz: Float = 9.81f,    // vertical specific force m/s² (includes gravity + road/engine vibration)
    val imuCalibrated: Boolean = false
)
