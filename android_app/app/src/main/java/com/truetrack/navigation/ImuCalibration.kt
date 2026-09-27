package com.truetrack.navigation

import android.content.Context
import kotlin.math.sqrt

data class ImuCalibration(
    val biasGx: Float,
    val biasGy: Float,
    val biasGz: Float,
    val upX: Float,
    val upY: Float,
    val upZ: Float,
    val gravityMagnitude: Float
) {
    val biasAlongVertical: Float
        get() = biasGx * upX + biasGy * upY + biasGz * upZ
}

/** Captures stationary gyro bias and the device-frame vertical axis. */
class ImuCalibrator(context: Context) {
    companion object {
        private const val PREFS = "imu_calibration"
        private const val REQUIRED_SAMPLES = 100
        private const val MIN_WINDOW_NS = 2_000_000_000L
        private const val MAX_WINDOW_NS = 12_000_000_000L
        private const val MAX_PAIR_AGE_NS = 80_000_000L
    }

    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    var calibration: ImuCalibration? = load()
        private set
    private var collecting = false
    private var startedNs = 0L
    private var count = 0
    private var sumAx = 0.0
    private var sumAy = 0.0
    private var sumAz = 0.0
    private var sumGx = 0.0
    private var sumGy = 0.0
    private var sumGz = 0.0

    fun begin(timestampNs: Long) {
        collecting = true
        startedNs = timestampNs
        clearWindow()
    }

    /** Returns a progress percentage, 100 on success, or -1 when idle. */
    fun addSample(
        timestampNs: Long,
        accel: FloatArray,
        gyro: FloatArray,
        gyroTimestampNs: Long
    ): Int {
        if (!collecting) return -1
        val elapsed = timestampNs - startedNs
        if (elapsed > MAX_WINDOW_NS) {
            collecting = false
            clearWindow()
            return -2
        }
        if (kotlin.math.abs(timestampNs - gyroTimestampNs) > MAX_PAIR_AGE_NS) return count * 100 / REQUIRED_SAMPLES

        val accelMagnitude = norm(accel)
        val gyroMagnitude = norm(gyro)
        if (accelMagnitude !in 9.2f..10.4f || gyroMagnitude > 0.12f) {
            clearWindow()
            return 0
        }

        sumAx += accel[0]; sumAy += accel[1]; sumAz += accel[2]
        sumGx += gyro[0]; sumGy += gyro[1]; sumGz += gyro[2]
        count++
        if (count >= REQUIRED_SAMPLES && elapsed >= MIN_WINDOW_NS) {
            finish()
            return 100
        }
        return (count * 100 / REQUIRED_SAMPLES).coerceAtMost(99)
    }

    fun isCollecting() = collecting

    fun correctedYawRate(gx: Float, gy: Float, gz: Float): Float {
        val c = calibration ?: return Float.NaN
        return (gx - c.biasGx) * c.upX + (gy - c.biasGy) * c.upY + (gz - c.biasGz) * c.upZ
    }

    private fun finish() {
        val ax = (sumAx / count).toFloat()
        val ay = (sumAy / count).toFloat()
        val az = (sumAz / count).toFloat()
        val g = sqrt(ax * ax + ay * ay + az * az)
        calibration = ImuCalibration(
            (sumGx / count).toFloat(), (sumGy / count).toFloat(), (sumGz / count).toFloat(),
            ax / g, ay / g, az / g, g
        ).also { c ->
            prefs.edit()
                .putFloat("bias_gx", c.biasGx).putFloat("bias_gy", c.biasGy).putFloat("bias_gz", c.biasGz)
                .putFloat("up_x", c.upX).putFloat("up_y", c.upY).putFloat("up_z", c.upZ)
                .putFloat("gravity", c.gravityMagnitude).apply()
        }
        collecting = false
    }

    private fun load(): ImuCalibration? {
        if (!prefs.contains("bias_gx")) return null
        return ImuCalibration(
            prefs.getFloat("bias_gx", 0f), prefs.getFloat("bias_gy", 0f), prefs.getFloat("bias_gz", 0f),
            prefs.getFloat("up_x", 0f), prefs.getFloat("up_y", 0f), prefs.getFloat("up_z", 1f),
            prefs.getFloat("gravity", 9.81f)
        )
    }

    private fun clearWindow() {
        count = 0
        sumAx = 0.0; sumAy = 0.0; sumAz = 0.0
        sumGx = 0.0; sumGy = 0.0; sumGz = 0.0
    }

    private fun norm(v: FloatArray) = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}
