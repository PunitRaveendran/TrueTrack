package com.truetrack.navigation

import kotlin.math.*

/**
 * TrueTrack - On-Device Lean Angle Estimator & R_x(phi) De-roll Module
 *
 * Implements a high-rate (50 Hz) complementary filter estimating roll angle phi(t)
 * and performs the R_x(-phi) de-roll rotation to remove gravity-projection bias
 * from lateral acceleration before neural network velocity inference and NHC updates.
 */
class LeanCorrector(
    private val sampleRate: Float = 50.0f,
    private val alpha: Float = 0.96f,
    private val gNominal: Float = 9.80665f,
    private val accelGate: Float = 2.5f
) {
    private val dt: Float = 1.0f / sampleRate
    var currentPhiRad: Float = 0.0f
        private set

    val currentPhiDeg: Float
        get() = Math.toDegrees(currentPhiRad.toDouble()).toFloat()

    /**
     * Update the roll lean angle estimate with a 6-axis IMU reading.
     *
     * @param ax Forward acceleration (m/s^2)
     * @param ay Lateral acceleration (m/s^2)
     * @param az Vertical acceleration (m/s^2)
     * @param gx Gyro roll rate (rad/s)
     * @return Estimated roll angle phi in radians
     */
    fun update(ax: Float, ay: Float, az: Float, gx: Float): Float {
        // 1. Gyro integration
        val phiPred = currentPhiRad + gx * dt

        // 2. Accel norm check
        val accelMag = sqrt(ax * ax + ay * ay + az * az)

        // 3. Adaptive gating: when total accel is near 1g, trust gravity vector
        currentPhiRad = if (abs(accelMag - gNominal) < accelGate) {
            val phiMeas = atan2(ay, az)
            alpha * phiPred + (1.0f - alpha) * phiMeas
        } else {
            phiPred
        }

        // Clamp to physically realistic limits (+/- 45 degrees)
        val maxPhi = Math.toRadians(45.0).toFloat()
        currentPhiRad = currentPhiRad.coerceIn(-maxPhi, maxPhi)

        return currentPhiRad
    }

    /**
     * Apply R_x(-phi) rotation around longitudinal axis.
     * De-rolls lateral (ay) and vertical (az) accelerations so that gravity
     * does not contaminate lateral vehicle dynamics.
     *
     * @return FloatArray of size 3: [ax, ay_derolled, az_derolled]
     */
    fun derollAccelerations(ax: Float, ay: Float, az: Float, phiRad: Float = currentPhiRad): FloatArray {
        val cosPhi = cos(phiRad)
        val sinPhi = sin(phiRad)

        val ayDerolled = ay * cosPhi - az * sinPhi
        val azDerolled = ay * sinPhi + az * cosPhi

        return floatArrayOf(ax, ayDerolled, azDerolled)
    }

    /**
     * Compute Non-Holonomic Constraint lateral slip residual.
     * In the de-rolled frame, lateral velocity should be zero.
     */
    fun computeNhcResidual(vy: Float, vz: Float, phiRad: Float = currentPhiRad): Float {
        val cosPhi = cos(phiRad)
        val sinPhi = sin(phiRad)
        return vy * cosPhi - vz * sinPhi
    }

    fun reset() {
        currentPhiRad = 0.0f
    }
}
