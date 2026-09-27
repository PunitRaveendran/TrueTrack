package com.truetrack.navigation

import kotlin.math.*

/**
 * TrueTrack - On-Device Lean Angle Estimator & 3D Acceleration De-roll Module
 *
 * Implements a high-rate complementary filter estimating vehicle roll angle phi(t)
 * calibrated for Android device coordinate frames (X: Lateral, Y: Longitudinal, Z: Vertical/Normal).
 * Performs R_x(-phi) de-rolling to isolate true dynamic acceleration from gravity.
 */
class LeanCorrector(
    private val sampleRate: Float = 50.0f,
    private val alpha: Float = 0.96f,
    private val gNominal: Float = 9.80665f,
    private val accelGate: Float = 3.5f
) {
    private val dt: Float = 1.0f / sampleRate
    var currentPhiRad: Float = 0.0f
        private set

    val currentPhiDeg: Float
        get() = Math.toDegrees(currentPhiRad.toDouble()).toFloat()

    /**
     * Update the roll lean angle estimate with 6-axis IMU readings.
     *
     * In Android portrait phone frame:
     * - ax: Lateral axis (left/right tilt)
     * - ay: Longitudinal axis (forward/backward)
     * - az: Out of screen (vertical when flat, normal when held)
     *
     * Lateral roll angle is measured by lateral gravity projection ax relative to
     * the orthogonal gravity vector sqrt(ay^2 + az^2).
     *
     * @param ax Lateral acceleration (m/s^2)
     * @param ay Longitudinal acceleration (m/s^2)
     * @param az Normal/Vertical acceleration (m/s^2)
     * @param gx Gyro roll rate (rad/s)
     * @return Estimated roll angle phi in radians
     */
    fun update(ax: Float, ay: Float, az: Float, gx: Float): Float {
        // 1. Gyro integration (Right tilt = positive roll angle)
        val phiPred = currentPhiRad - gx * dt

        // 2. 3D Acceleration magnitude (Total gravity + dynamic forces)
        val accelMag = sqrt(ax * ax + ay * ay + az * az)

        // 3. Adaptive gravity gating: when total accel is near 1g, trust gravity vector
        currentPhiRad = if (abs(accelMag - gNominal) < accelGate) {
            // Lateral tilt angle relative to total orthogonal gravity
            val gravOrthogonal = max(0.5f, sqrt(ay * ay + az * az))
            val phiMeas = atan2(-ax, gravOrthogonal)
            alpha * phiPred + (1.0f - alpha) * phiMeas
        } else {
            phiPred
        }

        // Smooth clamp to physically valid two-wheeler limits (+/- 75 degrees)
        val maxPhi = Math.toRadians(75.0).toFloat()
        currentPhiRad = currentPhiRad.coerceIn(-maxPhi, maxPhi)

        return currentPhiRad
    }

    /**
     * Apply R_x(-phi) rotation around longitudinal axis.
     * De-rolls lateral (ax) and vertical (az) accelerations so that gravity
     * does not contaminate lateral vehicle dynamics.
     *
     * @return FloatArray of size 3: [ax_derolled, ay, az_derolled]
     */
    fun derollAccelerations(ax: Float, ay: Float, az: Float, phiRad: Float = currentPhiRad): FloatArray {
        val cosPhi = cos(phiRad)
        val sinPhi = sin(phiRad)

        // Rotate lateral and normal components by estimated lean angle
        val axDerolled = ax * cosPhi - az * sinPhi
        val azDerolled = ax * sinPhi + az * cosPhi

        return floatArrayOf(axDerolled, ay, azDerolled)
    }

    /**
     * Compute Non-Holonomic Constraint lateral slip residual.
     */
    fun computeNhcResidual(vx: Float, vz: Float, phiRad: Float = currentPhiRad): Float {
        val cosPhi = cos(phiRad)
        val sinPhi = sin(phiRad)
        return vx * cosPhi - vz * sinPhi
    }

    fun reset() {
        currentPhiRad = 0.0f
    }
}
