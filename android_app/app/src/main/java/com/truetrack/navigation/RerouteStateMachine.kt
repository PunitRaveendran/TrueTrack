package com.truetrack.navigation

import kotlin.math.*

/**
 * States of the navigation rerouting state machine.
 */
enum class NavigationState {
    ON_ROUTE,
    POSSIBLE_DEVIATION,
    OFF_ROUTE_ONLINE,
    RECOMPUTING,
    OFF_ROUTE_OFFLINE,
    DEGRADED_DR_MODE
}

/**
 * Event emitted upon any state change or corridor management event.
 */
data class RerouteEvent(
    val state: NavigationState,
    val timestampMs: Long,
    val reason: String,
    val activeCorridorId: String,
    val secondaryCorridorId: String? = null,
    val crossTrackErrorM: Double = 0.0,
    val statusPayload: Map<String, Any> = emptyMap()
)

/**
 * Listener for rerouting state machine notifications.
 */
interface RerouteStateListener {
    fun onStateChanged(event: RerouteEvent)
    fun onAudioAlert(message: String)
}

/**
 * Production Reroute State Machine with 5-second debouncing, dual-corridor retention,
 * and offline degraded dead-reckoning mode.
 */
class RerouteStateMachine(
    private var activeCorridorId: String = "corridor_primary",
    private val debounceThresholdSec: Double = 5.0,
    private val corridorRetentionSec: Double = 30.0,
    private val listener: RerouteStateListener? = null
) {
    var currentState: NavigationState = NavigationState.ON_ROUTE
        private set

    private var deviationStartTimeMs: Long = 0L
    private var isOnline: Boolean = false

    // Dual-corridor management
    private var secondaryCorridorId: String? = null
    private var newCorridorActivatedTimeMs: Long = 0L

    /**
     * Sets network connectivity state (true = online cellular, false = offline tunnel/dead-zone).
     */
    fun setNetworkConnectivity(online: Boolean) {
        this.isOnline = online
    }

    /**
     * Evaluates current position against the active corridor ribbon boundary.
     *
     * @param currentTimeMs Current timestamp in ms
     * @param isInsideRibbon Whether current (lat, lon) is within the ribbon polygon
     * @param crossTrackDistanceM Distance in meters from the route centerline
     */
    fun updatePosition(
        currentTimeMs: Long,
        isInsideRibbon: Boolean,
        crossTrackDistanceM: Double
    ) {
        // Handle dual-corridor retention timer
        if (secondaryCorridorId != null) {
            val elapsedSec = (currentTimeMs - newCorridorActivatedTimeMs) / 1000.0
            if (elapsedSec >= corridorRetentionSec) {
                // Retention period expired: safely discard old secondary corridor
                val oldId = secondaryCorridorId
                secondaryCorridorId = null
                emitEvent(
                    currentState,
                    "Dual-corridor retention period (${corridorRetentionSec}s) elapsed. Purged old corridor: $oldId",
                    crossTrackDistanceM
                )
            }
        }

        when (currentState) {
            NavigationState.ON_ROUTE -> {
                if (!isInsideRibbon) {
                    currentState = NavigationState.POSSIBLE_DEVIATION
                    deviationStartTimeMs = currentTimeMs
                    emitEvent(
                        currentState,
                        "Vehicle exited corridor ribbon. Starting ${debounceThresholdSec}s debounce timer.",
                        crossTrackDistanceM
                    )
                }
            }

            NavigationState.POSSIBLE_DEVIATION -> {
                if (isInsideRibbon) {
                    // False alarm (GPS jitter) — recovered within debounce window
                    currentState = NavigationState.ON_ROUTE
                    deviationStartTimeMs = 0L
                    emitEvent(
                        currentState,
                        "Vehicle re-entered corridor ribbon. Jitter filtered, resumed ON_ROUTE.",
                        crossTrackDistanceM
                    )
                } else {
                    val offDurationSec = (currentTimeMs - deviationStartTimeMs) / 1000.0
                    if (offDurationSec >= debounceThresholdSec) {
                        // Confirmed deviation
                        if (isOnline) {
                            currentState = NavigationState.OFF_ROUTE_ONLINE
                            emitEvent(
                                currentState,
                                "Confirmed off-route deviation (> ${debounceThresholdSec}s). Online reroute initiated.",
                                crossTrackDistanceM
                            )
                            listener?.onAudioAlert("Route deviation detected. Recalculating route.")
                            startRecompute(currentTimeMs, crossTrackDistanceM)
                        } else {
                            currentState = NavigationState.OFF_ROUTE_OFFLINE
                            emitEvent(
                                currentState,
                                "Confirmed off-route deviation in offline mode. Entering DEGRADED_DR_MODE.",
                                crossTrackDistanceM
                            )
                            listener?.onAudioAlert("Off route. Operating in degraded dead reckoning mode.")
                            currentState = NavigationState.DEGRADED_DR_MODE
                            emitEvent(
                                currentState,
                                "status: off_route_no_connectivity. Tracking against last known corridor manifold.",
                                crossTrackDistanceM,
                                mapOf("status" to "off_route_no_connectivity")
                            )
                        }
                    }
                }
            }

            NavigationState.OFF_ROUTE_ONLINE -> {
                // Handled in startRecompute()
            }

            NavigationState.RECOMPUTING -> {
                // Waiting for router completion callback
            }

            NavigationState.OFF_ROUTE_OFFLINE -> {
                currentState = NavigationState.DEGRADED_DR_MODE
            }

            NavigationState.DEGRADED_DR_MODE -> {
                if (isInsideRibbon) {
                    currentState = NavigationState.ON_ROUTE
                    deviationStartTimeMs = 0L
                    listener?.onAudioAlert("Re-entered active route corridor.")
                    emitEvent(
                        currentState,
                        "Vehicle merged back into corridor. Degraded mode cleared.",
                        crossTrackDistanceM
                    )
                }
            }
        }
    }

    private fun startRecompute(currentTimeMs: Long, crossTrackDistanceM: Double) {
        currentState = NavigationState.RECOMPUTING
        emitEvent(
            currentState,
            "A* router computing new corridor to destination...",
            crossTrackDistanceM
        )
    }

    /**
     * Called when the A* router successfully completes a new route.
     * Retains the old corridor for 30s as a secondary backup.
     */
    fun onRerouteCompleted(newCorridorId: String, currentTimeMs: Long) {
        secondaryCorridorId = activeCorridorId  // Keep old corridor loaded
        activeCorridorId = newCorridorId
        newCorridorActivatedTimeMs = currentTimeMs
        currentState = NavigationState.ON_ROUTE
        deviationStartTimeMs = 0L

        emitEvent(
            currentState,
            "New route active: $newCorridorId. Old corridor $secondaryCorridorId retained for ${corridorRetentionSec}s.",
            0.0,
            mapOf("retained_corridor" to (secondaryCorridorId ?: ""))
        )
        listener?.onAudioAlert("New route established.")
    }

    private fun emitEvent(
        state: NavigationState,
        reason: String,
        crossTrackM: Double = 0.0,
        extra: Map<String, Any> = emptyMap()
    ) {
        val event = RerouteEvent(
            state = state,
            timestampMs = System.currentTimeMillis(),
            reason = reason,
            activeCorridorId = activeCorridorId,
            secondaryCorridorId = secondaryCorridorId,
            crossTrackErrorM = crossTrackM,
            statusPayload = extra
        )
        listener?.onStateChanged(event)
    }
}
