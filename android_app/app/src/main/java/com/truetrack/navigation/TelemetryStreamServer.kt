package com.truetrack.navigation

import android.util.Log
import org.java_websocket.WebSocket
import org.java_websocket.handshake.ClientHandshake
import org.java_websocket.server.WebSocketServer
import org.json.JSONObject
import java.net.InetSocketAddress
import java.util.concurrent.CopyOnWriteArraySet

/**
 * TrueTrack - Telemetry Stream Server (Green Light Bridge)
 *
 * Lightweight local WebSocket server running on the iQOO phone (default port 8765).
 * Streams live sensor/model telemetry to the laptop cockpit. The activity currently
 * broadcasts at the ONNX inference cadence (about 10 Hz), not at the 50 Hz sensor rate.
 */
class TelemetryStreamServer(port: Int = 8765) : WebSocketServer(InetSocketAddress("127.0.0.1", port)) {

    private val connectedClients = CopyOnWriteArraySet<WebSocket>()

    override fun onOpen(conn: WebSocket?, handshake: ClientHandshake?) {
        conn?.let {
            connectedClients.add(it)
            Log.i("TelemetryServer", "Laptop cockpit connected: ${it.remoteSocketAddress}")
        }
    }

    override fun onClose(conn: WebSocket?, code: Int, reason: String?, remote: Boolean) {
        conn?.let {
            connectedClients.remove(it)
            Log.i("TelemetryServer", "Laptop cockpit disconnected: ${it.remoteSocketAddress}")
        }
    }

    override fun onMessage(conn: WebSocket?, message: String?) {
        // The phone bridge is telemetry-only; incoming commands are logged, not executed.
        try {
            message?.let {
                val json = JSONObject(it)
                if (json.has("command")) {
                    Log.i("TelemetryServer", "Received command from laptop: ${json.getString("command")}")
                }
            }
        } catch (e: Exception) {
            Log.e("TelemetryServer", "Error parsing incoming cockpit message", e)
        }
    }

    override fun onError(conn: WebSocket?, ex: Exception?) {
        Log.e("TelemetryServer", "WebSocket error: ${ex?.message}")
    }

    override fun onStart() {
        Log.i("TelemetryServer", "Telemetry stream server started on port $port")
    }

    /**
     * Broadcast one telemetry frame at the caller's inference cadence (about 10 Hz).
     */
    fun broadcastTelemetry(
        timestampMs: Long,
        speedKmh: Float,
        yawRateDeg: Float,
        leanDeg: Float,
        isBlackout: Boolean,
        inferenceLatencyMs: Float,
        ax: Float,
        ayDerolled: Float,
        az: Float,
        lat: Double,
        lon: Double,
        headingDeg: Float
    ) {
        if (connectedClients.isEmpty()) return

        val json = JSONObject().apply {
            put("t", timestampMs)
            put("speed", speedKmh)
            put("yaw", yawRateDeg)
            put("lean", leanDeg)
            put("blackout", isBlackout)
            put("latency", inferenceLatencyMs)
            put("ax", ax)
            put("ay", ayDerolled)
            put("az", az)
            put("lat", lat)
            put("lon", lon)
            put("heading", headingDeg)
        }

        val payload = json.toString()
        for (client in connectedClients) {
            if (client.isOpen) {
                client.send(payload)
            }
        }
    }
}
