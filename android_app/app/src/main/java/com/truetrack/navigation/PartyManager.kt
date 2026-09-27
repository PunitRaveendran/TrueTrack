package com.truetrack.navigation

import android.content.Context
import android.provider.Settings
import com.google.firebase.FirebaseApp
import com.google.firebase.database.DataSnapshot
import com.google.firebase.database.DatabaseError
import com.google.firebase.database.DatabaseReference
import com.google.firebase.database.FirebaseDatabase
import com.google.firebase.database.ServerValue
import com.google.firebase.database.ValueEventListener
import java.security.SecureRandom

data class PartyMember(
    val lat: Double = 0.0,
    val lon: Double = 0.0,
    val timestamp: Long = 0L,
    val displayName: String = "Rider"
)

data class PartyRoute(
    val routeId: String,
    val coordinates: List<Pair<Double, Double>>,
    val destinationName: String,
    val destinationLat: Double,
    val destinationLon: Double
)

/** Connected-mode Firebase transport. It has no dependency on the IMU, EKF, or NPU path. */
class PartyManager(
    context: Context,
    private val onMembersChanged: (Map<String, PartyMember>) -> Unit,
    private val onRouteChanged: (PartyRoute?) -> Unit,
    private val onStatus: (String, Boolean) -> Unit
) {
    private val database: FirebaseDatabase? = run {
        if (FirebaseApp.getApps(context).isEmpty()) return@run null
        try {
            FirebaseDatabase.getInstance()
        } catch (e: Exception) {
            try {
                FirebaseDatabase.getInstance("https://truetrack-1070c-default-rtdb.firebaseio.com")
            } catch (e2: Exception) {
                null
            }
        }
    }
    private val deviceId = Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID)
    private val random = SecureRandom()
    private var roomCode: String? = null
    private var displayName: String = "Rider"
    private var membersRef: DatabaseReference? = null
    private var listener: ValueEventListener? = null
    private var routeRef: DatabaseReference? = null
    private var routeListener: ValueEventListener? = null

    fun startParty(name: String, onCreated: (String) -> Unit) {
        if (!ensureConfigured()) return
        val code = (1..6).map { "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"[random.nextInt(32)] }.joinToString("")
        joinParty(code, name, onCreated)
    }

    fun joinParty(code: String, name: String, onJoined: (String) -> Unit = {}) {
        if (!ensureConfigured()) return
        leaveParty(removeSelf = false)
        roomCode = code.trim().uppercase()
        displayName = name.ifBlank { "Rider" }.take(24)
        val partyRef = database!!.getReference("parties").child(roomCode!!)
        val ref = partyRef.child("members")
        membersRef = ref
        routeRef = partyRef.child("route")
        ref.child(deviceId).setValue(memberPayload(null, null)).addOnSuccessListener {
            attachListener(ref)
            attachRouteListener(routeRef!!)
            onStatus("Party $roomCode connected", true)
            onJoined(roomCode!!)
        }.addOnFailureListener {
            leaveParty(removeSelf = false)
            unavailable(it)
        }
    }

    fun broadcastLocation(lat: Double, lon: Double) {
        val ref = membersRef ?: return
        ref.child(deviceId).updateChildren(memberPayload(lat, lon)).addOnFailureListener { unavailable(it) }
    }

    fun publishRoute(
        coordinates: List<Pair<Double, Double>>,
        destinationName: String,
        destinationLat: Double,
        destinationLon: Double,
        routeId: String
    ) {
        val ref = routeRef ?: return
        if (coordinates.size < 2) return
        val step = maxOf(1, (coordinates.size - 1) / 798)
        val sampled = coordinates.filterIndexed { index, _ -> index % step == 0 }.toMutableList()
        if (sampled.last() != coordinates.last()) sampled += coordinates.last()
        val points = sampled.map { (lat, lon) -> mapOf("lat" to lat, "lon" to lon) }
        ref.setValue(mapOf(
            "routeId" to routeId,
            "coordinates" to points,
            "destinationName" to destinationName.take(120),
            "destinationLat" to destinationLat,
            "destinationLon" to destinationLon,
            "updatedAt" to ServerValue.TIMESTAMP
        )).addOnFailureListener { unavailable(it) }
    }

    fun leaveParty(removeSelf: Boolean = true) {
        listener?.let { membersRef?.removeEventListener(it) }
        routeListener?.let { routeRef?.removeEventListener(it) }
        listener = null
        routeListener = null
        if (removeSelf) membersRef?.child(deviceId)?.removeValue()?.addOnFailureListener { unavailable(it) }
        membersRef = null
        routeRef = null
        roomCode = null
        onMembersChanged(emptyMap())
        onRouteChanged(null)
    }

    fun activeRoomCode(): String? = roomCode
    fun localMemberId(): String = deviceId

    private fun memberPayload(lat: Double?, lon: Double?): Map<String, Any> = buildMap {
        put("displayName", displayName)
        put("timestamp", ServerValue.TIMESTAMP)
        if (lat != null && lon != null) {
            put("lat", lat)
            put("lon", lon)
        }
    }

    private fun attachListener(ref: DatabaseReference) {
        listener = object : ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                val members = snapshot.children.mapNotNull { child ->
                    child.getValue(PartyMember::class.java)?.let { child.key!! to it }
                }.toMap()
                onMembersChanged(members)
            }
            override fun onCancelled(error: DatabaseError) = unavailable(error.toException())
        }
        ref.addValueEventListener(listener!!)
    }

    private fun attachRouteListener(ref: DatabaseReference) {
        routeListener = object : ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                val coordinates = snapshot.child("coordinates").children.mapNotNull { point ->
                    val lat = point.child("lat").getValue(Double::class.javaObjectType)
                    val lon = point.child("lon").getValue(Double::class.javaObjectType)
                    if (lat != null && lon != null) lat to lon else null
                }
                val id = snapshot.child("routeId").getValue(String::class.java)
                val name = snapshot.child("destinationName").getValue(String::class.java)
                val lat = snapshot.child("destinationLat").getValue(Double::class.javaObjectType)
                val lon = snapshot.child("destinationLon").getValue(Double::class.javaObjectType)
                onRouteChanged(if (coordinates.size >= 2 && id != null && name != null && lat != null && lon != null) {
                    PartyRoute(id, coordinates, name, lat, lon)
                } else null)
            }
            override fun onCancelled(error: DatabaseError) = unavailable(error.toException())
        }
        ref.addValueEventListener(routeListener!!)
    }

    private fun unavailable(error: Exception) {
        val reason = error.localizedMessage?.takeIf { it.isNotBlank() } ?: "check connection and Firebase rules"
        onStatus("Party connection failed: $reason", false)
    }

    private fun ensureConfigured(): Boolean {
        if (database != null) return true
        onStatus("Party setup required - add Firebase config", false)
        return false
    }
}
