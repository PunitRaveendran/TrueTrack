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

/** Connected-mode Firebase transport. It has no dependency on the IMU, EKF, or NPU path. */
class PartyManager(
    context: Context,
    private val onMembersChanged: (Map<String, PartyMember>) -> Unit,
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
        val ref = database!!.getReference("parties").child(roomCode!!).child("members")
        membersRef = ref
        ref.child(deviceId).setValue(memberPayload(null, null)).addOnSuccessListener {
            attachListener(ref)
            onStatus("Party $roomCode connected", true)
            onJoined(roomCode!!)
        }.addOnFailureListener { unavailable(it) }
    }

    fun broadcastLocation(lat: Double, lon: Double) {
        val ref = membersRef ?: return
        ref.child(deviceId).updateChildren(memberPayload(lat, lon)).addOnFailureListener { unavailable(it) }
    }

    fun leaveParty(removeSelf: Boolean = true) {
        listener?.let { membersRef?.removeEventListener(it) }
        listener = null
        if (removeSelf) membersRef?.child(deviceId)?.removeValue()?.addOnFailureListener { unavailable(it) }
        membersRef = null
        roomCode = null
        onMembersChanged(emptyMap())
    }

    fun activeRoomCode(): String? = roomCode

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
