package com.truetrack.navigation

class NavigationForegroundService : android.app.Service() {
    private val handler = android.os.Handler(android.os.Looper.getMainLooper())
    private val heartbeat = object : Runnable {
        override fun run() {
            val now = System.currentTimeMillis()
            lastHeartbeatTimestampMs = now
            android.util.Log.i(TAG, "Navigation service heartbeat at $now")
            handler.postDelayed(this, HEARTBEAT_INTERVAL_MS)
        }
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification())
        handler.post(heartbeat)
    }

    override fun onStartCommand(intent: android.content.Intent?, flags: Int, startId: Int): Int = START_STICKY

    override fun onDestroy() {
        handler.removeCallbacks(heartbeat)
        super.onDestroy()
    }

    override fun onBind(intent: android.content.Intent?): android.os.IBinder? = null

    private fun buildNotification(): android.app.Notification = androidx.core.app.NotificationCompat.Builder(this, CHANNEL_ID)
        .setSmallIcon(android.R.drawable.ic_menu_mylocation)
        .setContentTitle(getString(R.string.app_name))
        .setContentText("Navigation sensors active")
        .setOngoing(true)
        .setCategory(androidx.core.app.NotificationCompat.CATEGORY_SERVICE)
        .build()

    private fun createNotificationChannel() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            val channel = android.app.NotificationChannel(
                CHANNEL_ID,
                "Navigation tracking",
                android.app.NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(android.app.NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    companion object {
        private const val TAG = "NavigationService"
        private const val CHANNEL_ID = "navigation_tracking"
        private const val NOTIFICATION_ID = 4101
        private const val HEARTBEAT_INTERVAL_MS = 30_000L

        @Volatile
        var lastHeartbeatTimestampMs: Long = 0L
            private set
    }
}
