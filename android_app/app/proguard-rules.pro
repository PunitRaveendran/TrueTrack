# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# Preserve ONNX Runtime classes
-keep class ai.onnxruntime.** { *; }

# Preserve WebSocket server classes
-keep class org.java_websocket.** { *; }

# Preserve TrueTrack navigation core
-keep class com.truetrack.navigation.** { *; }
