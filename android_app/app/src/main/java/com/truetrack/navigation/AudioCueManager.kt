package com.truetrack.navigation

import android.content.Context
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

/**
 * TrueTrack - Audio Cue & Acoustic Status Synthesizer
 *
 * Provides real-time tactical acoustic feedback and spoken navigation cues
 * when entering/exiting GNSS blackout zones, switching routes, and negotiating underground forks.
 */
class AudioCueManager(private val context: Context) : TextToSpeech.OnInitListener {

    private var tts: TextToSpeech? = null
    private var isTtsReady = false
    private var toneGenerator: ToneGenerator? = null
    var isEnabled: Boolean = true
    private val pendingUtterances = mutableListOf<String>()

    init {
        try {
            tts = TextToSpeech(context.applicationContext, this)
            toneGenerator = ToneGenerator(AudioManager.STREAM_MUSIC, 85)
        } catch (e: Exception) {
            Log.e("AudioCueManager", "Failed to initialize audio components", e)
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            var result = tts?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                result = tts?.setLanguage(Locale.getDefault())
            }
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                result = tts?.setLanguage(Locale.ENGLISH)
            }
            isTtsReady = (result != TextToSpeech.LANG_MISSING_DATA && result != TextToSpeech.LANG_NOT_SUPPORTED)
            tts?.setSpeechRate(1.05f)
            Log.i("AudioCueManager", "TTS initialized successfully (isTtsReady=$isTtsReady)")

            synchronized(pendingUtterances) {
                for (text in pendingUtterances) {
                    speak(text)
                }
                pendingUtterances.clear()
            }
        } else {
            Log.e("AudioCueManager", "TextToSpeech init failed with status: $status")
        }
    }

    /**
     * Announces system online status.
     */
    fun onSystemOnline(routeName: String = "HITEC City Corridor") {
        if (!isEnabled) return
        speak("TrueTrack navigation active. Route: $routeName.")
    }

    /**
     * Trigger alert when entering an underpass / losing satellite fix.
     */
    fun onBlackoutEntered() {
        if (!isEnabled) return
        try {
            toneGenerator?.startTone(ToneGenerator.TONE_PROP_BEEP2, 250)
            speak("GPS blackout active. Neural dead reckoning engaged.")
        } catch (e: Exception) {
            Log.e("AudioCueManager", "Error playing blackout alert", e)
        }
    }

    /**
     * Trigger alert when GPS fix is restored.
     */
    fun onGpsRestored() {
        if (!isEnabled) return
        try {
            toneGenerator?.startTone(ToneGenerator.TONE_PROP_ACK, 200)
            speak("GPS locked. Continuous trajectory reconciled.")
        } catch (e: Exception) {
            Log.e("AudioCueManager", "Error playing restoration alert", e)
        }
    }

    /**
     * Voice guidance for underground turns / lane constraints.
     */
    fun onUndergroundTurnGuidance(instruction: String) {
        if (!isEnabled) return
        speak(instruction)
    }

    fun speak(text: String) {
        if (!isEnabled) return
        if (!isTtsReady) {
            synchronized(pendingUtterances) {
                pendingUtterances.add(text)
            }
            Log.d("AudioCueManager", "TTS not ready yet, queued: $text")
            return
        }
        try {
            val params = Bundle().apply {
                putInt(TextToSpeech.Engine.KEY_PARAM_STREAM, AudioManager.STREAM_MUSIC)
                putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, 1.0f)
            }
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, params, "TrueTrackCue_${System.currentTimeMillis()}")
            Log.i("AudioCueManager", "Speaking: $text")
        } catch (e: Exception) {
            Log.e("AudioCueManager", "Error speaking text", e)
        }
    }

    fun shutdown() {
        try {
            tts?.stop()
            tts?.shutdown()
            toneGenerator?.release()
        } catch (e: Exception) {
            Log.e("AudioCueManager", "Error releasing audio resources", e)
        }
    }
}
