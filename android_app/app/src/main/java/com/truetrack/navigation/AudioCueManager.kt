package com.truetrack.navigation

import android.content.Context
import android.media.AudioManager
import android.media.ToneGenerator
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

/**
 * TrueTrack - Audio Cue & Acoustic Status Synthesizer
 *
 * Provides real-time tactical acoustic feedback and spoken navigation cues
 * when entering/exiting GNSS blackout zones and negotiating underground forks.
 */
class AudioCueManager(private val context: Context) : TextToSpeech.OnInitListener {

    private var tts: TextToSpeech? = null
    private var isTtsReady = false
    private var toneGenerator: ToneGenerator? = null
    var isEnabled: Boolean = true

    init {
        try {
            tts = TextToSpeech(context.applicationContext, this)
            toneGenerator = ToneGenerator(AudioManager.STREAM_NOTIFICATION, 80)
        } catch (e: Exception) {
            Log.e("AudioCueManager", "Failed to initialize audio components", e)
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US)
            isTtsReady = result != TextToSpeech.LANG_MISSING_DATA && result != TextToSpeech.LANG_NOT_SUPPORTED
            tts?.setSpeechRate(1.05f)
        }
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

    private fun speak(text: String) {
        if (isTtsReady && isEnabled) {
            tts?.speak(text, TextToSpeech.QUEUE_ADD, null, "TrueTrackCue_${System.currentTimeMillis()}")
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
