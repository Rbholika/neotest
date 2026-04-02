"""
tools/voice_tool.py — Text-to-Speech & Speech-to-Text
=======================================================
TTS:  pyttsx3 (offline, no API key needed)
STT:  SpeechRecognition with Google Web Speech API (fallback: Whisper)

Usage:
    from tools.voice_tool import VoiceTool
    voice = VoiceTool()

    # Text → Speech
    voice.speak("Analysis complete. Three critical incidents found.")

    # Speech → Text
    text = voice.listen()
    print(text)

    # Save speech to file
    voice.save_speech("Report summary here.", "output.mp3")
"""

import os
import threading
from typing import Optional

from core.logger import get_logger
from config import CONFIG

logger = get_logger("tool.voice")


class VoiceTool:

    def __init__(self):
        self._tts_engine  = None
        self._tts_lock    = threading.Lock()

    # ── TTS ───────────────────────────────────────────────────────────────────

    def _init_tts(self):
        if self._tts_engine is not None:
            return
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate",   CONFIG.TTS_RATE)
            engine.setProperty("volume", CONFIG.TTS_VOLUME)

            # Prefer a clearer voice if available
            voices = engine.getProperty("voices")
            for v in voices:
                if "english" in v.name.lower() or "zira" in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break

            self._tts_engine = engine
            logger.info("TTS engine initialised (pyttsx3)")
        except ImportError:
            logger.warning("pyttsx3 not installed — TTS unavailable. pip install pyttsx3")
        except Exception as exc:
            logger.error(f"TTS init failed: {exc}")

    def speak(self, text: str, blocking: bool = True):
        """Convert text to speech and play it."""
        if not text or not text.strip():
            return

        self._init_tts()
        if self._tts_engine is None:
            logger.warning("TTS unavailable — printing to console instead.")
            print(f"[TTS] {text}")
            return

        # Truncate very long texts for speech
        if len(text) > 2000:
            text = text[:2000] + " … (truncated for speech)"

        with self._tts_lock:
            try:
                self._tts_engine.say(text)
                if blocking:
                    self._tts_engine.runAndWait()
                else:
                    t = threading.Thread(
                        target=self._tts_engine.runAndWait, daemon=True
                    )
                    t.start()
                logger.debug(f"TTS played {len(text)} chars")
            except Exception as exc:
                logger.error(f"TTS speak failed: {exc}")

    def stop_speaking(self):
        """Stop ongoing speech."""
        if self._tts_engine:
            try:
                self._tts_engine.stop()
            except Exception:
                pass

    def save_speech(self, text: str, filename: str = "speech_output.mp3") -> Optional[str]:
        """Save TTS output to an audio file (requires pyttsx3 + driver support)."""
        self._init_tts()
        if self._tts_engine is None:
            return None

        os.makedirs(CONFIG.EXPORT_DIR, exist_ok=True)
        path = os.path.join(CONFIG.EXPORT_DIR, filename)

        try:
            self._tts_engine.save_to_file(text, path)
            self._tts_engine.runAndWait()
            logger.info(f"TTS saved → {path}")
            return path
        except Exception as exc:
            logger.error(f"TTS save failed: {exc}")
            return None

    # ── STT ───────────────────────────────────────────────────────────────────

    def listen(
        self,
        timeout: int = 10,
        phrase_limit: int = 30,
        use_whisper: bool = False,
    ) -> Optional[str]:
        """
        Listen from the microphone and return transcribed text.

        Args:
            timeout:      Seconds to wait before giving up listening
            phrase_limit: Max seconds of speech to capture
            use_whisper:  Use local Whisper model instead of Google API

        Returns:
            Transcribed string or None on failure
        """
        try:
            import speech_recognition as sr
        except ImportError:
            logger.error(
                "speech_recognition not installed. "
                "pip install SpeechRecognition pyaudio"
            )
            return None

        recogniser = sr.Recognizer()
        recogniser.energy_threshold    = 300
        recogniser.dynamic_energy_threshold = True
        recogniser.pause_threshold     = 0.8

        try:
            with sr.Microphone() as source:
                logger.info("Calibrating microphone for ambient noise …")
                recogniser.adjust_for_ambient_noise(source, duration=1)
                logger.info("Listening … speak now")

                audio = recogniser.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit,
                )

            if use_whisper:
                return self._transcribe_whisper(audio, recogniser)
            else:
                return self._transcribe_google(audio, recogniser)

        except sr.WaitTimeoutError:
            logger.warning("No speech detected within timeout.")
            return None
        except Exception as exc:
            logger.error(f"Microphone error: {exc}")
            return None

    def _transcribe_google(self, audio, recogniser) -> Optional[str]:
        """Use Google Web Speech API (requires internet)."""
        import speech_recognition as sr
        try:
            text = recogniser.recognize_google(audio)
            logger.info(f"STT (Google) transcribed: {text[:80]}")
            return text
        except sr.UnknownValueError:
            logger.warning("Google STT: speech not understood")
            return None
        except sr.RequestError as exc:
            logger.error(f"Google STT API error: {exc}")
            return None

    def _transcribe_whisper(self, audio, recogniser) -> Optional[str]:
        """Use local Whisper model (offline — requires openai-whisper)."""
        try:
            import speech_recognition as sr
            text = recogniser.recognize_whisper(
                audio, model="base", language="english"
            )
            logger.info(f"STT (Whisper) transcribed: {text[:80]}")
            return text
        except Exception as exc:
            logger.error(f"Whisper STT failed: {exc}")
            return None

    def transcribe_file(self, audio_path: str) -> Optional[str]:
        """Transcribe an existing audio file (WAV, FLAC, AIFF)."""
        try:
            import speech_recognition as sr
            recogniser = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio = recogniser.record(source)
            return self._transcribe_google(audio, recogniser)
        except Exception as exc:
            logger.error(f"File transcription failed: {exc}")
            return None


# Module-level singleton
voice_tool = VoiceTool()
