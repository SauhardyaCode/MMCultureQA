"""
Multilingual Automatic Speech Recognition (ASR) module for Task 1 (Spoken Visual QA).
Transcribes spoken audio questions in English, Arabic varieties, and other regional languages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import speech_recognition as sr

from .config import Config, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


class SpeechTranscriber:
    """Handles audio question transcription for Task 1 (Spoken Visual QA)."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        default_engine: str = "google",
    ) -> None:
        """Initialize the transcriber with caching and ASR backend.

        Args:
            cache_dir: Directory where transcriptions are cached to save computation/API calls.
            default_engine: Transcription engine ('google' or 'whisper').
        """
        self.recognizer = sr.Recognizer()
        self.cache_dir = cache_dir or (Config().data_dir / ".cache" / "asr")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_engine = default_engine

    def _compute_audio_hash(self, audio_path: Path) -> str:
        """Compute sha256 hash of an audio file for cache indexing."""
        hasher = hashlib.sha256()
        with open(audio_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_cache_path(self, audio_hash: str, lang: str) -> Path:
        """Generate path to cached transcription JSON file."""
        return self.cache_dir / f"{audio_hash}_{lang}.json"

    def get_cached_transcription(self, audio_path: Path, lang: str) -> Optional[str]:
        """Retrieve transcription from local cache if present."""
        if not audio_path.exists():
            return None
        audio_hash = self._compute_audio_hash(audio_path)
        cache_file = self._get_cache_path(audio_hash, lang)
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                return data.get("transcription")
            except Exception as e:
                logger.warning(f"Failed to read ASR cache {cache_file}: {e}")
        return None

    def save_cache(self, audio_path: Path, lang: str, transcription: str) -> None:
        """Save transcription result into local cache."""
        audio_hash = self._compute_audio_hash(audio_path)
        cache_file = self._get_cache_path(audio_hash, lang)
        try:
            cache_file.write_text(
                json.dumps({"audio_path": str(audio_path), "lang": lang, "transcription": transcription}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to write ASR cache {cache_file}: {e}")

    def inspect_audio(self, audio_path: Union[str, Path]) -> Dict[str, Union[int, float, str]]:
        """Inspect audio parameters (channels, sample rate, duration)."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            with wave.open(str(path), "rb") as w:
                channels = w.getnchannels()
                sample_width = w.getsampwidth()
                frame_rate = w.getframerate()
                num_frames = w.getnframes()
                duration = num_frames / float(frame_rate) if frame_rate > 0 else 0.0
                return {
                    "channels": channels,
                    "sample_width": sample_width,
                    "frame_rate": frame_rate,
                    "num_frames": num_frames,
                    "duration_seconds": round(duration, 2),
                    "path": str(path),
                }
        except Exception as e:
            return {"error": str(e), "path": str(path)}

    def transcribe(
        self,
        audio_path: Union[str, Path],
        lang: str = "en",
        use_cache: bool = True,
        engine: Optional[str] = None,
    ) -> str:
        """Transcribe an audio question into text.

        Args:
            audio_path: Path to the audio WAV file.
            lang: Target language track ('en', 'msa', 'arz', 'ajp', etc.).
            use_cache: If True, check and save results in cache.
            engine: Transcription backend ('google', 'whisper'). Defaults to self.default_engine.

        Returns:
            Transcribed question string.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {path}")

        engine = engine or self.default_engine

        # Check cache first
        if use_cache:
            cached = self.get_cached_transcription(path, lang)
            if cached is not None:
                return cached

        # Resolve ASR language code
        cfg = Config()
        asr_lang_code = cfg.get_asr_language_code(lang)

        # Transcribe using selected engine
        transcription = ""
        try:
            with sr.AudioFile(str(path)) as source:
                audio_data = self.recognizer.record(source)

            if engine == "google":
                transcription = self.recognizer.recognize_google(
                    audio_data,
                    language=asr_lang_code,
                )
            else:
                # Fallback to Google if unknown engine specified
                transcription = self.recognizer.recognize_google(
                    audio_data,
                    language=asr_lang_code,
                )

        except sr.UnknownValueError:
            logger.warning(f"ASR could not understand audio in {path} ({lang})")
            transcription = ""
        except sr.RequestError as e:
            logger.error(f"ASR service request failed: {e}")
            raise RuntimeError(f"ASR request error: {e}") from e

        # Normalize and clean transcription
        transcription = transcription.strip()

        # Cache the result
        if use_cache and transcription:
            self.save_cache(path, lang, transcription)

        return transcription
