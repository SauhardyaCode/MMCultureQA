"""
Unit tests for SpeechTranscriber and ASR audio handling.
"""

import unittest
from pathlib import Path

from mmcultureqa.asr import SpeechTranscriber
from mmcultureqa.config import Config


class TestASR(unittest.TestCase):
    """Test suite for ASR transcription and audio inspection."""

    def setUp(self):
        self.cfg = Config()
        self.transcriber = SpeechTranscriber()

    def test_audio_inspection(self):
        """Test reading audio metadata from downloaded WAV files."""
        audio_files = list(self.cfg.audio_dir.glob("*/*.wav"))
        if not audio_files:
            self.skipTest("No audio files available for testing")

        sample_audio = audio_files[0]
        meta = self.transcriber.inspect_audio(sample_audio)
        self.assertIn("channels", meta)
        self.assertIn("duration_seconds", meta)
        self.assertGreater(meta["duration_seconds"], 0)

    def test_transcribe_english_with_cache(self):
        """Test transcribing English audio question and caching behavior."""
        en_audio = list((self.cfg.audio_dir / "en").glob("*.wav"))
        if not en_audio:
            self.skipTest("No English audio files found")

        audio_file = en_audio[0]
        # First call (performs ASR or reads cache)
        text1 = self.transcriber.transcribe(audio_file, lang="en", use_cache=True)
        self.assertTrue(isinstance(text1, str))
        self.assertGreater(len(text1), 0)

        # Second call should immediately hit the local cache
        text2 = self.transcriber.transcribe(audio_file, lang="en", use_cache=True)
        self.assertEqual(text1, text2)


if __name__ == "__main__":
    unittest.main()
