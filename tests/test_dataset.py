"""
Unit tests for MMCultureQA dataset loader and schema validation.
"""

import unittest
from pathlib import Path

from mmcultureqa.config import Config
from mmcultureqa.dataset import MMCultureQADataset, MMCultureQARecord


class TestMMCultureQADataset(unittest.TestCase):
    """Test suite for dataset parsing, integrity, and statistics."""

    def setUp(self):
        self.cfg = Config()

    def test_load_train_qa_en(self):
        """Test loading English text QA training split."""
        train_path = self.cfg.qa_dir / "train_en.jsonl"
        if not train_path.exists():
            self.skipTest(f"Split file {train_path} not found")

        ds = MMCultureQADataset.from_jsonl(train_path, task="qa", lang="en")
        self.assertGreater(len(ds), 0)
        first = ds[0]
        self.assertTrue(first.id)
        self.assertTrue(first.image)
        self.assertTrue(first.question)
        self.assertTrue(first.answer)

    def test_media_resolution_and_validation(self):
        """Test image and audio resolution on disk."""
        dev_qa = self.cfg.qa_dir / "dev_en.jsonl"
        if not dev_qa.exists():
            self.skipTest("Dev qa file not found")

        ds = MMCultureQADataset.from_jsonl(dev_qa, task="qa", lang="en")
        report = ds.validate_integrity()
        self.assertTrue(report["valid"], f"Integrity failed: {report}")
        self.assertEqual(report["missing_images_count"], 0)

    def test_sqa_audio_verification(self):
        """Test Task 1 Spoken QA audio resolution."""
        train_sqa = self.cfg.sqa_dir / "train_en.jsonl"
        if not train_sqa.exists():
            self.skipTest("Train sqa file not found")

        ds = MMCultureQADataset.from_jsonl(train_sqa, task="sqa", lang="en")
        self.assertGreater(len(ds), 0)
        first = ds[0]
        self.assertIsNotNone(first.audio)
        audio_path = first.resolve_audio_path()
        self.assertIsNotNone(audio_path)
        self.assertTrue(audio_path.exists())

    def test_summary_statistics(self):
        """Test country and category distribution calculation."""
        train_path = self.cfg.qa_dir / "train_en.jsonl"
        ds = MMCultureQADataset.from_jsonl(train_path, task="qa", lang="en")
        stats = ds.summary_statistics()
        self.assertEqual(stats["total"], len(ds))
        self.assertTrue(len(stats["countries"]) > 0)
        self.assertTrue(len(stats["categories"]) > 0)


if __name__ == "__main__":
    unittest.main()
