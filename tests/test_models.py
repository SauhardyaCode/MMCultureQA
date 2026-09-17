"""
Unit tests for MMCultureQA solvers and end-to-end pipeline.
"""

import unittest
from pathlib import Path

from mmcultureqa.config import Config
from mmcultureqa.dataset import MMCultureQADataset
from mmcultureqa.models import LocalCulturalBaseline, MMCultureQAPipeline


class TestModels(unittest.TestCase):
    """Test suite for baseline solvers and pipeline execution."""

    def setUp(self):
        self.cfg = Config()
        self.solver = LocalCulturalBaseline()
        self.pipeline = MMCultureQAPipeline(solver=self.solver)

    def test_local_baseline_english_clothing(self):
        """Test local baseline answering English clothing question."""
        sample_img = list(self.cfg.images_dir.glob("*.jpg"))[0]
        q = "What can you infer about the cultural significance of the clothing worn by the seated individuals?"
        ans = self.solver.predict(sample_img, q, lang="en", country="Egypt")
        self.assertIn("clothing", ans.lower())
        self.assertIn("traditional", ans.lower())

    def test_local_baseline_arabic_clothing(self):
        """Test local baseline answering Arabic clothing question."""
        sample_img = list(self.cfg.images_dir.glob("*.jpg"))[0]
        q = "ماذا يمكنك أن تستنتج حول الأهمية الثقافية للملابس التي يرتديها الأفراد الجالسين في الصورة؟"
        ans = self.solver.predict(sample_img, q, lang="msa", country="Egypt")
        self.assertIn("الملابس", ans)
        self.assertIn("التقليدية", ans)

    def test_pipeline_task2_qa(self):
        """Test Task 2 Text QA pipeline."""
        sample_img = list(self.cfg.images_dir.glob("*.jpg"))[0]
        q = "What can you infer about the cultural significance of the clothing worn by the seated individuals?"
        res = self.pipeline.run_task2_qa(sample_img, q, lang="en", country="Egypt")
        self.assertEqual(res["task"], "qa")
        self.assertIn("prediction", res)
        self.assertTrue(len(res["prediction"]) > 0)

    def test_pipeline_task1_sqa(self):
        """Test Task 1 Spoken QA pipeline (Audio -> ASR -> Solver)."""
        sample_img = list(self.cfg.images_dir.glob("*.jpg"))[0]
        sample_aud = list((self.cfg.audio_dir / "en").glob("*.wav"))[0]
        res = self.pipeline.run_task1_sqa(sample_img, sample_aud, lang="en")
        self.assertEqual(res["task"], "sqa")
        self.assertIn("transcription", res)
        self.assertIn("prediction", res)
        self.assertTrue(len(res["prediction"]) > 0)

    def test_predict_dataset(self):
        """Test batch prediction on a sample dataset split."""
        dev_file = self.cfg.qa_dir / "dev_en.jsonl"
        ds = MMCultureQADataset.from_jsonl(dev_file, task="qa", lang="en")
        results = self.pipeline.predict_dataset(ds, limit=2)
        self.assertEqual(len(results), 2)
        self.assertIn("id", results[0])
        self.assertIn("prediction", results[0])


if __name__ == "__main__":
    unittest.main()
