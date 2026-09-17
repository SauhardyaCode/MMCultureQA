"""
Unit tests for CodaBench submission packaging and schema validation.
"""

import tempfile
import unittest
import zipfile
from pathlib import Path

from mmcultureqa.evaluation import CodaBenchPackager


class TestCodaBench(unittest.TestCase):
    """Test suite for CodaBench submission formatting and validation."""

    def test_validation_valid_input(self):
        """Test validator accepting valid predictions list."""
        preds = [
            {"id": "id1", "answer": "Answer 1"},
            {"id": "id2", "answer": "Answer 2"},
        ]
        is_valid, issues = CodaBenchPackager.validate_predictions(preds, expected_ids=["id1", "id2"])
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

    def test_validation_missing_id_and_duplicate(self):
        """Test validator catching duplicate IDs and empty answers."""
        bad_preds = [
            {"id": "id1", "answer": ""},
            {"id": "id1", "answer": "Duplicate"},
        ]
        is_valid, issues = CodaBenchPackager.validate_predictions(bad_preds)
        self.assertFalse(is_valid)
        self.assertTrue(any("Duplicate" in msg for msg in issues))
        self.assertTrue(any("empty" in msg for msg in issues))

    def test_create_submission_zip(self):
        """Test packaging predictions into a verified zip file."""
        preds = [
            {"id": "hash1", "prediction": "Prediction 1"},
            {"id": "hash2", "answer": "Answer 2"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "submission.zip"
            meta = CodaBenchPackager.create_submission_zip(preds, zip_path)

            self.assertTrue(zip_path.exists())
            self.assertEqual(meta["records_count"], 2)

            with zipfile.ZipFile(zip_path, "r") as zf:
                self.assertIn("predictions.json", zf.namelist())


if __name__ == "__main__":
    unittest.main()
