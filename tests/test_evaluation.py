"""
Unit tests for evaluation metrics and MMCultureQAEvaluator harness.
"""

import unittest
from pathlib import Path

from mmcultureqa.config import Config
from mmcultureqa.dataset import MMCultureQADataset
from mmcultureqa.evaluation import MMCultureQAEvaluator, compute_all_metrics, compute_bert_score, compute_bleu, compute_rouge


class TestEvaluation(unittest.TestCase):
    """Test suite for metrics and evaluation report generation."""

    def test_exact_match_metric(self):
        """Test metrics on identical strings."""
        preds = ["The national museum of qatar is inspired by desert rose."]
        refs = ["The National Museum of Qatar is inspired by desert rose."]
        metrics = compute_all_metrics(preds, refs, lang="en")
        self.assertEqual(metrics["exact_match"], 100.0)
        self.assertAlmostEqual(metrics["bert_score_f1"], 1.0, places=2)

    def test_bleu_and_rouge(self):
        """Test auxiliary BLEU and ROUGE scoring."""
        preds = ["traditional clothing worn by dancers in festival"]
        refs = ["traditional clothing worn by participants in public festival"]
        bleu = compute_bleu(preds, refs)
        rouge = compute_rouge(preds, refs)
        self.assertGreater(bleu["bleu_1"], 0.0)
        self.assertGreater(rouge["rouge_1"], 0.0)
        self.assertGreater(rouge["rouge_l"], 0.0)

    def test_evaluator_harness(self):
        """Test MMCultureQAEvaluator report with category breakdown."""
        cfg = Config()
        dev_path = cfg.qa_dir / "dev_en.jsonl"
        ds = MMCultureQADataset.from_jsonl(dev_path, lang="en")

        # Fake predictions matching dataset IDs
        mock_preds = [{"id": r.id, "prediction": r.answer or "sample answer"} for r in ds.records]
        evaluator = MMCultureQAEvaluator(mock_preds, ds, lang="en")
        results = evaluator.evaluate()

        self.assertEqual(results["total_evaluated"], len(ds.records))
        self.assertIn("overall", results)
        self.assertIn("bert_score_f1", results["overall"])
        self.assertIn("by_category", results)

        summary_table = evaluator.format_summary_table(results)
        self.assertIn("BERTScore F1", summary_table)
        self.assertIn("Category Breakdown", summary_table)


if __name__ == "__main__":
    unittest.main()
