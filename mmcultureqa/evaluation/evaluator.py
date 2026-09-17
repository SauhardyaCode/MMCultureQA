"""
Comprehensive evaluation harness for SemEval 2027 MMCultureQA.
Aligns system predictions with ground-truth references, computes aggregate
and per-category/per-country breakdowns, and exports formatted reports.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tabulate import tabulate

from ..dataset import MMCultureQADataset
from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class MMCultureQAEvaluator:
    """Evaluates MMCultureQA predictions against reference splits."""

    def __init__(
        self,
        predictions: Union[List[Dict[str, Any]], str, Path],
        references: Union[MMCultureQADataset, str, Path],
        lang: str = "en",
    ) -> None:
        # Load predictions
        if isinstance(predictions, (str, Path)):
            pred_p = Path(predictions)
            self.preds = json.loads(pred_p.read_text(encoding="utf-8"))
        else:
            self.preds = list(predictions)

        # Load references
        if isinstance(references, (str, Path)):
            ref_p = Path(references)
            self.dataset = MMCultureQADataset.from_jsonl(ref_p, lang=lang)
        else:
            self.dataset = references

        self.lang = lang
        self.ref_map = {r.id: r for r in self.dataset.records}

    def evaluate(self) -> Dict[str, Any]:
        """Compute full evaluation report including aggregate and granular breakdowns."""
        matched_preds: List[str] = []
        matched_refs: List[str] = []
        unmatched_ids: List[str] = []

        by_country = defaultdict(lambda: {"preds": [], "refs": []})
        by_category = defaultdict(lambda: {"preds": [], "refs": []})

        for p in self.preds:
            p_id = p.get("id")
            # Support both 'prediction' and 'answer' keys
            pred_text = p.get("prediction") or p.get("answer") or ""

            if p_id in self.ref_map:
                ref_rec = self.ref_map[p_id]
                ref_text = ref_rec.answer or ""

                matched_preds.append(pred_text)
                matched_refs.append(ref_text)

                country = ref_rec.country or "Unknown"
                category = ref_rec.category or "Unknown"

                by_country[country]["preds"].append(pred_text)
                by_country[country]["refs"].append(ref_text)

                by_category[category]["preds"].append(pred_text)
                by_category[category]["refs"].append(ref_text)
            else:
                unmatched_ids.append(p_id)

        if not matched_preds:
            raise ValueError("No matching prediction IDs found in reference dataset!")

        # Aggregate overall metrics
        overall_metrics = compute_all_metrics(matched_preds, matched_refs, lang=self.lang)

        # Breakdown by country
        country_metrics: Dict[str, Dict[str, float]] = {}
        for c, data in by_country.items():
            if data["preds"]:
                c_metrics = compute_all_metrics(data["preds"], data["refs"], lang=self.lang)
                c_metrics["count"] = len(data["preds"])
                country_metrics[c] = c_metrics

        # Breakdown by category
        category_metrics: Dict[str, Dict[str, float]] = {}
        for cat, data in by_category.items():
            if data["preds"]:
                cat_metrics = compute_all_metrics(data["preds"], data["refs"], lang=self.lang)
                cat_metrics["count"] = len(data["preds"])
                category_metrics[cat] = cat_metrics

        return {
            "total_evaluated": len(matched_preds),
            "unmatched_predictions_count": len(unmatched_ids),
            "language": self.lang,
            "overall": overall_metrics,
            "by_country": country_metrics,
            "by_category": category_metrics,
        }

    def format_summary_table(self, results: Dict[str, Any]) -> str:
        """Format evaluation results into a human-readable ASCII table."""
        overall = results["overall"]
        headers = ["Metric", "Score", "Description"]
        table_rows = [
            ["BERTScore F1 (Official)", f"{overall['bert_score_f1']:.4f}", "Semantic similarity (Official Ranking)"],
            ["BERTScore Precision", f"{overall['bert_score_precision']:.4f}", "Semantic precision"],
            ["BERTScore Recall", f"{overall['bert_score_recall']:.4f}", "Semantic recall"],
            ["BLEU-4", f"{overall['bleu_4']:.2f}", "Auxiliary 4-gram lexical overlap"],
            ["BLEU-1", f"{overall['bleu_1']:.2f}", "Auxiliary unigram overlap"],
            ["ROUGE-L", f"{overall['rouge_l']:.2f}", "Auxiliary longest common subsequence"],
            ["ROUGE-1", f"{overall['rouge_1']:.2f}", "Auxiliary unigram recall"],
            ["Exact Match (%)", f"{overall['exact_match']:.2f}%", "Exact text equality"],
            ["Token F1 (%)", f"{overall['token_f1']:.2f}%", "Token overlap F1"],
        ]
        summary_str = tabulate(table_rows, headers=headers, tablefmt="github")

        # Category breakdown table
        cat_rows = []
        for cat, scores in results["by_category"].items():
            cat_rows.append([
                cat,
                scores["count"],
                f"{scores['bert_score_f1']:.4f}",
                f"{scores['bleu_4']:.2f}",
                f"{scores['rouge_l']:.2f}",
            ])
        cat_table = tabulate(
            cat_rows,
            headers=["Cultural Category", "Count", "BERTScore F1", "BLEU-4", "ROUGE-L"],
            tablefmt="github",
        )

        return f"### MMCultureQA Evaluation Summary ({results['language'].upper()})\n\n{summary_str}\n\n### Category Breakdown\n\n{cat_table}"

    def export_report(self, results: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Save structured JSON evaluation report."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Exported evaluation report to {p}")
