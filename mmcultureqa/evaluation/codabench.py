"""
CodaBench submission packaging and validation module for SemEval 2027 MMCultureQA.
Formats model outputs and creates verified ZIP submissions ready for competition upload.
"""

from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class CodaBenchPackager:
    """Validates predictions and prepares CodaBench submission archives."""

    @staticmethod
    def validate_predictions(
        predictions: List[Dict[str, Any]],
        expected_ids: Optional[Union[Set[str], List[str]]] = None,
    ) -> Tuple[bool, List[str]]:
        """Verify prediction schema, data completeness, and non-empty answers.

        Returns:
            Tuple of (is_valid: bool, issues: List[str])
        """
        issues: List[str] = []

        if not predictions:
            issues.append("Predictions list is empty.")
            return False, issues

        seen_ids: Set[str] = set()
        for idx, item in enumerate(predictions):
            if not isinstance(item, dict):
                issues.append(f"Item at index {idx} is not a dictionary.")
                continue

            item_id = item.get("id")
            if not item_id or not isinstance(item_id, str):
                issues.append(f"Item at index {idx} missing valid 'id' string.")
            elif item_id in seen_ids:
                issues.append(f"Duplicate id '{item_id}' detected at index {idx}.")
            else:
                seen_ids.add(item_id)

            answer = item.get("answer") or item.get("prediction")
            if answer is None or not str(answer).strip():
                issues.append(f"Item with id '{item_id}' has empty answer/prediction.")

        if expected_ids:
            exp_set = set(expected_ids)
            missing = exp_set - seen_ids
            if missing:
                issues.append(f"Missing {len(missing)} expected IDs in predictions (e.g. {list(missing)[:3]}).")

        is_valid = len(issues) == 0
        return is_valid, issues

    @classmethod
    def create_submission_zip(
        cls,
        predictions: Union[List[Dict[str, Any]], str, Path],
        output_zip_path: Union[str, Path],
        expected_ids: Optional[Union[Set[str], List[str]]] = None,
    ) -> Dict[str, Any]:
        """Create a standard CodaBench submission.zip archive containing predictions.json.

        Args:
            predictions: List of prediction dicts or path to JSON file.
            output_zip_path: Target path for the output .zip file.
            expected_ids: Optional set of IDs required by the competition split.

        Returns:
            Dictionary with packaging metadata (size, item count, status).
        """
        # Load from file if string or Path
        if isinstance(predictions, (str, Path)):
            p_file = Path(predictions)
            pred_list = json.loads(p_file.read_text(encoding="utf-8"))
        else:
            pred_list = list(predictions)

        # Validate
        is_valid, issues = cls.validate_predictions(pred_list, expected_ids=expected_ids)
        if not is_valid:
            error_msg = f"Predictions failed validation: {'; '.join(issues[:5])}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Standardize format: {"id": str, "answer": str}
        cleaned_preds = []
        for p in pred_list:
            cleaned_preds.append({
                "id": str(p["id"]),
                "answer": str(p.get("answer") or p.get("prediction", "")).strip(),
            })

        out_zip = Path(output_zip_path)
        out_zip.parent.mkdir(parents=True, exist_ok=True)

        # Write predictions.json directly into zip
        predictions_json_content = json.dumps(cleaned_preds, ensure_ascii=False, indent=2)

        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("predictions.json", predictions_json_content)

        zip_size_bytes = out_zip.stat().st_size

        return {
            "status": "success",
            "submission_zip": str(out_zip),
            "records_count": len(cleaned_preds),
            "size_bytes": zip_size_bytes,
            "contains_file": "predictions.json",
        }
