"""
End-to-end inference pipeline for MMCultureQA.
Coordinates Task 1 (Spoken QA: Audio -> ASR -> Reasoner) and Task 2 (Text QA: Text -> Reasoner).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..asr import SpeechTranscriber
from ..dataset import MMCultureQADataset, MMCultureQARecord
from .base import BaseSolver
from .local_baseline import LocalCulturalBaseline

logger = logging.getLogger(__name__)


class MMCultureQAPipeline:
    """End-to-end pipeline for SemEval 2027 MMCultureQA."""

    def __init__(
        self,
        solver: Optional[BaseSolver] = None,
        transcriber: Optional[SpeechTranscriber] = None,
    ) -> None:
        self.solver = solver or LocalCulturalBaseline()
        self.transcriber = transcriber or SpeechTranscriber()

    def run_task1_sqa(
        self,
        image_path: Union[str, Path],
        audio_path: Union[str, Path],
        lang: str = "en",
        country: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Task 1: Spoken Visual Question Answering.

        Args:
            image_path: Path to image file.
            audio_path: Path to spoken question WAV file.
            lang: Language track identifier.
            country: Optional cultural country context.
            category: Optional cultural category context.

        Returns:
            Dictionary containing transcription, predicted answer, and metadata.
        """
        img_p = Path(image_path)
        aud_p = Path(audio_path)

        # Step 1: Transcribe audio question via ASR
        transcription = self.transcriber.transcribe(aud_p, lang=lang)

        # Step 2: Pass transcribed question + image to Multimodal Cultural Reasoner
        answer = self.solver.predict(
            image_path=img_p,
            question=transcription,
            lang=lang,
            country=country,
            category=category,
        )

        return {
            "task": "sqa",
            "image": str(img_p),
            "audio": str(aud_p),
            "transcription": transcription,
            "prediction": answer,
            "lang": lang,
        }

    def run_task2_qa(
        self,
        image_path: Union[str, Path],
        question: str,
        lang: str = "en",
        country: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Task 2: Textual Visual Question Answering.

        Args:
            image_path: Path to image file.
            question: Written question string.
            lang: Language track identifier.
            country: Optional cultural country context.
            category: Optional cultural category context.

        Returns:
            Dictionary containing question, predicted answer, and metadata.
        """
        img_p = Path(image_path)

        # Multimodal Cultural Reasoner generates answer directly from text question + image
        answer = self.solver.predict(
            image_path=img_p,
            question=question,
            lang=lang,
            country=country,
            category=category,
        )

        return {
            "task": "qa",
            "image": str(img_p),
            "question": question,
            "prediction": answer,
            "lang": lang,
        }

    def predict_record(self, record: MMCultureQARecord, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Process a single MMCultureQARecord instance."""
        img_path = record.resolve_image_path(base_dir)

        if record.task == "sqa" or (record.audio and not record.question):
            aud_path = record.resolve_audio_path(base_dir)
            if not aud_path or not aud_path.exists():
                raise FileNotFoundError(f"Audio file missing for record {record.id}: {record.audio}")
            res = self.run_task1_sqa(
                image_path=img_path,
                audio_path=aud_path,
                lang=record.lang,
                country=record.country,
                category=record.category,
            )
        else:
            q_text = record.question or ""
            res = self.run_task2_qa(
                image_path=img_path,
                question=q_text,
                lang=record.lang,
                country=record.country,
                category=record.category,
            )

        res["id"] = record.id
        if record.answer is not None:
            res["gold_reference"] = record.answer
        return res

    def predict_dataset(
        self,
        dataset: MMCultureQADataset,
        output_file: Optional[Union[str, Path]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run batch predictions over an entire dataset or split.

        Args:
            dataset: MMCultureQADataset instance.
            output_file: Optional path to save predictions JSON file.
            limit: Maximum number of records to process (useful for quick testing).

        Returns:
            List of prediction dictionaries.
        """
        records = dataset.records[:limit] if limit else dataset.records
        results: List[Dict[str, Any]] = []

        for record in records:
            pred_item = self.predict_record(record)
            results.append(pred_item)

        if output_file:
            out_p = Path(output_file)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"Saved {len(results)} predictions to {out_p}")

        return results
