"""
Base solver abstraction for MMCultureQA models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..dataset import MMCultureQARecord


class BaseSolver(ABC):
    """Abstract base class for all MMCultureQA prediction models."""

    @abstractmethod
    def predict(
        self,
        image_path: Union[str, Path],
        question: str,
        lang: str = "en",
        country: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        """Generate an open-ended answer for an image and question pair.

        Args:
            image_path: Path to image file.
            question: Text of the question (or transcribed audio).
            lang: Language code ('en', 'msa', etc.).
            country: Optional cultural country context.
            category: Optional cultural category context.

        Returns:
            Short, culturally grounded open-ended answer string.
        """
        raise NotImplementedError("Subclasses must implement predict()")

    def predict_record(self, record: MMCultureQARecord, base_dir: Optional[Path] = None) -> str:
        """Generate an answer for a dataset record."""
        image_path = record.resolve_image_path(base_dir)
        question = record.question or ""
        return self.predict(
            image_path=image_path,
            question=question,
            lang=record.lang,
            country=record.country,
            category=record.category,
        )
