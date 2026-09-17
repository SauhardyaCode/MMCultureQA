"""
Dataset management module for MMCultureQA (SemEval 2027).
Handles loading, schema verification, and media path resolution for both
Task 1 (Spoken Visual QA - SQA) and Task 2 (Textual Visual QA - QA).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .config import Config, PROJECT_ROOT


@dataclass
class MMCultureQARecord:
    """Represents a single instance in the MMCultureQA shared task."""

    id: str
    image: str
    question: Optional[str] = None
    audio: Optional[str] = None
    answer: Optional[str] = None
    country: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    lang: str = "en"
    task: str = "qa"  # 'qa' (Task 2: Text) or 'sqa' (Task 1: Spoken)

    def resolve_image_path(self, base_dir: Optional[Union[str, Path]] = None) -> Path:
        """Resolve full absolute path to the record's image file.

        Args:
            base_dir: Root directory containing data folder or image path.
                      Defaults to project data directory.

        Returns:
            Path object pointing to the image file.
        """
        base = Path(base_dir) if base_dir else (PROJECT_ROOT / "data")
        # Handle cases where self.image is 'images/<id>.jpg'
        candidate = base / self.image
        if candidate.exists():
            return candidate
        # Fallback if image path was relative to project root
        candidate_root = PROJECT_ROOT / self.image
        if candidate_root.exists():
            return candidate_root
        return candidate

    def resolve_audio_path(self, base_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Resolve full absolute path to the record's audio WAV file (if present).

        Args:
            base_dir: Root directory containing data folder or audio path.

        Returns:
            Path object pointing to the audio file, or None if no audio exists.
        """
        if not self.audio:
            return None
        base = Path(base_dir) if base_dir else (PROJECT_ROOT / "data")
        candidate = base / self.audio
        if candidate.exists():
            return candidate
        candidate_root = PROJECT_ROOT / self.audio
        if candidate_root.exists():
            return candidate_root
        return candidate

    def validate_media(self, base_dir: Optional[Union[str, Path]] = None) -> Tuple[bool, str]:
        """Check whether the referenced image and (optional) audio files exist on disk.

        Returns:
            Tuple of (is_valid: bool, status_message: str)
        """
        img_path = self.resolve_image_path(base_dir)
        if not img_path.exists():
            return False, f"Image not found at {img_path}"
        if img_path.stat().st_size == 0:
            return False, f"Image file is empty at {img_path}"

        if self.task == "sqa" or self.audio:
            aud_path = self.resolve_audio_path(base_dir)
            if not aud_path or not aud_path.exists():
                return False, f"Audio file not found at {aud_path}"
            if aud_path.stat().st_size == 0:
                return False, f"Audio file is empty at {aud_path}"

        return True, "Media files verified successfully"

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a dictionary matching SemEval JSONL format."""
        d = {
            "id": self.id,
            "image": self.image,
        }
        if self.question is not None:
            d["question"] = self.question
        if self.audio is not None:
            d["audio"] = self.audio
        if self.answer is not None:
            d["answer"] = self.answer
        if self.country is not None:
            d["country"] = self.country
        if self.category is not None:
            d["category"] = self.category
        if self.subcategory is not None:
            d["subcategory"] = self.subcategory
        return d


class MMCultureQADataset:
    """Container for managing collections of MMCultureQA records."""

    def __init__(
        self,
        records: Optional[List[MMCultureQARecord]] = None,
        task: str = "qa",
        lang: str = "en",
        source_file: Optional[Path] = None,
    ) -> None:
        self.records: List[MMCultureQARecord] = records or []
        self.task: str = task
        self.lang: str = lang
        self.source_file: Optional[Path] = source_file

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[MMCultureQARecord]:
        return iter(self.records)

    def __getitem__(self, index: int) -> MMCultureQARecord:
        return self.records[index]

    @classmethod
    def from_jsonl(
        cls,
        filepath: Union[str, Path],
        task: str = "qa",
        lang: str = "en",
    ) -> "MMCultureQADataset":
        """Load an official MMCultureQA JSONL split file.

        Args:
            filepath: Path to the .jsonl file (e.g., data/qa/train_en.jsonl).
            task: 'qa' (text) or 'sqa' (spoken).
            lang: Language code ('en', 'msa', etc.).

        Returns:
            Initialized MMCultureQADataset instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")

        records: List[MMCultureQARecord] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = MMCultureQARecord(
                        id=data.get("id", ""),
                        image=data.get("image", ""),
                        question=data.get("question"),
                        audio=data.get("audio"),
                        answer=data.get("answer"),
                        country=data.get("country"),
                        category=data.get("category"),
                        subcategory=data.get("subcategory"),
                        lang=lang,
                        task=task,
                    )
                    records.append(record)
                except json.JSONDecodeError as err:
                    raise ValueError(f"Malformed JSON on line {line_idx} of {path}: {err}") from err

        return cls(records=records, task=task, lang=lang, source_file=path)

    @classmethod
    def load_split(
        cls,
        split: str = "train",
        task: str = "qa",
        lang: str = "en",
        data_dir: Optional[Union[str, Path]] = None,
    ) -> "MMCultureQADataset":
        """Convenience loader for standard data directory structures.

        Example:
            ds = MMCultureQADataset.load_split("dev", task="qa", lang="en")
        """
        cfg = Config()
        base = Path(data_dir) if data_dir else cfg.data_dir
        folder = base / task
        filename = f"{split}_{lang}.jsonl"
        filepath = folder / filename

        if not filepath.exists():
            # Try finding without language suffix or alternate naming
            alternates = list(folder.glob(f"*{split}*{lang}*.jsonl"))
            if alternates:
                filepath = alternates[0]
            else:
                raise FileNotFoundError(
                    f"Could not find split '{split}' for task='{task}' and lang='{lang}' at {filepath}"
                )

        return cls.from_jsonl(filepath, task=task, lang=lang)

    def filter_by_country(self, country: str) -> "MMCultureQADataset":
        """Return a subset of records matching the given country."""
        filtered = [r for r in self.records if (r.country or "").lower() == country.lower()]
        return MMCultureQADataset(records=filtered, task=self.task, lang=self.lang, source_file=self.source_file)

    def filter_by_category(self, category: str) -> "MMCultureQADataset":
        """Return a subset of records matching the given cultural category."""
        filtered = [r for r in self.records if (r.category or "").lower() == category.lower()]
        return MMCultureQADataset(records=filtered, task=self.task, lang=self.lang, source_file=self.source_file)

    def validate_integrity(self, base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Verify that all records have valid media files and required fields."""
        missing_images = []
        missing_audio = []
        missing_questions = []
        missing_answers = []

        for r in self.records:
            is_valid, msg = r.validate_media(base_dir)
            if not is_valid:
                if "Image" in msg:
                    missing_images.append((r.id, msg))
                if "Audio" in msg:
                    missing_audio.append((r.id, msg))
            if self.task == "qa" and not r.question:
                missing_questions.append(r.id)
            if not r.answer:
                missing_answers.append(r.id)

        all_ok = len(missing_images) == 0 and len(missing_audio) == 0
        return {
            "total_records": len(self.records),
            "valid": all_ok,
            "missing_images_count": len(missing_images),
            "missing_audio_count": len(missing_audio),
            "missing_questions_count": len(missing_questions),
            "missing_answers_count": len(missing_answers),
            "missing_images": missing_images,
            "missing_audio": missing_audio,
        }

    def summary_statistics(self) -> Dict[str, Any]:
        """Compute aggregate distributions over country, category, and subcategory."""
        countries: Dict[str, int] = {}
        categories: Dict[str, int] = {}
        subcategories: Dict[str, int] = {}

        for r in self.records:
            c = r.country or "Unknown"
            cat = r.category or "Unknown"
            subcat = r.subcategory or "Unknown"

            countries[c] = countries.get(c, 0) + 1
            categories[cat] = categories.get(cat, 0) + 1
            subcategories[subcat] = subcategories.get(subcat, 0) + 1

        return {
            "total": len(self.records),
            "task": self.task,
            "lang": self.lang,
            "countries": sorted(countries.items(), key=lambda x: x[1], reverse=True),
            "categories": sorted(categories.items(), key=lambda x: x[1], reverse=True),
            "subcategories": sorted(subcategories.items(), key=lambda x: x[1], reverse=True),
        }
