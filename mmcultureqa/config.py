"""
Configuration module for MMCultureQA.
Handles paths, language definitions, model configurations, and environment defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Base repository root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Standard Hugging Face dataset identifier for SemEval 2027
HF_DATASET_ID = "QCRI/MMCQA-SemEval27"

# 18 planned language tracks defined by the SemEval 2027 task specification
SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "en": {"name": "English", "native": "English", "script": "latin", "asr_code": "en-US"},
    "msa": {"name": "Modern Standard Arabic", "native": "العربية الفصحى", "script": "arabic", "asr_code": "ar-SA"},
    "arz": {"name": "Egyptian Arabic", "native": "العربية المصرية", "script": "arabic", "asr_code": "ar-EG"},
    "ajp": {"name": "Levantine Arabic", "native": "العربية الشامية", "script": "arabic", "asr_code": "ar-JO"},
    "bn": {"name": "Bangla", "native": "বাংলা", "script": "bengali", "asr_code": "bn-BD"},
    "ur": {"name": "Urdu", "native": "اردو", "script": "urdu", "asr_code": "ur-PK"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "script": "devanagari", "asr_code": "hi-IN"},
    "as": {"name": "Assamese", "native": "অসমীয়া", "script": "bengali", "asr_code": "as-IN"},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી", "script": "gujarati", "asr_code": "gu-IN"},
    "mr": {"name": "Marathi", "native": "मराठी", "script": "devanagari", "asr_code": "mr-IN"},
    "it": {"name": "Italian", "native": "Italiano", "script": "latin", "asr_code": "it-IT"},
    "es": {"name": "Spanish", "native": "Español", "script": "latin", "asr_code": "es-ES"},
    "pt": {"name": "Portuguese", "native": "Português", "script": "latin", "asr_code": "pt-PT"},
    "tr": {"name": "Turkish", "native": "Türkçe", "script": "latin", "asr_code": "tr-TR"},
    "am": {"name": "Amharic", "native": "አማርኛ", "script": "ethiopic", "asr_code": "am-ET"},
    "om": {"name": "Oromo", "native": "Afaan Oromoo", "script": "latin", "asr_code": "om-ET"},
    "so": {"name": "Somali", "native": "Soomaali", "script": "latin", "asr_code": "so-SO"},
    "ti": {"name": "Tigrinya", "native": "ትግርኛ", "script": "ethiopic", "asr_code": "ti-ET"},
}


@dataclass
class Config:
    """Global configuration parameters for MMCultureQA pipelines."""
    
    # Filesystem directories
    root_dir: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    images_dir: Path = PROJECT_ROOT / "data" / "images"
    audio_dir: Path = PROJECT_ROOT / "data" / "audio"
    qa_dir: Path = PROJECT_ROOT / "data" / "qa"
    sqa_dir: Path = PROJECT_ROOT / "data" / "sqa"
    submissions_dir: Path = PROJECT_ROOT / "submissions"
    
    # Model and inference settings
    openai_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    default_vlm_model: str = "gpt-4o-mini"
    max_tokens: int = 150
    temperature: float = 0.2
    
    # Evaluation settings
    primary_metric: str = "bert_score_f1"
    auxiliary_metrics: List[str] = field(default_factory=lambda: ["bleu_1", "bleu_4", "rouge_1", "rouge_2", "rouge_l"])
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they do not already exist."""
        for path in [
            self.data_dir,
            self.images_dir,
            self.audio_dir,
            self.qa_dir,
            self.sqa_dir,
            self.submissions_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
            
    def get_asr_language_code(self, lang: str) -> str:
        """Resolve a task language identifier to an ASR locale code."""
        lang_info = SUPPORTED_LANGUAGES.get(lang.lower())
        if lang_info and "asr_code" in lang_info:
            return lang_info["asr_code"]
        # Fallback to language code directly
        return lang
