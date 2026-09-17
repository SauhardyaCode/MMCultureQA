"""
MMCultureQA 2027 Prototype
==========================
SemEval 2027 Shared Task: Multilingual Multimodal Cultural Question Answering.

This package provides:
- Dataset loading and validation for OASIS and MMCultureQA formats.
- Multilingual speech recognition front-end (Task 1: Spoken QA).
- Culturally grounded multimodal reasoning pipelines (Task 2: Text QA).
- Official BERTScore F1 and auxiliary BLEU/ROUGE evaluation suite.
- CodaBench submission packaging and validation.
- Interactive Streamlit dashboard and comprehensive CLI.
"""

__version__ = "0.1.0"
__author__ = "SemEval 2027 MMCultureQA Participant"

from .config import Config
from .dataset import MMCultureQADataset, MMCultureQARecord

__all__ = [
    "__version__",
    "Config",
    "MMCultureQADataset",
    "MMCultureQARecord",
]
