"""
Vision-Language Model (VLM) solver with Cultural RAG grounding.
Uses OpenAI-compatible multimodal APIs (e.g., GPT-4o, GPT-4o-mini) to perform
culturally grounded visual question answering.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config import Config
from ..cultural_knowledge import CulturalKnowledgeBase, CulturalRetriever
from .base import BaseSolver

logger = logging.getLogger(__name__)


class CulturalVLMSolver(BaseSolver):
    """Multimodal Vision-Language Model solver with cultural context injection."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        kb: Optional[CulturalKnowledgeBase] = None,
    ) -> None:
        cfg = Config()
        self.model_name = model_name or cfg.default_vlm_model
        self.api_key = api_key or cfg.openai_api_key
        self.base_url = base_url
        self.kb = kb or CulturalKnowledgeBase()
        self.retriever = CulturalRetriever(self.kb)
        self._client = None

    def _get_client(self):
        """Lazy load the OpenAI API client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required to run CulturalVLMSolver. "
                    "Please set the OPENAI_API_KEY environment variable, or use LocalCulturalBaseline."
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _encode_image(self, image_path: Path) -> str:
        """Read and encode an image file into base64 string."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def predict(
        self,
        image_path: Union[str, Path],
        question: str,
        lang: str = "en",
        country: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        """Perform multimodal cultural VQA using Vision-Language Model."""
        img_p = Path(image_path)
        if not img_p.exists():
            raise FileNotFoundError(f"Image not found at {img_p}")

        client = self._get_client()
        base64_image = self._encode_image(img_p)

        # Retrieve cultural context
        retrieved = self.retriever.retrieve(question, country=country, category=category, top_k=2)
        cultural_context_str = self.retriever.format_cultural_context(retrieved)

        # System prompt tailored to SemEval 2027 MMCultureQA guidelines
        system_prompt = (
            "You are an expert AI participant in the SemEval 2027 MMCultureQA shared task. "
            "Your task is to provide a concise, direct, culturally accurate open-ended answer (1-2 sentences) "
            "to a question about an image. Many questions require regional cultural knowledge beyond plain "
            "visual recognition. Ground your answer in cultural facts and what is visible in the image. "
            "Do not include conversational pleasantries, introductory phrases, or filler."
        )

        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Question ({lang}): {question}\n\n"
                    f"{cultural_context_str}\n\n"
                    f"Provide a short, culturally grounded answer in {lang}."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "auto",
                },
            },
        ]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=Config().max_tokens,
                temperature=Config().temperature,
            )
            answer = response.choices[0].message.content or ""
            return answer.strip()
        except Exception as e:
            logger.error(f"VLM API inference error: {e}")
            raise
