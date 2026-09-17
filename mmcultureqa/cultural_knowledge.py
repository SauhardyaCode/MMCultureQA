"""
Cultural Knowledge Base and Grounding Retriever for MMCultureQA.
Provides external regional cultural knowledge (customs, attire, cuisine, landmarks,
and ceremonies) to augment visual reasoning models when visual content alone is insufficient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CulturalEntity:
    """Represents a cultural artifact, tradition, landmark, or custom."""

    name: str
    country_or_region: str
    category: str
    description: str
    keywords: List[str] = field(default_factory=list)
    arabic_name: Optional[str] = None
    aliases: List[str] = field(default_factory=list)


# Curated foundational knowledge base of cultural entities across the 18 tracks
DEFAULT_CULTURAL_KNOWLEDGE: List[CulturalEntity] = [
    CulturalEntity(
        name="National Museum of Qatar",
        country_or_region="Qatar",
        category="Geography, Buildings & Landmarks",
        description="Designed by architect Jean Nouvel, its dramatic interlocking discs are directly inspired by the mineral formation known as the desert rose (wardat al-sahra).",
        keywords=["museum", "qatar", "building", "rose", "desert rose", "discs", "architecture", "doha"],
        arabic_name="متحف قطر الوطني",
        aliases=["NMoQ", "Qatar National Museum"],
    ),
    CulturalEntity(
        name="Traditional MENA Clothing and Public Ceremonies",
        country_or_region="Egypt, UAE, Syria, Morocco",
        category="Objects, Materials & Clothing",
        description="Colorful, intricately embroidered traditional robes, vests, or thobes worn during public square storytelling (hakawati), folkloric dances (dabke, tanoura), or ceremonial festivals, reflecting regional heritage and craftsmanship.",
        keywords=["clothing", "seated", "garment", "cultural significance", "traditional", "ceremonial", "performance", "storytelling", "embroidered", "vest", "robes", "public square", "festival"],
        arabic_name="الملابس التقليدية والاحتفالات الشعبية",
        aliases=["Traditional dress", "Folkloric costume"],
    ),
    CulturalEntity(
        name="Desert Rose (Mineral Formation)",
        country_or_region="MENA",
        category="Geography, Buildings & Landmarks",
        description="A natural crystal formation of gypsum or baryte with sand inclusions forming rosette-like clusters found in arid regions, frequently inspiring modern regional architecture.",
        keywords=["desert rose", "crystals", "formation", "gypsum", "cluster", "architecture inspiration"],
        arabic_name="وردة الصحراء",
        aliases=["Sand rose", "Rose rock"],
    ),
    CulturalEntity(
        name="Dallah and Arabic Coffee Hospitality",
        country_or_region="Gulf / MENA",
        category="People, Society & Education",
        description="The Dallah is a traditional Arabic coffee pot used to brew and serve spiced gahwa (Arabic coffee with cardamom) to guests as a timeless symbol of generosity and hospitality.",
        keywords=["coffee", "pot", "dallah", "gahwa", "hospitality", "cup", "finjan", "dates", "welcome"],
        arabic_name="دلة قهوة عربية",
        aliases=["Dallah", "Gahwa"],
    ),
    CulturalEntity(
        name="Falconry (Bayzara)",
        country_or_region="UAE / Gulf",
        category="Sports & Recreation",
        description="An ancient UNESCO-recognized heritage practice where trained falcons were traditionally used for hunting in the desert and now represent prestige, sportsmanship, and desert culture.",
        keywords=["falcon", "falconry", "hawk", "bird of prey", "desert", "prestige", "traditional sport", "bayzara"],
        arabic_name="الصيد بالصقور / البيزرة",
        aliases=["Falconry", "Al-Bayzara"],
    ),
    CulturalEntity(
        name="Ramadan Fanous (Lantern)",
        country_or_region="Egypt / MENA",
        category="Religion & Spirituality",
        description="Intricate metal and colored glass lanterns traditionally illuminated during the holy month of Ramadan, originating in Fatimid Cairo and spreading across the Islamic world.",
        keywords=["lantern", "fanous", "ramadan", "fasting", "glass", "brass", "illumination", "celebration"],
        arabic_name="فانوس رمضان",
        aliases=["Fanous", "Fanoos"],
    ),
    CulturalEntity(
        name="Traditional Water Transportation (Felucca & Abra)",
        country_or_region="Egypt / UAE",
        category="Vehicles & Transportation",
        description="Traditional wooden sailing boats on the Nile (Felucca) and motorized wooden ferries on Dubai Creek (Abra) utilized for generations for commerce and communal transport.",
        keywords=["boat", "sail", "felucca", "abra", "nile", "creek", "wooden", "water transport", "river"],
        arabic_name="الفلوكة والعبرة",
        aliases=["Felucca", "Abra"],
    ),
    CulturalEntity(
        name="Henna Body Art",
        country_or_region="North Africa, South Asia, Middle East",
        category="Culture, Arts & Entertainment",
        description="Plant-based dye applied in intricate geometric and floral patterns on hands and feet during wedding nights (Laylat al-Henna), Eid celebrations, and auspicious cultural milestones.",
        keywords=["henna", "mehndi", "patterns", "hands", "wedding", "dye", "celebration", "bride", "festivity"],
        arabic_name="الحناء",
        aliases=["Mehndi", "Henna"],
    ),
]


class CulturalKnowledgeBase:
    """Knowledge repository for cultural facts and entities."""

    def __init__(self, entities: Optional[List[CulturalEntity]] = None) -> None:
        self.entities: List[CulturalEntity] = entities or list(DEFAULT_CULTURAL_KNOWLEDGE)

    def add_entity(self, entity: CulturalEntity) -> None:
        """Register a new cultural entity into the knowledge base."""
        self.entities.append(entity)

    def all_entities(self) -> List[CulturalEntity]:
        """Return all registered cultural entities."""
        return list(self.entities)


class CulturalRetriever:
    """Retrieves relevant cultural context to ground multimodal QA."""

    def __init__(self, kb: Optional[CulturalKnowledgeBase] = None) -> None:
        self.kb = kb or CulturalKnowledgeBase()

    def _tokenize(self, text: str) -> Set[str]:
        """Simple multilingual tokenization for lexical matching."""
        tokens = re.findall(r"\w+", text.lower())
        # Filter very short tokens
        return {t for t in tokens if len(t) > 2}

    def retrieve(
        self,
        query: str,
        country: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Tuple[CulturalEntity, float]]:
        """Find the most culturally relevant knowledge entries for a query.

        Args:
            query: The question text or transcribed question.
            country: Optional known country context.
            category: Optional known category context.
            top_k: Number of entries to retrieve.

        Returns:
            List of (CulturalEntity, score) tuples.
        """
        query_tokens = self._tokenize(query)
        scored_entries: List[Tuple[CulturalEntity, float]] = []

        for entity in self.kb.all_entities():
            score = 0.0

            # Match against entity keywords
            for kw in entity.keywords:
                kw_tokens = self._tokenize(kw)
                overlap = len(query_tokens.intersection(kw_tokens))
                if overlap > 0:
                    score += overlap * 2.0
                elif kw.lower() in query.lower():
                    score += 3.0

            # Match against entity name and aliases
            for name in [entity.name] + entity.aliases:
                if name.lower() in query.lower():
                    score += 5.0

            # Bonus for country match
            if country and country.lower() in entity.country_or_region.lower():
                score += 1.5

            # Bonus for category match
            if category and (category.lower() in entity.category.lower() or entity.category.lower() in category.lower()):
                score += 1.0

            if score > 0.0:
                scored_entries.append((entity, score))

        # Sort descending by relevance score
        scored_entries.sort(key=lambda x: x[1], reverse=True)

        if not scored_entries:
            # If no direct match, provide general cultural entities
            return [(e, 0.5) for e in self.kb.all_entities()[:top_k]]

        return scored_entries[:top_k]

    def format_cultural_context(self, retrieved: List[Tuple[CulturalEntity, float]]) -> str:
        """Format retrieved entities into an informative context prompt snippet."""
        if not retrieved:
            return ""

        lines = ["Relevant Cultural Context:"]
        for entity, score in retrieved:
            ar_part = f" ({entity.arabic_name})" if entity.arabic_name else ""
            lines.append(f"- **{entity.name}{ar_part}** [{entity.country_or_region}]: {entity.description}")

        return "\n".join(lines)
