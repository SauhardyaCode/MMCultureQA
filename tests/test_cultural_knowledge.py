"""
Unit tests for Cultural Knowledge Base and Retriever.
"""

import unittest

from mmcultureqa.cultural_knowledge import CulturalEntity, CulturalKnowledgeBase, CulturalRetriever


class TestCulturalKnowledge(unittest.TestCase):
    """Test suite for cultural knowledge base and retriever."""

    def setUp(self):
        self.kb = CulturalKnowledgeBase()
        self.retriever = CulturalRetriever(self.kb)

    def test_retrieve_clothing_query(self):
        """Test retrieving traditional clothing context from a clothing question."""
        query = "What can you infer about the cultural significance of the clothing worn by the seated individuals?"
        results = self.retriever.retrieve(query, country="Egypt", top_k=2)
        self.assertGreater(len(results), 0)
        top_entity, score = results[0]
        self.assertTrue("Clothing" in top_entity.name or "Ceremonies" in top_entity.name)
        self.assertGreater(score, 1.0)

    def test_retrieve_landmark_query(self):
        """Test retrieving Qatar Museum info from architecture query."""
        query = "What is the name of the building and what inspired its desert rose design?"
        results = self.retriever.retrieve(query, country="Qatar", top_k=2)
        self.assertGreater(len(results), 0)
        entity_names = [e.name for e, _ in results]
        self.assertTrue(any("Qatar" in name or "Desert Rose" in name for name in entity_names))

    def test_format_cultural_context(self):
        """Test formatting of retrieved context block for prompts."""
        query = "Tell me about the dallah and coffee hospitality"
        results = self.retriever.retrieve(query, top_k=1)
        formatted = self.retriever.format_cultural_context(results)
        self.assertIn("Relevant Cultural Context", formatted)
        self.assertIn("Dallah", formatted)


if __name__ == "__main__":
    unittest.main()
