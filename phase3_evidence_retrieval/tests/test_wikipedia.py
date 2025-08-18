"""
Tests for Wikipedia fallback (background grounding).

- Fetch summary + relevant sections for claim entities
- Keep page revision id for reproducibility
- Store as low-priority background evidence
"""

import os
import sys
import re
import pytest

# Ensure package import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from connectors.wikipedia import WikipediaConnector  # type: ignore


ENTITIES = [
	"World Health Organization",
	"Coronavirus disease 2019",
	"Renewable energy",
	"Inflation (economics)",
	"Narendra Modi",
]


def _key_sentences(text: str, k: int = 3):
	# Simple split into sentences
	sentences = re.split(r"(?<=[.!?])\s+", text or "")
	return [s.strip() for s in sentences if s.strip()][:k]


@pytest.mark.asyncio
async def test_wikipedia_background_print(capsys):
	connector = WikipediaConnector(lang="en")
	items = connector.fetch_background_for_entities(ENTITIES)
	assert len(items) >= 3
	
	for ev in items:
		rev = ev.metadata.custom_fields.get("wikipedia_revid") if ev.metadata and ev.metadata.custom_fields else None
		title = ev.title
		print(f"Title: {title}")
		print(f"Revision: {rev}")
		print(f"URL: {ev.url}")
		sents = _key_sentences(ev.full_text or ev.snippet or "")
		print("Key sentences:")
		for s in sents:
			print(f" - {s}")
		print()
		assert rev is not None
