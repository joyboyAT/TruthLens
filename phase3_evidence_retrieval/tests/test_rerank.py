"""
Tests for first-stage retrieval + re-ranking.

- Build synthetic chunks; vector search for initial ranking
- Rerank with BAAI/bge-reranker-v2-m3 and ensure on-topic chunk rises
"""

import os
import sys
import pytest

# Ensure package import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from schemas.evidence import TextChunk, Language
from retrieval.vector_search import VectorSearcher
from retrieval.rerank import Reranker


@pytest.mark.slow
def test_reranker_lifts_on_topic():
	claim = "COVID-19 vaccines reduce severe illness"
	chunks = [
		TextChunk(chunk_id="c0", text="A recipe for chocolate cake.", chunk_index=0, start_char=0, end_char=10, embedding=None, embedding_model=None, language=Language.ENGLISH),
		TextChunk(chunk_id="c1", text="Football match results from last night.", chunk_index=1, start_char=0, end_char=10, embedding=None, embedding_model=None, language=Language.ENGLISH),
		TextChunk(chunk_id="c2", text="Clinical trials show COVID-19 vaccines reduce hospitalization and severe disease.", chunk_index=2, start_char=0, end_char=10, embedding=None, embedding_model=None, language=Language.ENGLISH),
		TextChunk(chunk_id="c3", text="Stock market news and analysis.", chunk_index=3, start_char=0, end_char=10, embedding=None, embedding_model=None, language=Language.ENGLISH),
	]

	# First-stage
	vs = VectorSearcher()
	initial = vs.search(claim, chunks, top_k=4)
	initial_order = [chunks[i] for i, _ in initial]
	print("Initial order (vector search):")
	for c in initial_order:
		print(" -", c.text[:60])

	# Rerank
	rr = Reranker()
	ranked = rr.rerank(claim, initial_order)
	print("\nAfter reranking:")
	ranked_texts = []
	for c, s in ranked:
		print(f" - {s:.3f} | {c.text[:60]}")
		ranked_texts.append(c.text)

	# Ensure the on-topic chunk is in top-3
	on_topic = "vaccine" in ranked_texts[0].lower() or any("vaccine" in ranked_texts[i].lower() for i in range(min(3, len(ranked_texts))))
	assert on_topic, "Reranker failed to place on-topic chunk in top-3"
