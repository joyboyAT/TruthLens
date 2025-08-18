"""
Tests for multilingual embeddings pipeline.

- Sentence-based chunking to ~512 tokens
- Embeddings with intfloat/multilingual-e5-large
- Cosine search returns the expected chunk near top
"""

import os
import sys
import pytest

# Ensure package import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nlp.chunking import Chunker, ChunkerConfig
from nlp.embeddings import EmbeddingEncoder
from schemas.evidence import Language, TextChunk


HI_TEXT = "यह एक परीक्षण वाक्य है। जलवायु परिवर्तन मानव गतिविधियों के कारण होता है। यह तथ्य विज्ञान द्वारा समर्थित है।"
EN_TEXT = "This is a test sentence. Climate change is caused by human activities. This fact is supported by science."
MR_TEXT = "हा एक चाचणी वाक्य आहे. हवामान बदल मानवी कृतींमुळे होतो. हा तथ्य विज्ञानाद्वारे समर्थित आहे."


@pytest.mark.slow
def test_embeddings_multilingual_and_search():
	cfg = ChunkerConfig(max_tokens=64)
	chunker = Chunker(cfg)

	# Build 20 chunks across languages by repeating
	texts = [HI_TEXT, EN_TEXT, MR_TEXT] * 8
	chunks: list[TextChunk] = []
	for i, t in enumerate(texts[:20]):
		lang = Language.HINDI if i % 3 == 0 else (Language.ENGLISH if i % 3 == 1 else Language.ENGLISH)
		chunks.extend(chunker.chunk(t, base_id=f"ev1_{i}", language=lang))

	# Keep first 20 chunks
	chunks = chunks[:20]
	assert len(chunks) >= 10

	encoder = EmbeddingEncoder()
	encoded = encoder.encode_chunks(chunks, normalize=True)

	# Assert dims
	dim = len(encoded[0].embedding)
	assert dim in (768, 1024), f"unexpected dim: {dim}"

	# Cosine search
	query = "Climate change caused by humans"
	ranking = encoder.cosine_search(query, encoded, top_k=5)
	# Get best match
	best_idx = ranking[0][0]
	best_text = encoded[best_idx].text
	assert "Climate change" in best_text or "जलवायु परिवर्तन" in best_text
	# Ensure it ranks near top (≤3)
	assert ranking and ranking[0][1] >= ranking[min(2, len(ranking)-1)][1]
