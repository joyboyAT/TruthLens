"""
Multilingual embeddings for TruthLens Phase 3.

- Model: intfloat/multilingual-e5-large (good for Indic & English)
- Encodes TextChunk objects and attaches embeddings
- Provides cosine similarity search helper for debugging

Note: DB upsert is provided as a stub; integrate with your storage layer.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import math

import torch
from sentence_transformers import SentenceTransformer

# Schemas
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from schemas.evidence import TextChunk  # type: ignore


MODEL_NAME = "intfloat/multilingual-e5-large"


def _normalize(vec: torch.Tensor) -> torch.Tensor:
	return vec / (vec.norm(dim=-1, keepdim=True) + 1e-12)


class EmbeddingEncoder:
	"""Embeds text chunks using multilingual-e5 model."""

	def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.model = SentenceTransformer(model_name, device=self.device)
		self.model.eval()
		# Dimension
		test = self.model.encode(["test"], normalize_embeddings=False)
		self.dim = len(test[0])

	def encode_chunks(self, chunks: List[TextChunk], normalize: bool = True, show_progress_bar: bool = False) -> List[TextChunk]:
		texts = [c.text for c in chunks]
		vectors = self.model.encode(texts, normalize_embeddings=False, show_progress_bar=show_progress_bar)
		if normalize:
			vectors = _normalize(torch.tensor(vectors)).cpu().numpy()
		for i, c in enumerate(chunks):
			c.embedding = vectors[i].tolist()
			c.embedding_model = MODEL_NAME
		return chunks

	def cosine_search(self, query: str, chunks: List[TextChunk], top_k: int = 5) -> List[Tuple[int, float]]:
		qs = self.model.encode([query], normalize_embeddings=False)
		q = _normalize(torch.tensor(qs[0]))
		M = torch.tensor([c.embedding for c in chunks], dtype=torch.float)
		M = _normalize(M)
		scores = torch.mv(M, q)
		vals, idx = torch.topk(scores, k=min(top_k, len(chunks)))
		return [(int(idx[i].item()), float(vals[i].item())) for i in range(len(vals))]


# ---- Storage stub (replace with your DB upsert) ----

def upsert_chunks(evidence_id: str, chunks: List[TextChunk]) -> int:
	"""Stub: upsert chunks into evidence_chunks table.

	Replace this with real DB upsert logic (e.g., PostgreSQL + pgvector).
	Returns number of rows upserted.
	"""
	# Here we simply return count; integration is left to persistence layer
	return len(chunks)
