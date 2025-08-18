"""
First-stage retrieval (ANN-like) for TruthLens Phase 3.

- Query embedding: multilingual-e5 of the claim
- Top-k from an in-memory vector store (cosine similarity)
- Can be adapted to Vector DB / BigQuery vectors
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import torch

# Reuse encoder and schema
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from nlp.embeddings import EmbeddingEncoder  # type: ignore
from schemas.evidence import TextChunk  # type: ignore


def _normalize(M: torch.Tensor) -> torch.Tensor:
	return M / (M.norm(dim=-1, keepdim=True) + 1e-12)


class InMemoryVectorIndex:
	"""Lightweight in-memory vector index over TextChunks (cosine similarity)."""

	def __init__(self, chunks: List[TextChunk]):
		self.chunks = chunks
		emb = [c.embedding for c in chunks]
		self.M = _normalize(torch.tensor(emb, dtype=torch.float)) if emb else torch.empty(0, 1)

	def search(self, qvec: torch.Tensor, top_k: int = 10) -> List[Tuple[int, float]]:
		if self.M.numel() == 0:
			return []
		q = _normalize(qvec)
		scores = torch.mv(self.M, q)
		vals, idx = torch.topk(scores, k=min(top_k, len(self.chunks)))
		return [(int(idx[i].item()), float(vals[i].item())) for i in range(len(vals))]


class VectorSearcher:
	"""Encodes claim → searches chunks by cosine similarity."""

	def __init__(self, encoder: Optional[EmbeddingEncoder] = None):
		self.encoder = encoder or EmbeddingEncoder()

	def build_index(self, chunks: List[TextChunk]) -> InMemoryVectorIndex:
		# Ensure chunks have embeddings
		missing = [c for c in chunks if not c.embedding]
		if missing:
			self.encoder.encode_chunks(missing, normalize=True)
		return InMemoryVectorIndex(chunks)

	def encode_query(self, claim: str) -> torch.Tensor:
		vec = self.encoder.model.encode([claim], normalize_embeddings=False)
		return torch.tensor(vec[0], dtype=torch.float)

	def search(self, claim: str, chunks: List[TextChunk], top_k: int = 10) -> List[Tuple[int, float]]:
		index = self.build_index(chunks)
		q = self.encode_query(claim)
		return index.search(q, top_k=top_k)
