"""
Second-stage re-ranking for TruthLens Phase 3.

- Re-ranker: BAAI/bge-reranker-v2-m3 (multilingual)
- Input: list of (claim, TextChunk)
- Output: sorted candidates with relevance_score attached
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Schemas
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from schemas.evidence import TextChunk  # type: ignore

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
	"""Scores (claim, chunk_text) pairs and sorts by relevance."""

	def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
		self.model.eval()

	@torch.no_grad()
	def score_pairs(self, claim: str, chunks: List[TextChunk], batch_size: int = 8) -> List[float]:
		scores: List[float] = []
		for i in range(0, len(chunks), batch_size):
			batch = chunks[i : i + batch_size]
			pairs = [[claim, c.text] for c in batch]
			inputs = self.tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
			logits = self.model(**inputs).logits.view(-1)
			batch_scores = logits.detach().cpu().tolist()
			scores.extend(batch_scores)
		return scores

	def rerank(self, claim: str, chunks: List[TextChunk], top_k: int | None = None) -> List[Tuple[TextChunk, float]]:
		scores = self.score_pairs(claim, chunks)
		paired = list(zip(chunks, scores))
		paired.sort(key=lambda x: x[1], reverse=True)
		if top_k is not None:
			paired = paired[:top_k]
		return paired
