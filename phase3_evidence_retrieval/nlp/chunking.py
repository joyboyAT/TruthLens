"""
Semantic chunking for TruthLens Phase 3.

- Sentence-based splitting
- ~512 token limit using HF tokenizer lengths
- Optional sentence overlap between chunks
- Keeps language per chunk
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

from transformers import AutoTokenizer

# Schemas
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from schemas.evidence import TextChunk, Language  # type: ignore


_SENT_SPLIT_REGEX = re.compile(r"(?<=[.!?।])\s+")


def split_sentences(text: str) -> List[str]:
	text = (text or "").strip()
	if not text:
		return []
	# Fast split on end punctuation; fallback to whole text
	sents = [s.strip() for s in _SENT_SPLIT_REGEX.split(text) if s.strip()]
	return sents if sents else [text]


@dataclass
class ChunkerConfig:
	model_name: str = "intfloat/multilingual-e5-large"
	max_tokens: int = 512
	overlap_sentences: int = 1


class Chunker:
	"""Sentence-based semantic chunker using a HF tokenizer for token budget."""

	def __init__(self, config: Optional[ChunkerConfig] = None):
		self.config = config or ChunkerConfig()
		self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)

	def _count_tokens(self, text: str) -> int:
		return len(self.tokenizer.encode(text, add_special_tokens=False))

	def chunk(self, text: str, base_id: str = "chunk", language: Language = Language.ENGLISH) -> List[TextChunk]:
		sentences = split_sentences(text)
		chunks: List[TextChunk] = []
		if not sentences:
			return chunks

		current: List[str] = []
		current_token_count = 0
		start_char = 0
		chunk_index = 0

		for i, sent in enumerate(sentences):
			sent_tokens = self._count_tokens(sent)
			if not current:
				start_char = text.find(sent, start_char)
			# If adding this sentence would exceed budget, flush current
			if current and current_token_count + sent_tokens > self.config.max_tokens:
				chunk_text = " ".join(current).strip()
				end_char = start_char + len(chunk_text)
				chunks.append(
					TextChunk(
						chunk_id=f"{base_id}_{chunk_index}",
						text=chunk_text,
						chunk_index=chunk_index,
						start_char=start_char,
						end_char=end_char,
						embedding=None,
						embedding_model=None,
						language=language,
						metadata=None,
					)
				)
				chunk_index += 1
				# Prepare next window with overlap
				overlap = max(0, self.config.overlap_sentences)
				current = current[-overlap:] if overlap > 0 else []
				current_token_count = self._count_tokens(" ".join(current)) if current else 0
				start_char = end_char  # approximate
			# Add sentence
			current.append(sent)
			current_token_count += sent_tokens

		# Flush remainder
		if current:
			chunk_text = " ".join(current).strip()
			end_char = start_char + len(chunk_text)
			chunks.append(
				TextChunk(
					chunk_id=f"{base_id}_{chunk_index}",
					text=chunk_text,
					chunk_index=chunk_index,
					start_char=start_char,
					end_char=end_char,
					embedding=None,
					embedding_model=None,
					language=language,
					metadata=None,
				)
			)

		return chunks
