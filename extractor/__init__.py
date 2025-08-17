# TruthLens extractor package

from __future__ import annotations

from typing import Any

__all__ = [
	"is_claim",
	"extract_claim_spans",
	"to_atomic",
	"analyze_context",
	"score_claim",
	"process_text",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy import
	if name == "is_claim":
		from .claim_detector import is_claim as f
		return f
	if name == "extract_claim_spans":
		from .claim_extractor import extract_claim_spans as f
		return f
	if name == "to_atomic":
		from .atomicizer import to_atomic as f
		return f
	if name == "analyze_context":
		from .context import analyze_context as f
		return f
	if name == "score_claim":
		from .ranker import score_claim as f
		return f
	if name == "process_text":
		from .pipeline import process_text as f
		return f
	raise AttributeError(name)
