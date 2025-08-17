from __future__ import annotations

import re
import uuid
from typing import Dict, List, TypedDict

from .claim_detector import is_claim
from .claim_extractor import extract_claim_spans
from .atomicizer import to_atomic
from .context import analyze_context
from .ranker import score_claim


class AtomicClaimJSON(TypedDict):
	id: str
	text: str
	subject: str
	predicate: str
	object: str
	context: Dict[str, object]
	checkworthiness: float


def _normalize(text: str) -> str:
	return re.sub(r"\s+", " ", (text or "").strip())


def _split_sentences(doc: str) -> List[str]:
	# Lightweight sentence splitter; keeps punctuation
	text = _normalize(doc)
	if not text:
		return []
	parts = re.split(r"(?<=[.!?])\s+", text)
	return [p.strip() for p in parts if p.strip()]


def process_text(doc: str) -> List[AtomicClaimJSON]:
	"""Process raw text into atomic claims with context and scores.

	Steps: preprocess → claim_detect → span_extract → atomicize → context → rank
	"""
	results: List[AtomicClaimJSON] = []
	for sent in _split_sentences(doc):
		label, prob = is_claim(sent)
		if not label:
			continue
		spans = extract_claim_spans(sent)
		if not spans:
			# Fall back to treating the whole sentence as a single span
			spans = [{"text": sent, "start": 0, "end": len(sent), "conf": prob}]
		for sp in spans:
			atomic_claims = to_atomic(sp.get("text", ""), None) or []
			if not atomic_claims:
				# Minimal fallback when atomicization yields nothing
				atomic_claims = [{"text": sp.get("text", ""), "subject": "", "predicate": "", "object": ""}]
			for ac in atomic_claims:
				claim_text = _normalize(ac.get("text", ""))
				ctx = analyze_context(claim_text, sent)
				score = score_claim(claim_text)
				results.append(
					AtomicClaimJSON(
						id=str(uuid.uuid4()),
						text=claim_text,
						subject=_normalize(ac.get("subject", "")),
						predicate=_normalize(ac.get("predicate", "")),
						object=_normalize(ac.get("object", "")),
						context=ctx,
						checkworthiness=float(max(0.0, min(1.0, score))),
					)
				)
	return results
