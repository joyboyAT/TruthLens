from __future__ import annotations

import os
from typing import Dict, List

from extractor.claim_detector import is_claim
from extractor.claim_extractor import extract_claim_spans
from extractor.atomicizer import to_atomic


class ClaimExtractor:
	def __init__(self, use_ml: bool = False) -> None:
		self.use_ml = use_ml

	def process(self, item: Dict[str, str]) -> List[Dict[str, str]]:
		"""Accepts {'claim': str, 'context': str} and returns a list of such dicts with extracted claims.

		The 'claim' field in input can be empty; 'context' is the sentence/source.
		"""
		if not self.use_ml:
			os.environ.setdefault("TRUTHLENS_DISABLE_ML", "1")
		context_text = (item.get("context") or "").strip()
		if not context_text:
			return []
		ok, _ = is_claim(context_text)
		if not ok:
			return []
		results: List[Dict[str, str]] = []
		spans = extract_claim_spans(context_text)
		if not spans:
			spans = [{"text": context_text, "start": 0, "end": len(context_text), "conf": 0.5}]
		for sp in spans:
			atoms = to_atomic(sp.get("text", ""), None) or []
			for a in atoms:
				claim_text = (a.get("text") or "").strip()
				if claim_text:
					results.append({"claim": claim_text, "context": context_text})
		return results

	# Backward compatibility
	def extract(self, text: str) -> List[str]:
		items = self.process({"claim": "", "context": text})
		return [it["claim"] for it in items]
