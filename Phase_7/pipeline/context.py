from __future__ import annotations

from typing import Dict

from extractor.context import analyze_context


class ContextProcessor:
	def process(self, item: Dict[str, str]) -> Dict[str, object]:
		ctx = analyze_context(item.get("claim", ""), item.get("context", ""))
		return {**item, "context_info": ctx}

	# Backward compatibility
	def analyze(self, claim: str, sentence: str) -> Dict[str, object]:
		return analyze_context(claim, sentence)
