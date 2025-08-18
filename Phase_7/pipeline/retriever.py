from __future__ import annotations

from typing import Dict, List


class EvidenceRetriever:
	def __init__(self, evidence_corpus: List[Dict[str, str]] | None = None) -> None:
		self.evidence_corpus = evidence_corpus or []

	def top_evidence(self, claim: str, k: int = 1) -> List[Dict[str, str]]:
		return self.evidence_corpus[:k]

	def process(self, item: Dict[str, str]) -> Dict[str, object]:
		evid = self.top_evidence(item.get("claim", ""), k=1)
		return {**item, "evidence": evid}
