from __future__ import annotations

from typing import Dict

from extractor.ranker import score_claim
from phase6_output_ux.verdict_mapping import map_verdict


class Verifier:
	def verify(self, claim: str) -> Dict[str, object]:
		score = score_claim(claim)
		verdict = map_verdict(score)
		return {"score": score, "verdict": verdict}

	def process(self, item: Dict[str, str]) -> Dict[str, object]:
		verification = self.verify(item.get("claim", ""))
		return {**item, "verification": verification}
