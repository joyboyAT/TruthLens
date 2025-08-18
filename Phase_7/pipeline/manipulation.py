from __future__ import annotations

from typing import Iterable, List, Dict

from phase6_output_ux.cue_badges import generate_cue_badges


class ManipulationCueDetector:
	def __init__(self, cues: Iterable[str] | None = None) -> None:
		self.cues = list(cues or [])

	def detect(self, cues: Iterable[str]) -> List[str]:
		return generate_cue_badges(cues)

	def process(self, item: Dict[str, str]) -> Dict[str, object]:
		badges = self.detect(self.cues)
		return {**item, "cue_badges": badges}
