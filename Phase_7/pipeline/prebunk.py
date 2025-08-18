from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

from phase6_output_ux.prebunk_card import build_prebunk_card


class PrebunkGenerator:
	def __init__(self, cues: Iterable[str] | None = None, tips: Mapping[str, str] | Sequence[str] | None = None) -> None:
		self.cues = list(cues or [])
		self.tips = tips or {}

	def build(self, cues: Iterable[str], tips: Mapping[str, str] | Sequence[str]) -> Dict[str, object]:
		return build_prebunk_card(cues, tips)

	def process(self, item: Dict[str, str]) -> Dict[str, object]:
		card = self.build(self.cues, self.tips)
		return {**item, "prebunk": card}
