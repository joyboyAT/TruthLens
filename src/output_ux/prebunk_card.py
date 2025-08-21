from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

from .utils import normalize_space


def _norm(name: str) -> str:
	return normalize_space(name).lower().replace("-", "").replace(" ", "")


def build_prebunk_card(
	cues: Iterable[str],
	prebunk_tips: Mapping[str, str] | Sequence[str],
) -> Dict[str, object]:
	collected: list[str] = []
	seen: set[str] = set()
	if isinstance(prebunk_tips, dict):  # type: ignore[untyped-call]
		key_to_tip = {_norm(k): normalize_space(v) for k, v in prebunk_tips.items()}
		for cue in cues or []:
			t = key_to_tip.get(_norm(str(cue)), "")
			if t and t not in seen:
				collected.append(t)
				seen.add(t)
	else:
		for tip in prebunk_tips or []:  # type: ignore[assignment]
			ts = normalize_space(str(tip))
			if ts and ts not in seen:
				collected.append(ts)
				seen.add(ts)
	return {"title": "Prebunk Tips", "tips": collected}
