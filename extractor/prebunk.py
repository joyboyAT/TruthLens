from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence


def _normalize(name: str) -> str:
	return (name or "").strip().lower().replace("-", "").replace(" ", "")


def build_prebunk_card(
	cues: Iterable[str],
	prebunk_tips: Mapping[str, str] | Sequence[str],
) -> Dict[str, object]:
	"""Build a prebunk tips card from cues and their tips.

	- If prebunk_tips is a mapping, cue names are normalized (case/space/hyphen-insensitive)
	  and matched to tips in the provided mapping.
	- If prebunk_tips is a sequence of strings, they are used as-is.
	"""
	collected: List[str] = []
	seen: set[str] = set()

	# If a dict mapping was provided, select tips matching the cues order
	if isinstance(prebunk_tips, dict):  # type: ignore[untyped-call]
		cue_keys = [_normalize(str(c)) for c in (cues or [])]
		key_to_tip: Dict[str, str] = { _normalize(k): str(v).strip() for k, v in prebunk_tips.items() }
		for key in cue_keys:
			tip = key_to_tip.get(key, "")
			if tip and tip not in seen:
				collected.append(tip)
				seen.add(tip)
	else:
		# Assume iterable of tip strings
		for tip in (prebunk_tips or []):  # type: ignore[assignment]
			tip_str = str(tip).strip()
			if tip_str and tip_str not in seen:
				collected.append(tip_str)
				seen.add(tip_str)

	return {
		"title": "Prebunk Tips",
		"tips": collected,
	}
