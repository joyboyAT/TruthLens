from __future__ import annotations

from typing import Iterable, List


def _normalize(name: str) -> str:
	return (name or "").strip().lower().replace("-", "").replace(" ", "")


_CUE_TO_BADGE = {
	"clickbait": "📢 Clickbait",
	"falseauthority": "👤 False Authority",
	"fearappeal": "⚠️ Fear Appeal",
	"cherrypicking": "📊 Cherry-picking",
	"conspiracy": "🎭 Conspiracy",
}


def generate_cue_badges(cues: Iterable[str]) -> List[str]:
	"""Map a list of cue names to emoji badges.

	Unknown cues are ignored. Matching is case/space/hyphen-insensitive.
	"""
	badges: List[str] = []
	seen = set()
	for cue in cues or []:
		key = _normalize(str(cue))
		badge = _CUE_TO_BADGE.get(key)
		if badge and badge not in seen:
			badges.append(badge)
			seen.add(badge)
	return badges
