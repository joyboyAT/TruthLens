from __future__ import annotations

from typing import Iterable, List

from .utils import normalize_space, uniq_keep_order


def _key(name: str) -> str:
	return normalize_space(name).lower().replace("-", "").replace(" ", "")


_CUE_TO_BADGE = {
	"clickbait": "📢 Clickbait",
	"falseauthority": "👤 False Authority",
	"fearappeal": "⚠️ Fear Appeal",
	"cherrypicking": "📊 Cherry-picking",
	"conspiracy": "🎭 Conspiracy",
}


def generate_cue_badges(cues: Iterable[str]) -> List[str]:
	badges: List[str] = []
	for cue in cues or []:
		badge = _CUE_TO_BADGE.get(_key(str(cue)))
		if badge:
			badges.append(badge)
	return uniq_keep_order(badges)
