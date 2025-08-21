from __future__ import annotations

from typing import Dict, Iterable, List

from .utils import normalize_space


def _truncate(text: str, max_len: int = 240) -> str:
	if len(text) <= max_len:
		return text
	return text[: max_len - 1].rstrip() + "…"


def build_evidence_cards(evidence_list: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
	cards: List[Dict[str, str]] = []
	for item in evidence_list or []:
		text = _truncate(normalize_space(str(item.get("text", ""))))
		url = str(item.get("url", "")).strip()
		source = normalize_space(str(item.get("source", "")))
		if not (text and url and source):
			continue
		cards.append({"source": source, "highlight": text, "url": url})
	return cards
