from __future__ import annotations

import re
from typing import Dict, Iterable, List


def _normalize_space(text: str) -> str:
	return re.sub(r"\s+", " ", (text or "").strip())


def _truncate(text: str, max_len: int = 240) -> str:
	if len(text) <= max_len:
		return text
	return text[: max_len - 1].rstrip() + "…"


def build_evidence_cards(evidence_list: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
	"""Build formatted evidence cards from raw evidence items.

	Input items should have keys: "text", "url", "source".
	Output cards will have keys: {"source", "highlight", "url"}.
	"""
	cards: List[Dict[str, str]] = []
	for item in evidence_list or []:
		text = _truncate(_normalize_space(str(item.get("text", ""))))
		url = str(item.get("url", "")).strip()
		source = _normalize_space(str(item.get("source", "")))
		if not (text and url and source):
			# Skip incomplete items
			continue
		cards.append({
			"source": source,
			"highlight": text,
			"url": url,
		})
	return cards
