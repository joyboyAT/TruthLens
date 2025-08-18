from __future__ import annotations

import re
from typing import List


def normalize_whitespace(text: str) -> str:
	return re.sub(r"\s+", " ", (text or "").strip())


def split_sentences(doc: str) -> List[str]:
	text = normalize_whitespace(doc)
	if not text:
		return []
	parts = re.split(r"(?<=[.!?])\s+", text)
	return [p.strip() for p in parts if p.strip()]
