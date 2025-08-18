from __future__ import annotations

import re
from typing import Iterable, List


def normalize_space(text: str) -> str:
	return re.sub(r"\s+", " ", (text or "").strip())


def clamp01(value: float) -> float:
	try:
		v = float(value)
	except (TypeError, ValueError):
		v = 0.0
	return max(0.0, min(1.0, v))


def uniq_keep_order(items: Iterable[str]) -> List[str]:
	seen = set()
	out: List[str] = []
	for it in items:
		if it not in seen:
			seen.add(it)
			out.append(it)
	return out
