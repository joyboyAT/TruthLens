from __future__ import annotations

from .utils import clamp01


def map_verdict(score: float) -> str:
	"""Map a score in [0,1] to a verdict string with emoji and score."""
	s = clamp01(score)
	if s >= 0.7:
		label = "🟢 Likely True"
	elif s >= 0.4:
		label = "🟡 Unclear"
	else:
		label = "🔴 Likely False"
	return f"{label} — {s:.2f}"
