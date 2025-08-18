from __future__ import annotations

def map_verdict(score: float) -> str:
	"""Map a score in [0,1] to a human-readable verdict with the score.

	- score >= 0.7 -> "🟢 Likely True"
	- 0.4 <= score < 0.7 -> "🟡 Unclear"
	- score < 0.4 -> "🔴 Likely False"
	"""
	try:
		s = float(score)
	except (TypeError, ValueError):
		s = 0.0
	# Clamp to [0,1]
	s = max(0.0, min(1.0, s))
	if s >= 0.7:
		verdict = "🟢 Likely True"
	elif s >= 0.4:
		verdict = "🟡 Unclear"
	else:
		verdict = "🔴 Likely False"
	return f"{verdict} — {s:.2f}"
