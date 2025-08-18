from __future__ import annotations

import re
from typing import Mapping

from .utils import normalize_space


def _shorten_url(url: str, max_len: int = 28) -> str:
	u = re.sub(r"^https?://", "", (url or "").strip())
	if len(u) <= max_len:
		return u
	return u[: max_len - 1].rstrip("/") + "…"


def _evidence_text(e: Mapping[str, str] | str) -> str:
	if isinstance(e, str):
		return normalize_space(e)
	text = e.get("highlight") or e.get("text") or ""
	url = e.get("url") or ""
	high = normalize_space(text)
	short = _shorten_url(url)
	if high and short:
		return f"{high} ({short})"
	return high or short


def build_share_card(claim: str, verdict: str, evidence: Mapping[str, str] | str, tip: str) -> str:
	claim_text = normalize_space(claim)
	verdict_text = normalize_space(verdict)
	evidence_text = _evidence_text(evidence)
	tip_text = normalize_space(tip)
	lines = [f"Claim: {claim_text}", f"Verdict: {verdict_text}"]
	if evidence_text:
		lines.append(f"Evidence: {evidence_text}")
	if tip_text:
		lines.append(f"Tip: {tip_text}")
	return "\n".join(lines)
