from __future__ import annotations

import re
from typing import Dict, Mapping, Optional


def _normalize_space(text: str) -> str:
	return re.sub(r"\s+", " ", (text or "").strip())


def _shorten_url(url: str, max_len: int = 28) -> str:
	if not url:
		return ""
	u = url.strip()
	u = re.sub(r"^https?://", "", u)
	if len(u) <= max_len:
		return u
	return u[: max_len - 1].rstrip("/") + "…"


def _evidence_to_text(evidence: Mapping[str, str] | str) -> str:
	if isinstance(evidence, str):
		return _normalize_space(evidence)
	text = evidence.get("highlight") or evidence.get("text") or ""
	url = evidence.get("url") or ""
	highlight = _normalize_space(text)
	short_url = _shorten_url(url)
	if highlight and short_url:
		return f"{highlight} ({short_url})"
	return highlight or short_url


def build_share_card(
	claim: str,
	verdict: str,
	evidence: Mapping[str, str] | str,
	tip: str,
) -> str:
	"""Build a shareable debunk text block.

	Format:
	Claim: <claim>
	Verdict: <verdict>
	Evidence: <highlight> (<short_url>)
	Tip: <tip>
	"""
	claim_text = _normalize_space(claim)
	verdict_text = _normalize_space(verdict)
	evidence_text = _evidence_to_text(evidence)
	tip_text = _normalize_space(tip)
	lines = [
		f"Claim: {claim_text}",
		f"Verdict: {verdict_text}",
	]
	if evidence_text:
		lines.append(f"Evidence: {evidence_text}")
	if tip_text:
		lines.append(f"Tip: {tip_text}")
	return "\n".join(lines)
