from __future__ import annotations

import re
from typing import Dict


_NEGATION_PATTERNS = [
	" not ",
	"n't",
	" no ",
	" never ",
	" cannot ",
	" can't ",
	" without ",
	" none ",
]

_SPECULATIVE_PATTERNS = [
	" might ",
	" may ",
	" could ",
	" possibly ",
	" perhaps ",
	" likely ",
	" unlikely ",
	" appears to ",
	" seems to ",
	" reportedly ",
	" suggests ",
]

_DEONTIC_PATTERNS = [
	" should ",
	" must ",
	" ought to ",
	" need to ",
]

_CONDITIONAL_PATTERNS = [
	" if ",
	" unless ",
	" provided that ",
	" in case ",
	" assuming ",
	" when ",
]

_ATTRIB_REGEXES = [
	re.compile(r"\baccording to\s+([^,.;:]+)", re.IGNORECASE),
	re.compile(r"\bas per\s+([^,.;:]+)", re.IGNORECASE),
	re.compile(r"\breported by\s+([^,.;:]+)", re.IGNORECASE),
	re.compile(r"\bvia\s+([^,.;:]+)", re.IGNORECASE),
	re.compile(r"\bper\s+([^,.;:]+)", re.IGNORECASE),
]

_SARCASM_STRONG = ["/s", "yeah right", "as if"]
_SARCASM_MEDIUM = ["lol", "lmao", "sure,"]
_SARCASM_WEAK = ["""""", "'" ]  # quotes around words, handled specially


def _contains_any(text: str, patterns: list[str]) -> bool:
	return any(p in text for p in patterns)


def _first_match(text: str, patterns: list[str]) -> str:
	for p in patterns:
		idx = text.find(p)
		if idx != -1:
			return p.strip()
	return ""


def _detect_negation(text: str) -> bool:
	return _contains_any(f" {text} ".lower(), _NEGATION_PATTERNS)


def _detect_modality(text: str) -> str:
	lt = f" {text.lower()} "
	if _contains_any(lt, _SPECULATIVE_PATTERNS):
		return "speculative"
	if _contains_any(lt, _DEONTIC_PATTERNS):
		return "deontic"
	return "assertive"


def _detect_conditional(text: str) -> str:
	return _first_match(f" {text.lower()} ", _CONDITIONAL_PATTERNS)


def _detect_attribution(text: str) -> str:
	for rx in _ATTRIB_REGEXES:
		m = rx.search(text)
		if m:
			return m.group(1).strip()
	return ""


def _sarcasm_score(text: str) -> float:
	lt = text.lower()
	score = 0.0
	if any(tok in lt for tok in _SARCASM_STRONG):
		score = max(score, 0.9)
	if any(tok in lt for tok in _SARCASM_MEDIUM):
		score = max(score, 0.6)
	# Weak heuristic: quoted words can indicate sarcasm/irony
	if re.search(r"\b\"[^\"]+\"\b", text) or re.search(r"\b'[^']+'\b", text):
		score = max(score, 0.4)
	return min(max(score, 0.0), 1.0)


def analyze_context(claim: str, sentence: str) -> Dict[str, object]:
	"""Analyze the context around a claim within a sentence.

	Returns a dict with: {negation, modality, conditional_trigger, sarcasm_score, attribution}
	"""
	text = sentence if sentence else claim
	return {
		"negation": _detect_negation(text),
		"modality": _detect_modality(text),
		"conditional_trigger": _detect_conditional(text),
		"sarcasm_score": _sarcasm_score(text),
		"attribution": _detect_attribution(text),
	}
