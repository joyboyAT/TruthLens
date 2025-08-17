from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, TypedDict


class AtomicClaim(TypedDict, total=False):
	text: str
	subject: str
	predicate: str
	object: str


def _normalize_space(text: str) -> str:
	return re.sub(r"\s+", " ", text.strip())


def _split_tail_modifiers(obj_text: str) -> (str, str):
	"""Split trailing modifiers like 'in 2020', 'on Monday', 'at X', etc."""
	m = re.search(r"\s+(in|on|at|by|during|since|from)\s+.+$", obj_text, flags=re.IGNORECASE)
	if not m:
		return obj_text, ""
	idx = m.start()
	return obj_text[:idx], obj_text[idx:]


def _split_enumeration(text: str) -> List[str]:
	# Split on commas or 'and' when used for coordination
	parts = re.split(r"\s*,\s*|\s+and\s+", text)
	return [p for p in (t.strip() for t in parts) if p]


def _extract_triplet_rule_based(span: str, srl_frames: Optional[Sequence[Dict[str, Any]]]) -> Optional[AtomicClaim]:
	span = _normalize_space(span)
	# Use SRL if provided (expecting ARG0=subject, V=predicate, ARG1=object)
	if srl_frames:
		for fr in srl_frames:
			roles = fr.get("roles") or {}
			subj = roles.get("ARG0") or roles.get("subject")
			pred = roles.get("V") or roles.get("predicate")
			obj = roles.get("ARG1") or roles.get("object")
			if subj and pred and obj:
				text = f"{subj} {pred} {obj}"
				return AtomicClaim(text=text, subject=subj, predicate=pred, object=obj)
	# Naive regex fallback: subject + predicate + object(+mods)
	m = re.match(r"^(?P<subj>[^,;]+?)\s+(?P<pred>\b\w+(?:[\s-]+\w+){0,2})\s+(?P<obj>.+)$", span)
	if not m:
		return None
	subj = _normalize_space(m.group("subj"))
	pred = _normalize_space(m.group("pred"))
	obj = _normalize_space(m.group("obj"))
	return AtomicClaim(text=f"{subj} {pred} {obj}", subject=subj, predicate=pred, object=obj)


def _llm_atomicize(span: str) -> List[AtomicClaim]:
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		return []
	try:
		from openai import OpenAI
		client = OpenAI(api_key=api_key)
		prompt = (
			"Split into independent factual statements JSON [{text, subject, predicate, object}].\n"
			f"Input: {span}"
		)
		resp = client.chat.completions.create(
			model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
			messages=[
				{"role": "system", "content": "You are a precise information extraction assistant."},
				{"role": "user", "content": prompt},
			],
			temperature=0.0,
		)
		content = resp.choices[0].message.content.strip()
		data = json.loads(content)
		results: List[AtomicClaim] = []
		for item in data:
			claim = AtomicClaim(
				text=item.get("text", ""),
				subject=item.get("subject", ""),
				predicate=item.get("predicate", ""),
				object=item.get("object", ""),
			)
			results.append(claim)
		return results
	except Exception:
		return []


def to_atomic(span: str, srl_frames: Optional[Sequence[Dict[str, Any]]]) -> List[AtomicClaim]:
	"""Split a span into atomic factual claims.

	1) Try rule-based using SRL roles or simple patterns. If the object contains
	   coordination (commas/and), split into multiple claims and keep trailing
	   modifiers (e.g., 'in 2020').
	2) If ambiguous or nothing splits cleanly, try an LLM prompt.
	"""
	base = _extract_triplet_rule_based(span, srl_frames)
	if not base:
		llm = _llm_atomicize(span)
		return llm or []
	obj_core, tail = _split_tail_modifiers(base["object"]) if base.get("object") else ("", "")
	items = _split_enumeration(obj_core)
	if len(items) <= 1:
		# Try LLM if we couldn't split
		llm = _llm_atomicize(span)
		return llm or [base]
	claims: List[AtomicClaim] = []
	for it in items:
		obj_text = (it + tail).strip()
		text = f"{base['subject']} {base['predicate']} {obj_text}".strip()
		claims.append(
			AtomicClaim(text=text, subject=base["subject"], predicate=base["predicate"], object=obj_text)
		)
	return claims
