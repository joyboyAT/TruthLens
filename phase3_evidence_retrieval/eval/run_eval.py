"""
Evaluation runner for TruthLens Phase 3.

Metrics:
- Top-1 / Top-3 evidence accuracy (heuristic via title overlap)
- Label accuracy (support/contradict) if labels available
- Freshness coverage (≤30 days)

Also: blocklist low-cred domains and log if any slip through.
Print a small leaderboard per source type; persist CSV.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from retrieval.grounded_search import make_default_client, GroundedSearcher  # type: ignore
from connectors.wikipedia import WikipediaConnector  # type: ignore
from schemas.evidence import RawEvidence  # type: ignore

LOW_CRED_BLOCKLIST = [
	"beforeitsnews.com",
	"naturalnews.com",
	"yournewswire.com",
	"infowars.com",
	"thegatewaypundit.com",
]


def blocked(domain: str) -> bool:
	d = (domain or "").lower()
	return any(b in d for b in LOW_CRED_BLOCKLIST)


def load_claims(path: Path) -> List[Dict[str, Any]]:
	with path.open("r", encoding="utf-8") as f:
		return [json.loads(line) for line in f if line.strip()]


def search_evidence(claim: str, entities: Optional[List[str]], max_results: int = 5) -> List[RawEvidence]:
	client = make_default_client()
	if client:
		return GroundedSearcher(client).search(claim, entities=entities, top_k=max_results, days=90)
	# fallback to wikipedia background
	conn = WikipediaConnector(lang="en")
	items: List[RawEvidence] = []
	ents = entities or [claim]
	for e in ents[:max_results]:
		ev = conn.fetch_background(e)
		if ev:
			if ev.published_at is None:
				ev.published_at = ev.retrieved_at or datetime.now(timezone.utc)
			items.append(ev)
	return items


def title_overlap_hit(claim: str, title: str) -> bool:
	claim_l = set([t.lower() for t in claim.split() if len(t) >= 4])
	title_l = title.lower()
	return any(t in title_l for t in claim_l)


@dataclass
class EvalStats:
	top1: int = 0
	top3: int = 0
	label_correct: int = 0
	count: int = 0
	fresh_30d: int = 0


def run_eval(claims: List[Dict[str, Any]]) -> Dict[str, Any]:
	rows = []
	stats = EvalStats()
	per_source = {}
	for entry in claims:
		claim = entry["claim"]
		lang = entry.get("lang", "en")
		verdict = entry.get("verdict")
		items = search_evidence(claim, entities=None, max_results=5)
		items = [ev for ev in items if not blocked(ev.domain)]

		# Compute top-k evidence accuracy (requires title overlap heuristic)
		stats.count += 1
		if not items:
			continue
		if title_overlap_hit(claim, items[0].title or ""):
			stats.top1 += 1
		if any(title_overlap_hit(claim, it.title or "") for it in items[:3]):
			stats.top3 += 1

		# Freshness coverage ≤30 days
		now = datetime.now(timezone.utc)
		if any((now - (it.published_at or now)).days <= 30 for it in items):
			stats.fresh_30d += 1

		# Label accuracy if labels exist (support/contradict)
		labels = [getattr(it, "support_label", None) for it in items]
		if any(x for x in labels):
			# crude mapping: verdict true→supports, false→refutes
			exp = "supports" if verdict == "true" else ("refutes" if verdict == "false" else None)
			if exp:
				if any((str(x or "").lower().startswith(exp[:3])) for x in labels[:3]):
					stats.label_correct += 1

		# Leaderboard per source type (domain prefix)
		for it in items:
			key = it.domain.split("/")[0]
			per_source.setdefault(key, {"count": 0})
			per_source[key]["count"] += 1

		rows.append({
			"claim": claim,
			"lang": lang,
			"verdict": verdict,
			"top1_hit": int(title_overlap_hit(claim, items[0].title or "")),
			"top3_hit": int(any(title_overlap_hit(claim, it.title or "") for it in items[:3])),
			"fresh_30d": int(any((now - (it.published_at or now)).days <= 30 for it in items)),
			"domains": ",".join([it.domain for it in items]),
		})

	return {
		"stats": {
			"top1": stats.top1,
			"top3": stats.top3,
			"label_correct": stats.label_correct,
			"count": stats.count,
			"fresh_30d": stats.fresh_30d,
		},
		"per_source": per_source,
		"rows": rows,
	}


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
	if not rows:
		return
	cols = list(rows[0].keys())
	with path.open("w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=cols)
		w.writeheader()
		for r in rows:
			w.writerow(r)


def main():
	import argparse
	parser = argparse.ArgumentParser(description="Run evaluation for TruthLens")
	parser.add_argument("--claims", default=str(Path(__file__).with_name("claims.jsonl")), help="Path to claims.jsonl")
	parser.add_argument("--out", default="eval_results.csv", help="Output CSV path")
	args = parser.parse_args()

	claims = load_claims(Path(args.claims))
	res = run_eval(claims)

	print("== Aggregate Stats ==")
	print(json.dumps(res["stats"], indent=2))
	print("\n== Per-Source Leaderboard ==")
	for k, v in sorted(res["per_source"].items(), key=lambda x: -x[1]["count"]):
		print(f"{k}: {v['count']}")

	save_csv(res["rows"], Path(args.out))
	print(f"Saved CSV: {args.out}")


if __name__ == "__main__":
	main()
