"""
Cron-style refresh job for TruthLens Phase 3.

Usage:
  python -m jobs.refresh --claims-file claims.txt --cache-file cache.json

- Reads claims, checks TTL policy via cache.policy
- If needed, triggers a live search (broader window) using a provided Searcher
- Writes back updated cache
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from datetime import datetime, timezone

# Import policy and (optional) grounded search client
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from cache.policy import refresh_claim_if_needed  # type: ignore
from retrieval.grounded_search import make_default_client, GroundedSearcher  # type: ignore


class GroundedSearcherAdapter:
	"""Adapt GroundedSearcher to the Searcher Protocol used in policy."""
	def __init__(self):
		client = make_default_client()
		if client is None:
			raise RuntimeError("No search API key configured for grounded search")
		self.searcher = GroundedSearcher(client)
	def search(self, claim: str, entities: Optional[List[str]] = None, top_k: int = 5, days: Optional[int] = 90) -> List[Dict[str, Any]]:
		# Return list of dicts to align with policy normalization path
		out: List[Dict[str, Any]] = []
		items = self.searcher.search(claim, entities=entities, top_k=top_k, days=days)
		for ev in items:
			out.append({
				"domain": ev.domain,
				"retrieved_at": ev.retrieved_at,
				"support_label": getattr(ev, "support_label", ""),
				"url": ev.url,
				"title": ev.title,
			})
		return out


def load_claims(path: Path) -> List[str]:
	with path.open("r", encoding="utf-8") as f:
		return [line.strip() for line in f if line.strip()]


def load_cache(path: Path) -> Dict[str, List[Dict[str, Any]]]:
	if not path.exists():
		return {}
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def save_cache(path: Path, cache: Dict[str, List[Dict[str, Any]]]) -> None:
	with path.open("w", encoding="utf-8") as f:
		json.dump(cache, f, ensure_ascii=False, default=str, indent=2)


def main():
	parser = argparse.ArgumentParser(description="Refresh hot items for claims cache")
	parser.add_argument("--claims-file", required=True, help="Path to a text file with one claim per line")
	parser.add_argument("--cache-file", required=True, help="Path to JSON cache file")
	parser.add_argument("--entities-file", help="Optional JSON file with claim->entities mapping")
	parser.add_argument("--dry-run", action="store_true", help="Don't write cache; only print actions")
	args = parser.parse_args()

	claims_path = Path(args.claims_file)
	cache_path = Path(args.cache_file)
	entities_map: Dict[str, List[str]] = {}
	if args.entities_file:
		p = Path(args.entities_file)
		if p.exists():
			entities_map = json.loads(p.read_text(encoding="utf-8"))

	claims = load_claims(claims_path)
	cache = load_cache(cache_path)
	searcher = GroundedSearcherAdapter()

	refreshed = 0
	for claim in claims:
		ents = entities_map.get(claim)
		if refresh_claim_if_needed(claim, cache, searcher, entities=ents):
			print(f"Refreshed: {claim}")
			refreshed += 1
		else:
			print(f"Up-to-date: {claim}")

	print(f"Total refreshed: {refreshed}/{len(claims)}")
	if not args.dry_run:
		save_cache(cache_path, cache)


if __name__ == "__main__":
	main()
