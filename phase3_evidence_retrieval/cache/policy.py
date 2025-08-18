"""
Caching policy (TTL & refresh) for TruthLens Phase 3.

- TTL by source: gov/health (7–30 days), news (3–7 days), fact-checks (30–180 days)
- If claim has no supporting evidence or only stale hits → trigger live search again with broader window
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Protocol

# Simple source categorization
GOV_HEALTH_DOMAINS = (
	".gov",
	"who.int",
	"cdc.gov",
	"nih.gov",
	"fda.gov",
	"europa.eu",
	"gov.in",
	"gov.uk",
)
FACT_CHECK_DOMAINS = (
	"politifact.com",
	"boomlive.in",
	"factcheck.org",
	"snopes.com",
	"fullfact.org",
)
NEWS_DOMAINS = (
	"reuters.com",
	"apnews.com",
	"ap.org",
	"bbc.com",
	"nytimes.com",
	"washingtonpost.com",
	"wsj.com",
	"theguardian.com",
)


def _domain_contains(domain: str, needles: tuple[str, ...]) -> bool:
	d = (domain or "").lower()
	return any(n in d for n in needles)


def categorize_source(domain: str) -> str:
	if _domain_contains(domain, GOV_HEALTH_DOMAINS):
		return "gov_health"
	if _domain_contains(domain, FACT_CHECK_DOMAINS):
		return "fact_check"
	if _domain_contains(domain, NEWS_DOMAINS):
		return "news"
	return "other"


@dataclass
class TTLConfig:
	gov_health_days: int = 21   # within 7–30
	news_days: int = 5          # within 3–7
	fact_check_days: int = 90   # within 30–180
	other_days: int = 7


def ttl_days_for_domain(domain: str, cfg: Optional[TTLConfig] = None) -> int:
	cfg = cfg or TTLConfig()
	cat = categorize_source(domain)
	return {
		"gov_health": cfg.gov_health_days,
		"fact_check": cfg.fact_check_days,
		"news": cfg.news_days,
		"other": cfg.other_days,
	}[cat]


def is_expired(retrieved_at: Optional[datetime], domain: str, now: Optional[datetime] = None, cfg: Optional[TTLConfig] = None) -> bool:
	if retrieved_at is None:
		return True
	now = now or datetime.now(timezone.utc)
	if retrieved_at.tzinfo is None:
		retrieved_at = retrieved_at.replace(tzinfo=timezone.utc)
	days = (now - retrieved_at).total_seconds() / 86400.0
	return days >= ttl_days_for_domain(domain, cfg)


def should_live_search(claim: str, evidence_items: List[Dict[str, Any]], now: Optional[datetime] = None, cfg: Optional[TTLConfig] = None) -> bool:
	"""Decide whether to trigger a live search.

	- If all cached items are expired → True
	- If there is no supporting evidence (label supports/entail) → True
	Otherwise False
	"""
	now = now or datetime.now(timezone.utc)
	cfg = cfg or TTLConfig()
	if not evidence_items:
		return True
	all_expired = True
	has_support = False
	for it in evidence_items:
		domain = it.get("domain", "")
		retrieved_at = it.get("retrieved_at")  # datetime expected
		if isinstance(retrieved_at, str):
			try:
				from dateutil.parser import isoparse  # optional; if missing, skip parsing
				retrieved_at = isoparse(retrieved_at)
			except Exception:
				retrieved_at = None
		if not is_expired(retrieved_at, domain, now, cfg):
			all_expired = False
		label = str(it.get("support_label", "")).lower()
		if label in ("supports", "entail"):
			has_support = True
	# Live search if everything is expired or we lack support
	return all_expired or (not has_support)


class Searcher(Protocol):
	def search(self, claim: str, entities: Optional[List[str]] = None, top_k: int = 5, days: Optional[int] = 90) -> List[Dict[str, Any]]: ...


def refresh_claim_if_needed(
	claim: str,
	cached: Dict[str, List[Dict[str, Any]]],
	searcher: Searcher,
	entities: Optional[List[str]] = None,
	broader_days: int = 365,
	now: Optional[datetime] = None,
	cfg: Optional[TTLConfig] = None,
) -> bool:
	"""Refresh cache for claim if policy says so.

	Returns True if refreshed (cache updated), else False.
	"""
	now = now or datetime.now(timezone.utc)
	cfg = cfg or TTLConfig()
	items = cached.get(claim, [])
	if should_live_search(claim, items, now=now, cfg=cfg):
		# Broaden time window to increase recall
		results = searcher.search(claim, entities=entities, top_k=5, days=broader_days)
		# Normalize cached format: dicts require at least domain, retrieved_at, support_label
		normalized: List[Dict[str, Any]] = []
		for r in results:
			# r can be RawEvidence-like or dict
			if hasattr(r, "domain"):
				domain = r.domain
				retrieved_at = getattr(r, "retrieved_at", now)
				support_label = getattr(r, "support_label", "")
				url = getattr(r, "url", "")
				title = getattr(r, "title", "")
			elif isinstance(r, dict):
				domain = r.get("domain", "")
				retrieved_at = r.get("retrieved_at", now)
				support_label = r.get("support_label", "")
				url = r.get("url", "")
				title = r.get("title", "")
			else:
				continue
			normalized.append({
				"domain": domain,
				"retrieved_at": retrieved_at,
				"support_label": support_label,
				"url": url,
				"title": title,
			})
		cached[claim] = normalized
		return True
	return False
