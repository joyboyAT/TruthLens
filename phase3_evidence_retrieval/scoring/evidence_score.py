"""
Evidence scoring for TruthLens Phase 3.

- source_score: gov/health > fact-check > top news > others
- freshness_score: newer is better with exponential decay after N days
- relevance_score: from reranker (expected to be unbounded logit or similarity); normalized here
- final_score = w1*relevance + w2*freshness + w3*source

Weights (default): relevance 0.5, freshness 0.3, source 0.2
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, log
from typing import Dict, Optional

# Schemas
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from schemas.evidence import RawEvidence, EvidenceChunk  # type: ignore


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

TOP_NEWS_DOMAINS = (
	"reuters.com",
	"apnews.com",
	"ap.org",
	"bbc.com",
	"nytimes.com",
	"washingtonpost.com",
	"wsj.com",
	"theguardian.com",
)


@dataclass
class ScoreWeights:
	relevance: float = 0.5
	freshness: float = 0.3
	source: float = 0.2


@dataclass
class FreshnessConfig:
	half_life_days: float = 30.0
	min_score: float = 0.1


def _domain_contains(domain: str, needles: tuple[str, ...]) -> bool:
	d = (domain or "").lower()
	return any(n in d for n in needles)


def source_score(domain: str) -> float:
	"""Return source trust score in [0,1]."""
	if _domain_contains(domain, GOV_HEALTH_DOMAINS):
		return 0.9
	if _domain_contains(domain, FACT_CHECK_DOMAINS):
		return 0.8
	if _domain_contains(domain, TOP_NEWS_DOMAINS):
		return 0.7
	return 0.4


def freshness_score(published_at: Optional[datetime], now: Optional[datetime] = None, cfg: Optional[FreshnessConfig] = None) -> float:
	"""Exponential decay after half-life.

	If published_at is None, return a small baseline.
	"""
	cfg = cfg or FreshnessConfig()
	if published_at is None:
		return cfg.min_score
	now = now or datetime.now(timezone.utc)
	if published_at.tzinfo is None:
		published_at = published_at.replace(tzinfo=timezone.utc)
	days = max(0.0, (now - published_at).total_seconds() / 86400.0)
	# exp decay with half-life
	lam = log(2.0) / cfg.half_life_days
	score = exp(-lam * days)
	return max(cfg.min_score, min(1.0, score))


def normalize_relevance(raw_score: float) -> float:
	"""Normalize relevance score to [0,1].

	- If input looks like a cosine similarity in [-1,1], map to [0,1].
	- If input looks like a logit (unbounded), map via 1/(1+exp(-x)).
	- Otherwise, clamp to [0,1].
	"""
	# Heuristic: treat values <=1.5 as similarity
	if raw_score <= 1.5 and raw_score >= -1.5:
		return max(0.0, min(1.0, 0.5 * (raw_score + 1.0)))
	# logistic mapping
	try:
		from math import exp as _exp
		p = 1.0 / (1.0 + _exp(-raw_score))
		return max(0.0, min(1.0, p))
	except Exception:
		return max(0.0, min(1.0, raw_score))


def final_score(relevance: float, freshness: float, source: float, w: Optional[ScoreWeights] = None) -> float:
	w = w or ScoreWeights()
	return (
		w.relevance * relevance +
		w.freshness * freshness +
		w.source * source
	)


def score_evidence(ev: RawEvidence, relevance_raw: float, w: Optional[ScoreWeights] = None, fcfg: Optional[FreshnessConfig] = None) -> Dict[str, float]:
	rel = normalize_relevance(relevance_raw)
	fresh = freshness_score(ev.published_at, cfg=fcfg)
	src = source_score(ev.domain)
	fs = final_score(rel, fresh, src, w)
	return {"relevance": rel, "freshness": fresh, "source": src, "final": fs}


def score_chunk(chunk: EvidenceChunk, evidence_domain: str, evidence_published_at: Optional[datetime], relevance_raw: float, w: Optional[ScoreWeights] = None, fcfg: Optional[FreshnessConfig] = None) -> Dict[str, float]:
	rel = normalize_relevance(relevance_raw)
	fresh = freshness_score(evidence_published_at, cfg=fcfg)
	src = source_score(evidence_domain)
	fs = final_score(rel, fresh, src, w)
	return {"relevance": rel, "freshness": fresh, "source": src, "final": fs}
