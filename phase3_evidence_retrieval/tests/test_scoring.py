"""
Tests for evidence scoring (source trust, freshness decay, relevance combination).
"""

import os
import sys
from datetime import datetime, timedelta, timezone
import pytest

# Ensure package import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scoring.evidence_score import (
	ScoreWeights,
	FreshnessConfig,
	source_score,
	freshness_score,
	normalize_relevance,
	final_score,
	score_evidence,
)
from schemas.evidence import RawEvidence, SourceType, Language, EvidenceMetadata


def _mk_ev(domain: str, published_days_ago: int, title: str) -> RawEvidence:
	pub = datetime.now(timezone.utc) - timedelta(days=published_days_ago)
	return RawEvidence(
		id=f"ev_{domain}_{published_days_ago}",
		source_type=SourceType.NEWS,
		url=f"https://{domain}/x",
		domain=domain,
		title=title,
		published_at=pub,
		retrieved_at=datetime.now(timezone.utc),
		language=Language.ENGLISH,
		raw_html=None,
		raw_text=None,
		snippet=title,
		full_text=title,
		text_hash=None,
		metadata=EvidenceMetadata(),
		source_config={},
		retrieval_metadata={},
	)


def test_scoring_sorting_changes_with_dates_and_domains():
	w = ScoreWeights(relevance=0.5, freshness=0.3, source=0.2)
	fcfg = FreshnessConfig(half_life_days=30.0, min_score=0.1)

	# Same relevance, different domains and publish dates
	ev1 = _mk_ev("reuters.com", 5, "Article about vaccines")  # top news, fresh
	ev2 = _mk_ev("pib.gov.in", 40, "Government release on vaccines")  # gov, older
	ev3 = _mk_ev("randomblog.com", 2, "Random blog post")  # other, very fresh

	rel_raw = 2.0  # unbounded logit-like; will go through logistic to ~0.88

	s1 = score_evidence(ev1, rel_raw, w=w, fcfg=fcfg)
	s2 = score_evidence(ev2, rel_raw, w=w, fcfg=fcfg)
	s3 = score_evidence(ev3, rel_raw, w=w, fcfg=fcfg)

	rows = [("reuters", s1["final"], s1), ("pib", s2["final"], s2), ("blog", s3["final"], s3)]
	print("Scores:")
	for name, fs, comp in rows:
		print(name, fs, comp)

	# With these weights, fresh top news vs older gov could be close; ensure gov score isn't always worse
	assert s2["source"] > s1["source"]  # gov > news
	# Ensure freshness differentiates recent blog from older gov
	assert s3["freshness"] > s2["freshness"]

	# Tweak publish date to make gov fresh and ensure ordering changes
	ev2b = _mk_ev("pib.gov.in", 1, "Government release fresh")
	s2b = score_evidence(ev2b, rel_raw, w=w, fcfg=fcfg)
	assert s2b["freshness"] > s1["freshness"]  # now fresher than reuters
	# Final score for fresh gov should beat reuters
	assert s2b["final"] > s1["final"]
