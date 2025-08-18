"""
Tests for grounded search (News/Gov/Health + recency).

- Uses pluggable web search client (Serper/Bing/SerpAPI) via env keys
- Builds queries with trusted domain site filters
- Applies time window
- Fetches pages and extracts text + published_at

Skips if no API key is configured in environment.
"""

import os
import sys
import pytest
from datetime import timezone

# Ensure package import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.grounded_search import make_default_client, GroundedSearcher  # type: ignore


CLAIMS = [
	"Climate change is caused by human activities",
	"WHO guidance on COVID-19 boosters",
	"Government inflation data India site:gov.in",
	"Reuters report on interest rates",
	"Wikipedia page for renewable energy policy",
]


@pytest.mark.asyncio
async def test_grounded_search_top5_and_dates(capsys):
	client = make_default_client()
	if client is None:
		pytest.skip("No search API key configured (SERPER_API_KEY / BING_API_KEY / SERPAPI_API_KEY)")
	searcher = GroundedSearcher(client)

	total = 0
	with_dates = 0

	for claim in CLAIMS:
		results = searcher.search(claim, entities=None, top_k=5, days=90)
		print(f"\nClaim: {claim}")
		for i, ev in enumerate(results, 1):
			date_str = ev.published_at.isoformat() if ev.published_at else "N/A"
			print(f"{i}. {ev.url}")
			print(f"   Published: {date_str}")
			print(f"   Snippet: {(ev.snippet or '')[:256]}")
			print()
			total += 1
			if ev.published_at is not None:
				with_dates += 1

	# Require at least some results to make the assertion meaningful
	if total >= 5:
		assert with_dates / total >= 0.8, f"published_at parsed for only {with_dates}/{total} results"
	else:
		pytest.skip("Too few results returned to assert date coverage")
