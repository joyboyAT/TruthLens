"""
Tests for caching policy: TTL & refresh.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
import pytest

# Ensure package import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cache.policy import TTLConfig, is_expired, ttl_days_for_domain, should_live_search, refresh_claim_if_needed


class DummySearcher:
	def __init__(self, url: str = "https://reuters.com/x"):
		self.calls = 0
		self.url = url
	def search(self, claim, entities=None, top_k=5, days=90):
		self.calls += 1
		return [{
			"domain": "reuters.com",
			"retrieved_at": datetime.now(timezone.utc),
			"support_label": "supports",
			"url": self.url,
			"title": f"New: {claim}",
		}]


def test_expired_news_triggers_refresh_and_updates_cache():
	now = datetime.now(timezone.utc)
	cfg = TTLConfig(news_days=3)

	claim = "Vaccines reduce severe illness"
	old_item = {
		"domain": "reuters.com",
		"retrieved_at": now - timedelta(days=10),
		"support_label": "supports",
		"url": "https://reuters.com/old",
		"title": "Old news",
	}
	cache = {claim: [old_item]}
	searcher = DummySearcher(url="https://reuters.com/new")

	# Ensure policy detects expiration
	assert is_expired(old_item["retrieved_at"], old_item["domain"], now=now, cfg=cfg)
	# Should live search due to expiration
	assert should_live_search(claim, cache[claim], now=now, cfg=cfg) is True

	# Refresh
	refreshed = refresh_claim_if_needed(claim, cache, searcher, entities=None, broader_days=365, now=now, cfg=cfg)
	assert refreshed is True
	assert searcher.calls == 1
	# Cache should be updated with new URL and recent retrieved_at
	assert cache[claim][0]["url"] == "https://reuters.com/new"
	assert (now - cache[claim][0]["retrieved_at"]).total_seconds() <= 86400  # within a day
