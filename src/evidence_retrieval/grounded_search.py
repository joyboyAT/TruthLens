"""
Grounded search for TruthLens Phase 3.

- Pluggable web search clients (Serper, SerpAPI, Bing)
- Query building with site filters from trusted domains
- Time window filtering (e.g., last 30/90 days)
- Fetch pages → extract clean text with trafilatura
- Capture published_at via meta tags, JSON-LD, OpenGraph; fallback to first seen
"""

from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import trafilatura
from bs4 import BeautifulSoup
from dateparser import parse as parse_date
from tldextract import extract as tld_extract

# Reuse schemas
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from schemas.evidence import Evidence, SourceType  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class SearchResult:
	url: str
	title: str
	snippet: str
	score: Optional[float] = None


class SearchClientBase:
	"""Interface for web search clients."""

	def search(self, query: str, num_results: int = 10, days: Optional[int] = None) -> List[SearchResult]:
		raise NotImplementedError


class SerperClient(SearchClientBase):
	"""Serper.dev Google Search API client."""

	ENDPOINT = "https://google.serper.dev/search"

	def __init__(self, api_key: str):
		self.api_key = api_key

	def search(self, query: str, num_results: int = 10, days: Optional[int] = None) -> List[SearchResult]:
		headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
		payload: Dict[str, Any] = {"q": query, "num": num_results}
		# Recency via tbs (qdr:d[n], w[n], m[n])
		if days:
			if days <= 7:
				payload["tbs"] = f"qdr:w{max(1, days // 7)}"
			elif days <= 31:
				payload["tbs"] = f"qdr:m1"
			elif days <= 93:
				payload["tbs"] = f"qdr:m3"
			else:
				payload["tbs"] = f"qdr:y1"
		resp = requests.post(self.ENDPOINT, headers=headers, json=payload, timeout=30)
		resp.raise_for_status()
		data = resp.json()
		results: List[SearchResult] = []
		for item in data.get("organic", [])[:num_results]:
			results.append(
				SearchResult(
					url=item.get("link", ""),
					title=item.get("title", ""),
					snippet=item.get("snippet", ""),
					score=None,
				)
			)
		return results


class BingClient(SearchClientBase):
	"""Bing Web Search API client."""

	ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

	def __init__(self, api_key: str):
		self.api_key = api_key

	def search(self, query: str, num_results: int = 10, days: Optional[int] = None) -> List[SearchResult]:
		headers = {"Ocp-Apim-Subscription-Key": self.api_key}
		params: Dict[str, Any] = {"q": query, "count": num_results, "textDecorations": False}
		if days:
			# freshness: Day, Week, Month
			if days <= 7:
				params["freshness"] = "Week"
			elif days <= 31:
				params["freshness"] = "Month"
			else:
				params["freshness"] = "Month"
		resp = requests.get(self.ENDPOINT, headers=headers, params=params, timeout=30)
		resp.raise_for_status()
		data = resp.json()
		web_pages = data.get("webPages", {}).get("value", [])
		results: List[SearchResult] = []
		for item in web_pages[:num_results]:
			results.append(
				SearchResult(url=item.get("url", ""), title=item.get("name", ""), snippet=item.get("snippet", ""))
			)
		return results


class SerpAPIClient(SearchClientBase):
	"""SerpAPI client (Google Search)."""

	ENDPOINT = "https://serpapi.com/search.json"

	def __init__(self, api_key: str):
		self.api_key = api_key

	def search(self, query: str, num_results: int = 10, days: Optional[int] = None) -> List[SearchResult]:
		params: Dict[str, Any] = {"engine": "google", "q": query, "num": num_results, "api_key": self.api_key}
		if days:
			if days <= 7:
				params["tbs"] = "qdr:w1"
			elif days <= 31:
				params["tbs"] = "qdr:m1"
			elif days <= 93:
				params["tbs"] = "qdr:m3"
			else:
				params["tbs"] = "qdr:y1"
		resp = requests.get(self.ENDPOINT, params=params, timeout=30)
		resp.raise_for_status()
		data = resp.json()
		results: List[SearchResult] = []
		for item in data.get("organic_results", [])[:num_results]:
			results.append(
				SearchResult(
					url=item.get("link", ""), title=item.get("title", ""), snippet=item.get("snippet", ""), score=None
				)
			)
		return results


DEFAULT_TRUSTED_DOMAINS: List[str] = [
	"pib.gov.in",
	"who.int",
	"cdc.gov",
	"nih.gov",
	"fda.gov",
	"gov.in",
	"gov.uk",
	"europa.eu",
	"reuters.com",
	"apnews.com",
	"ap.org",
	"bbc.com",
	"wikipedia.org",
]


def load_trusted_domains(config_dir: Optional[str] = None) -> List[str]:
	"""Load trusted domains from config, fallback to default hard-coded list."""
	try:
		base_dir = Path(config_dir) if config_dir else Path(__file__).resolve().parents[1] / "config"
		path = base_dir / "domains_whitelist.json"
		if path.exists():
			with open(path, "r", encoding="utf-8") as f:
				data = json.load(f)
			all_domains: List[str] = []
			for _, domains in data.items():
				all_domains.extend(domains)
			# Keep only a curated subset for grounded search
			preferred = [
				"pib.gov.in",
				"who.int",
				"cdc.gov",
				"nih.gov",
				"fda.gov",
				"gov.in",
				"gov.uk",
				"europa.eu",
				"reuters.com",
				"apnews.com",
				"bbc.com",
				"wikipedia.org",
			]
			# Add if present in whitelist
			return list(dict.fromkeys([d for d in preferred if d in all_domains] + DEFAULT_TRUSTED_DOMAINS))
	except Exception as e:
		logger.warning(f"Failed to load domains_whitelist.json: {e}")
	return DEFAULT_TRUSTED_DOMAINS


def build_site_filter(domains: List[str]) -> str:
	# Example: (site:who.int OR site:pib.gov.in OR ...)
	parts = [f"site:{d}" for d in domains]
	return " (" + " OR ".join(parts) + ")"


def extract_domain(url: str) -> str:
	parts = tld_extract(url)
	if parts.subdomain:
		return f"{parts.subdomain}.{parts.domain}.{parts.suffix}"
	return f"{parts.domain}.{parts.suffix}" if parts.suffix else parts.domain


def classify_source_type(domain: str) -> SourceType:
	domain_l = domain.lower()
	if "wikipedia.org" in domain_l:
		return SourceType.WIKIPEDIA
	if any(g in domain_l for g in [".gov", "who.int", "cdc.gov", "nih.gov", "fda.gov", "europa.eu", "gov.in", "gov.uk"]):
		return SourceType.GOVERNMENT
	return SourceType.NEWS


def detect_language_simple(text: str) -> str:
	"""Lightweight language detection with safe enum mapping (defaults to English)."""
	try:
		from langdetect import detect  # optional but present in requirements
		code = (detect(text or "") or "en").lower()
		if code == "en":
			return "en"
		if code == "hi":
			return "hi"
		return "en"
	except Exception:
		return "en"


def parse_published_at(html: str, url: str) -> Optional[datetime]:
	soup = BeautifulSoup(html, "html.parser")
	# Common meta tags
	selectors = [
		('meta', {'property': 'article:published_time'}),
		('meta', {'property': 'og:published_time'}),
		('meta', {'name': 'publish_date'}),
		('meta', {'name': 'pubdate'}),
		('meta', {'itemprop': 'datePublished'}),
		('time', {'datetime': True}),
		('meta', {'name': 'DC.date.issued'}),
	]
	for tag, attrs in selectors:
		el = soup.find(tag, attrs=attrs)
		if el:
			val = el.get('content') or el.get('datetime') or el.get_text(strip=True)
			if val:
				try:
					dt = parse_date(val)
					if dt:
						return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
				except Exception:
					pass
	# JSON-LD
	for script in soup.find_all('script', {'type': 'application/ld+json'}):
		try:
			data = json.loads(script.string or "{}")
			if isinstance(data, dict):
				date_str = data.get('datePublished') or data.get('dateCreated') or data.get('uploadDate')
				if date_str:
					dt = parse_date(date_str)
					if dt:
						return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
			elif isinstance(data, list):
				for obj in data:
					if isinstance(obj, dict):
						date_str = obj.get('datePublished') or obj.get('dateCreated') or obj.get('uploadDate')
						if date_str:
							dt = parse_date(date_str)
							if dt:
								return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
		except Exception:
			continue
	return None


def fetch_and_extract(url: str) -> Tuple[str, str, Optional[datetime]]:
	"""Return (clean_text, snippet256, published_at)."""
	resp = requests.get(url, timeout=30)
	resp.raise_for_status()
	html = resp.text
	text = trafilatura.extract(html) or ""
	text = (text or "").strip()
	snippet = text[:256]
	published = parse_published_at(html, url)
	return text, snippet, published


class GroundedSearcher:
	"""Grounded search orchestrator with domain filtering and recency."""

	def __init__(self, client: SearchClientBase, trusted_domains: Optional[List[str]] = None):
		self.client = client
		self.trusted_domains = trusted_domains or load_trusted_domains()

	def build_query(self, claim_text: str, entities: Optional[List[str]] = None, days: Optional[int] = 90) -> str:
		entities = entities or []
		terms = [claim_text] + entities
		q = " ".join([t for t in terms if t])
		sites = build_site_filter(self.trusted_domains)
		return f"{q} {sites}"

	def search(self, claim_text: str, entities: Optional[List[str]] = None, top_k: int = 5, days: Optional[int] = 90) -> List[Evidence]:
		query = self.build_query(claim_text, entities, days)
		results = self.client.search(query, num_results=top_k, days=days)
		normalized: List[Evidence] = []
		for r in results:
			if not r.url:
				continue
			try:
				text, snippet, published = fetch_and_extract(r.url)
				domain = extract_domain(r.url)
				source_type = classify_source_type(domain)
				lang = detect_language_simple(text[:500])
				
				evidence = Evidence(
					id=f"grounded_{hash((r.url, r.title))}",
					claim_id="",  # Will be set when linked to a claim
					source_type=source_type,
					url=r.url,
					domain=domain,
					title=r.title or r.url,
					published_at=published,
					language=lang,
					snippet=snippet,
					full_text=text,
					metadata={
						"query": query, 
						"client": self.client.__class__.__name__,
						"score": r.score
					}
				)
				normalized.append(evidence)
			except Exception as e:
				logger.warning(f"Failed to fetch/extract {r.url}: {e}")
		return normalized


def make_default_client() -> Optional[SearchClientBase]:
	# Prefer Serper → Bing → SerpAPI
	serper_key = os.getenv("SERPER_API_KEY")
	if not serper_key:
		serper_key = os.getenv("GOOGLE_SERPER_API_KEY")
	if serper_key:
		return SerperClient(serper_key)
	bing_key = os.getenv("BING_API_KEY") or os.getenv("AZURE_BING_SEARCH_KEY")
	if bing_key:
		return BingClient(bing_key)
	serp_key = os.getenv("SERPAPI_API_KEY")
	if serp_key:
		return SerpAPIClient(serp_key)
	return None


if __name__ == "__main__":
	client = make_default_client()
	if not client:
		print("No search API key configured (SERPER_API_KEY / BING_API_KEY / SERPAPI_API_KEY). Exiting.")
		exit(0)
	searcher = GroundedSearcher(client)
	claims = [
		"Climate change is caused by human activities",
		"Vaccines cause autism",
		"WHO guidelines for Covid-19 boosters",
		"Indian GDP growth last quarter",
		"New malaria vaccine approval WHO",
	]
	for claim in claims:
		print(f"\nClaim: {claim}")
		evidence_list = searcher.search(claim, entities=None, top_k=5, days=90)
		for i, ev in enumerate(evidence_list, 1):
			date_str = ev.published_at.isoformat() if ev.published_at else "N/A"
			print(f"{i}. [{ev.source_type}] {ev.title[:90]}")
			print(f"   URL: {ev.url}")
			print(f"   Published: {date_str}")
			print(f"   Snippet: {ev.snippet[:120]}")
