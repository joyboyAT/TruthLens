"""
Wikipedia connector for TruthLens Phase 3 (background grounding).

- Fetch summary + relevant sections for claim entities via MediaWiki API
- Keep page revision id for reproducibility
- Normalize as low-priority background evidence
"""

from __future__ import annotations

import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

# Reuse schemas
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from schemas.evidence import RawEvidence, EvidenceMetadata, SourceType, Language  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "TruthLens/1.0 (Wikipedia Connector; background grounding)"


def _clean_text(html_or_text: str) -> str:
	# If HTML, strip tags
	if "<" in html_or_text and ">" in html_or_text:
		soup = BeautifulSoup(html_or_text, "html.parser")
		for el in soup(["sup", "table", "style", "script"]):
			el.decompose()
		text = soup.get_text(separator=" ", strip=True)
	else:
		text = html_or_text
	# Collapse whitespace
	text = re.sub(r"\s+", " ", text or "").strip()
	return text


def fetch_page_basic(title: str, lang: str = "en") -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[str], Optional[str]]:
	"""Fetch page summary + revision id + fullurl.

	Returns: (page_dict, pageid, revid, fullurl)
	"""
	params = {
		"action": "query",
		"format": "json",
		"prop": "extracts|revisions|info",
		"explaintext": 1,
		"exintro": 1,
		"rvprop": "ids|timestamp",
		"inprop": "url",
		"titles": title,
	}
	headers = {"User-Agent": USER_AGENT}
	resp = requests.get(WIKI_API.format(lang=lang), params=params, headers=headers, timeout=30)
	resp.raise_for_status()
	data = resp.json()
	pages = data.get("query", {}).get("pages", {})
	if not pages:
		return None, None, None, None
	# Take first page
	page = next(iter(pages.values()))
	if page.get("missing"):
		return None, None, None, None
	pageid = page.get("pageid")
	revisions = page.get("revisions", [])
	revid = revisions[0].get("revid") if revisions else None
	fullurl = page.get("fullurl")
	return page, pageid, revid, fullurl


def fetch_sections_html(title: str, lang: str = "en") -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
	"""Fetch section list and a map of section_index -> HTML for selected sections."""
	headers = {"User-Agent": USER_AGENT}
	# Get sections
	params_sections = {
		"action": "parse",
		"format": "json",
		"prop": "sections",
		"page": title,
	}
	resp = requests.get(WIKI_API.format(lang=lang), params=params_sections, headers=headers, timeout=30)
	resp.raise_for_status()
	sections = resp.json().get("parse", {}).get("sections", [])
	# Choose first few relevant sections
	selected: List[int] = []
	for s in sections:
		name = (s.get("line") or "").lower()
		index = int(s.get("index")) if s.get("index") else None
		if not index:
			continue
		# Skip administrative sections
		if any(skip in name for skip in ["see also", "references", "external links", "further reading", "notes", "bibliography", "sources"]):
			continue
		selected.append(index)
		if len(selected) >= 2:
			break
	# Fetch HTML for selected sections
	html_map: Dict[int, str] = {}
	for idx in selected:
		params_html = {
			"action": "parse",
			"format": "json",
			"prop": "text",
			"page": title,
			"section": idx,
		}
		res = requests.get(WIKI_API.format(lang=lang), params=params_html, headers=headers, timeout=30)
		res.raise_for_status()
		text_html = res.json().get("parse", {}).get("text", {}).get("*", "")
		html_map[idx] = text_html
	return sections, html_map


def build_background_text(summary: str, sections_html: Dict[int, str]) -> str:
	parts: List[str] = []
	if summary:
		parts.append(summary)
	for idx in sorted(sections_html.keys()):
		clean = _clean_text(sections_html[idx])
		if clean:
			parts.append(clean)
	return "\n\n".join(parts)


class WikipediaConnector:
	"""Fetch background grounding evidence from Wikipedia."""

	def __init__(self, lang: str = "en") -> None:
		self.lang = lang

	def fetch_background(self, entity: str) -> Optional[RawEvidence]:
		page, pageid, revid, fullurl = fetch_page_basic(entity, lang=self.lang)
		if not page:
			logger.info(f"Wikipedia page not found for entity: {entity}")
			return None
		summary = _clean_text(page.get("extract") or "")
		sections, html_map = fetch_sections_html(page.get("title") or entity, lang=self.lang)
		full_text = build_background_text(summary, html_map)
		if not full_text:
			full_text = summary
		# Metadata
		published = None  # Wikipedia pages are living; we carry revision info instead
		domain = f"{self.lang}.wikipedia.org"
		metadata = EvidenceMetadata(
			author=None,
			publication_date=published,
			tags=["wikipedia", "background"],
			categories=["background", "wikipedia"],
			verification_status="background",
			custom_fields={
				"wikipedia_pageid": pageid,
				"wikipedia_revid": revid,
				"wikipedia_fullurl": fullurl,
			}
		)
		# Build RawEvidence
		return RawEvidence(
			id=f"wiki_{revid or pageid}_{hash(entity)}",
			claim_id=None,
			source_type=SourceType.WIKIPEDIA,
			url=fullurl or f"https://{domain}/wiki/{(page.get('title') or entity).replace(' ', '_')}",
			domain=domain,
			title=page.get("title") or entity,
			published_at=None,
			retrieved_at=datetime.now(timezone.utc),
			language=Language.ENGLISH if self.lang == "en" else Language.ENGLISH,
			raw_html=None,
			raw_text=None,
			snippet=(summary or full_text)[:256],
			full_text=full_text or summary,
			text_hash=None,
			metadata=metadata,
			source_config={"connector": "Wikipedia", "lang": self.lang},
			retrieval_metadata={"priority": "low"},
		)

	def fetch_background_for_entities(self, entities: List[str]) -> List[RawEvidence]:
		results: List[RawEvidence] = []
		for e in entities:
			try:
				item = self.fetch_background(e)
				if item:
					results.append(item)
			except Exception as ex:
				logger.warning(f"Failed to fetch background for {e}: {ex}")
		return results


if __name__ == "__main__":
	connector = WikipediaConnector(lang="en")
	entities = [
		"World Health Organization",
		"Coronavirus disease 2019",
		"Renewable energy",
		"Inflation (economics)",
		"Narendra Modi",
	]
	items = connector.fetch_background_for_entities(entities)
	for i, ev in enumerate(items, 1):
		rev = ev.metadata.custom_fields.get("wikipedia_revid") if ev.metadata and ev.metadata.custom_fields else None
		print(f"{i}. {ev.title} (rev_id={rev})")
		print(f"   URL: {ev.url}")
		print(f"   Snippet: {ev.snippet}")
