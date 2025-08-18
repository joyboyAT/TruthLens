"""
Fact-checking source connectors for TruthLens Phase 3.

This module implements RSS/HTML scrapers for major fact-checking organizations
including PIB Fact Check, PolitiFact, BOOM Live, and AltNews.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urljoin

import requests
import trafilatura
from dateparser import parse as parse_date
from tldextract import extract
from fuzzywuzzy import fuzz
from bs4 import BeautifulSoup
import feedparser

from ..schemas.evidence import RawEvidence, SourceType, Language, EvidenceMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactCheckConnector:
    """Base class for fact-checking source connectors."""
    
    def __init__(self, base_url: str, name: str, rate_limit: int = 10):
        self.base_url = base_url.rstrip('/')
        self.name = name
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TruthLens/1.0 (Fact-Checking Bot)'
        })
    
    def _rate_limit(self):
        """Implement rate limiting for robots.txt compliance."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.rate_limit
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        extracted = extract(url)
        return f"{extracted.domain}.{extracted.suffix}"
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        return " ".join(text.split())
    
    def _detect_language(self, text: str) -> Language:
        """Detect language from text content."""
        # Simple language detection
        if any(char in text for char in 'अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञड़ढ़'):
            return Language.HINDI
        elif any(char in text for char in 'அஆஇஈஉஊஎஏஐஒஓகஙசஜஞடணதநனபமயரலவஶஷஸஹ'):
            return Language.TAMIL
        elif any(char in text for char in 'अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'):
            return Language.MARATHI
        else:
            return Language.ENGLISH
    
    async def fetch_rss_feed(self, rss_url: str) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed."""
        try:
            self._rate_limit()
            response = self.session.get(rss_url, timeout=30)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            items = []
            
            for entry in feed.entries:
                item = {
                    'title': entry.get('title', ''),
                    'url': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'summary': entry.get('summary', ''),
                    'author': entry.get('author', '')
                }
                items.append(item)
            
            logger.info(f"Fetched {len(items)} items from {self.name} RSS feed")
            return items
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed from {self.name}: {e}")
            return []
    
    async def scrape_article(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape individual article content."""
        try:
            self._rate_limit()
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Try trafilatura first
            extracted_text = trafilatura.extract(response.text)
            
            if not extracted_text:
                # Fallback to BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                extracted_text = soup.get_text()
            
            # Clean the extracted text
            extracted_text = self._clean_text(extracted_text)
            
            # Extract metadata
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find publication date
            published_date = None
            date_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="publish_date"]',
                'time[datetime]',
                '.published-date'
            ]
            
            for selector in date_selectors:
                element = soup.select_one(selector)
                if element:
                    date_str = element.get('content') or element.get('datetime') or element.get_text()
                    try:
                        published_date = parse_date(date_str)
                        break
                    except:
                        continue
            
            # Extract title and snippet
            title = soup.find('title')
            title_text = title.get_text() if title else ''
            
            snippet = ''
            if extracted_text:
                paragraphs = extracted_text.split('\n\n')
                snippet = paragraphs[0][:200] + '...' if len(paragraphs[0]) > 200 else paragraphs[0]
            
            return {
                'title': self._clean_text(title_text),
                'url': url,
                'published': published_date,
                'summary': self._clean_text(snippet),
                'full_text': extracted_text,
                'domain': self._extract_domain(url)
            }
            
        except Exception as e:
            logger.error(f"Error scraping article {url}: {e}")
            return None
    
    def normalize_to_evidence(self, item: Dict[str, Any]) -> RawEvidence:
        """Normalize scraped item to RawEvidence format."""
        import hashlib
        item_id = hashlib.md5(f"{item['url']}_{item['title']}".encode()).hexdigest()
        
        text_for_lang = item.get('summary', '') or item.get('title', '')
        language = self._detect_language(text_for_lang)
        
        metadata = EvidenceMetadata(
            author=item.get('author', ''),
            publication_date=item.get('published'),
            categories=['fact_check'],
            verification_status='verified'
        )
        
        return RawEvidence(
            id=f"{self.name.lower()}_{item_id}",
            source_type=SourceType.FACT_CHECK,
            url=item['url'],
            domain=item.get('domain', self._extract_domain(item['url'])),
            title=item['title'],
            published_at=item.get('published'),
            language=language,
            snippet=item.get('summary', ''),
            full_text=item.get('full_text', ''),
            metadata=metadata,
            source_config={'connector': self.name}
        )
    
    async def search_claims(self, claims: List[str], max_results: int = 10) -> List[Tuple[str, RawEvidence, float]]:
        """Search for claims in fact-check articles."""
        raise NotImplementedError("Subclasses must implement search_claims method")


class PIBFactCheckConnector(FactCheckConnector):
    """Connector for PIB Fact Check."""
    
    def __init__(self):
        super().__init__(
            base_url="https://pib.gov.in",
            name="PIB Fact Check",
            rate_limit=5
        )
        self.rss_url = "https://pib.gov.in/rss/fact-check.xml"
    
    async def fetch_fact_checks(self) -> List[RawEvidence]:
        """Fetch fact-check articles from PIB."""
        items = await self.fetch_rss_feed(self.rss_url)
        evidence_list = []
        
        for item in items:
            article_data = await self.scrape_article(item['url'])
            if article_data:
                merged_item = {**item, **article_data}
                evidence = self.normalize_to_evidence(merged_item)
                evidence_list.append(evidence)
        
        return evidence_list
    
    async def search_claims(self, claims: List[str], max_results: int = 10) -> List[Tuple[str, RawEvidence, float]]:
        """Search for claims in PIB fact-check articles."""
        fact_checks = await self.fetch_fact_checks()
        results = []
        
        for claim in claims:
            claim_matches = []
            
            for evidence in fact_checks:
                title_score = fuzz.partial_ratio(claim.lower(), evidence.title.lower())
                snippet_score = fuzz.partial_ratio(claim.lower(), evidence.snippet.lower())
                max_score = max(title_score, snippet_score)
                
                if max_score > 50:
                    claim_matches.append((evidence, max_score))
            
            claim_matches.sort(key=lambda x: x[1], reverse=True)
            top_matches = claim_matches[:max_results]
            
            for evidence, score in top_matches:
                results.append((claim, evidence, score))
        
        return results


class PolitiFactConnector(FactCheckConnector):
    """Connector for PolitiFact."""
    
    def __init__(self):
        super().__init__(
            base_url="https://www.politifact.com",
            name="PolitiFact",
            rate_limit=15
        )
        self.rss_url = "https://www.politifact.com/feeds/fact-check.xml"
    
    async def fetch_fact_checks(self) -> List[RawEvidence]:
        """Fetch fact-check articles from PolitiFact."""
        items = await self.fetch_rss_feed(self.rss_url)
        evidence_list = []
        
        for item in items:
            article_data = await self.scrape_article(item['url'])
            if article_data:
                merged_item = {**item, **article_data}
                evidence = self.normalize_to_evidence(merged_item)
                evidence_list.append(evidence)
        
        return evidence_list
    
    async def search_claims(self, claims: List[str], max_results: int = 10) -> List[Tuple[str, RawEvidence, float]]:
        """Search for claims in PolitiFact articles."""
        fact_checks = await self.fetch_fact_checks()
        results = []
        
        for claim in claims:
            claim_matches = []
            
            for evidence in fact_checks:
                title_score = fuzz.partial_ratio(claim.lower(), evidence.title.lower())
                snippet_score = fuzz.partial_ratio(claim.lower(), evidence.snippet.lower())
                max_score = max(title_score, snippet_score)
                
                if max_score > 50:
                    claim_matches.append((evidence, max_score))
            
            claim_matches.sort(key=lambda x: x[1], reverse=True)
            top_matches = claim_matches[:max_results]
            
            for evidence, score in top_matches:
                results.append((claim, evidence, score))
        
        return results


class BOOMLiveConnector(FactCheckConnector):
    """Connector for BOOM Live."""
    
    def __init__(self):
        super().__init__(
            base_url="https://www.boomlive.in",
            name="BOOM Live",
            rate_limit=10
        )
        self.rss_url = "https://www.boomlive.in/feed"
    
    async def fetch_fact_checks(self) -> List[RawEvidence]:
        """Fetch fact-check articles from BOOM Live."""
        items = await self.fetch_rss_feed(self.rss_url)
        evidence_list = []
        
        for item in items:
            if 'fact' in item['title'].lower() or 'fact' in item['summary'].lower():
                article_data = await self.scrape_article(item['url'])
                if article_data:
                    merged_item = {**item, **article_data}
                    evidence = self.normalize_to_evidence(merged_item)
                    evidence_list.append(evidence)
        
        return evidence_list
    
    async def search_claims(self, claims: List[str], max_results: int = 10) -> List[Tuple[str, RawEvidence, float]]:
        """Search for claims in BOOM Live articles."""
        fact_checks = await self.fetch_fact_checks()
        results = []
        
        for claim in claims:
            claim_matches = []
            
            for evidence in fact_checks:
                title_score = fuzz.partial_ratio(claim.lower(), evidence.title.lower())
                snippet_score = fuzz.partial_ratio(claim.lower(), evidence.snippet.lower())
                max_score = max(title_score, snippet_score)
                
                if max_score > 50:
                    claim_matches.append((evidence, max_score))
            
            claim_matches.sort(key=lambda x: x[1], reverse=True)
            top_matches = claim_matches[:max_results]
            
            for evidence, score in top_matches:
                results.append((claim, evidence, score))
        
        return results


class AltNewsConnector(FactCheckConnector):
    """Connector for AltNews."""
    
    def __init__(self):
        super().__init__(
            base_url="https://www.altnews.in",
            name="AltNews",
            rate_limit=8
        )
        self.rss_url = "https://www.altnews.in/feed"
    
    async def fetch_fact_checks(self) -> List[RawEvidence]:
        """Fetch fact-check articles from AltNews."""
        items = await self.fetch_rss_feed(self.rss_url)
        evidence_list = []
        
        for item in items:
            if 'fact' in item['title'].lower() or 'fact' in item['summary'].lower():
                article_data = await self.scrape_article(item['url'])
                if article_data:
                    merged_item = {**item, **article_data}
                    evidence = self.normalize_to_evidence(merged_item)
                    evidence_list.append(evidence)
        
        return evidence_list
    
    async def search_claims(self, claims: List[str], max_results: int = 10) -> List[Tuple[str, RawEvidence, float]]:
        """Search for claims in AltNews articles."""
        fact_checks = await self.fetch_fact_checks()
        results = []
        
        for claim in claims:
            claim_matches = []
            
            for evidence in fact_checks:
                title_score = fuzz.partial_ratio(claim.lower(), evidence.title.lower())
                snippet_score = fuzz.partial_ratio(claim.lower(), evidence.snippet.lower())
                max_score = max(title_score, snippet_score)
                
                if max_score > 50:
                    claim_matches.append((evidence, max_score))
            
            claim_matches.sort(key=lambda x: x[1], reverse=True)
            top_matches = claim_matches[:max_results]
            
            for evidence, score in top_matches:
                results.append((claim, evidence, score))
        
        return results


class FactCheckAggregator:
    """Aggregates results from multiple fact-check connectors."""
    
    def __init__(self):
        self.connectors = {
            'pib': PIBFactCheckConnector(),
            'politifact': PolitiFactConnector(),
            'boom': BOOMLiveConnector(),
            'altnews': AltNewsConnector()
        }
    
    async def search_all_sources(self, claims: List[str], max_results: int = 10) -> Dict[str, List[Tuple[str, RawEvidence, float]]]:
        """Search all fact-check sources for claims."""
        results = {}
        
        for source_name, connector in self.connectors.items():
            try:
                logger.info(f"Searching {source_name} for {len(claims)} claims...")
                source_results = await connector.search_claims(claims, max_results)
                results[source_name] = source_results
                logger.info(f"Found {len(source_results)} matches in {source_name}")
            except Exception as e:
                logger.error(f"Error searching {source_name}: {e}")
                results[source_name] = []
        
        return results


if __name__ == "__main__":
    async def main():
        print("Testing Fact-Check Connectors...")
        
        aggregator = FactCheckAggregator()
        test_claims = [
            "Climate change is caused by human activities",
            "Vaccines cause autism",
            "The Earth is flat"
        ]
        
        results = await aggregator.search_all_sources(test_claims, max_results=3)
        
        for source, source_results in results.items():
            print(f"\n{source.upper()} Results:")
            for claim, evidence, score in source_results:
                print(f"  Claim: {claim[:50]}...")
                print(f"  Evidence: {evidence.title[:50]}...")
                print(f"  Score: {score}")
                print(f"  URL: {evidence.url}")
                print()
    
    asyncio.run(main())
