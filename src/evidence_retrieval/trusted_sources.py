"""
Trusted Sources Database for TruthLens Phase 3
Implements integration with trusted sources: PIB, WHO, RBI, major news portals, and fact-checking sites
Enhanced with additional fact-checking sources: Snopes, PolitiFact, AltNews, BOOM Live
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import requests
import time

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Types of trusted sources."""
    PIB = "pib"  # Press Information Bureau
    WHO = "who"  # World Health Organization
    RBI = "rbi"  # Reserve Bank of India
    NEWS_PORTAL = "news_portal"
    FACT_CHECK = "fact_check"
    RESEARCH = "research"
    GOVERNMENT = "government"
    # New fact-checking sources
    SNOPES = "snopes"
    POLITIFACT = "politifact"
    ALTNEWS = "altnews"
    BOOM_LIVE = "boom_live"
    SCIENCE_FEEDBACK = "science_feedback"


@dataclass
class TrustedSource:
    """Represents a trusted source with metadata."""
    id: str
    name: str
    source_type: SourceType
    domain: str
    url: str
    description: str
    reliability_score: float
    last_updated: datetime
    is_active: bool = True
    api_key: Optional[str] = None
    rate_limit: Optional[int] = None  # requests per minute


@dataclass
class TrustedContent:
    """Represents content from a trusted source."""
    id: str
    source_id: str
    title: str
    url: str
    snippet: str
    full_text: str
    published_date: datetime
    last_updated: datetime
    tags: List[str]
    relevance_score: float = 0.0
    fact_check_rating: Optional[str] = None  # For fact-checking sources
    verdict: Optional[str] = None  # True, False, Mixed, Unproven


class TrustedSourcesDatabase:
    """
    Database of trusted sources and their content.
    
    Enhanced Features:
    - Curated list of trusted sources including fact-checking sites
    - Content caching and freshness tracking
    - Relevance scoring
    - API integration for real-time updates
    - Fact-checking source integration (Snopes, PolitiFact, AltNews, BOOM Live)
    """
    
    def __init__(self, data_dir: str = "data/trusted_sources"):
        """
        Initialize the trusted sources database.
        
        Args:
            data_dir: Directory to store trusted sources data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.sources_file = self.data_dir / "sources.json"
        self.content_file = self.data_dir / "content.json"
        self.cache_file = self.data_dir / "cache.json"
        
        # Initialize sources
        self.sources: Dict[str, TrustedSource] = {}
        self.content: Dict[str, TrustedContent] = {}
        self.cache: Dict[str, Any] = {}
        
        self._load_data()
        self._initialize_default_sources()
    
    def _load_data(self):
        """Load existing data from files."""
        try:
            if self.sources_file.exists():
                with open(self.sources_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for source_data in data.values():
                        source = TrustedSource(
                            id=source_data["id"],
                            name=source_data["name"],
                            source_type=SourceType(source_data["source_type"]),
                            domain=source_data["domain"],
                            url=source_data["url"],
                            description=source_data["description"],
                            reliability_score=source_data["reliability_score"],
                            last_updated=datetime.fromisoformat(source_data["last_updated"]),
                            is_active=source_data.get("is_active", True),
                            api_key=source_data.get("api_key"),
                            rate_limit=source_data.get("rate_limit")
                        )
                        self.sources[source.id] = source
            
            if self.content_file.exists():
                with open(self.content_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for content_data in data.values():
                        content = TrustedContent(
                            id=content_data["id"],
                            source_id=content_data["source_id"],
                            title=content_data["title"],
                            url=content_data["url"],
                            snippet=content_data["snippet"],
                            full_text=content_data["full_text"],
                            published_date=datetime.fromisoformat(content_data["published_date"]),
                            last_updated=datetime.fromisoformat(content_data["last_updated"]),
                            tags=content_data["tags"],
                            relevance_score=content_data.get("relevance_score", 0.0),
                            fact_check_rating=content_data.get("fact_check_rating"),
                            verdict=content_data.get("verdict")
                        )
                        self.content[content.id] = content
            
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading trusted sources data: {e}")
    
    def _initialize_default_sources(self):
        """Initialize default trusted sources including fact-checking sites."""
        default_sources = [
            # Government sources
            TrustedSource(
                id="pib",
                name="Press Information Bureau",
                source_type=SourceType.PIB,
                domain="pib.gov.in",
                url="https://pib.gov.in",
                description="Official press releases from Government of India",
                reliability_score=0.95,
                last_updated=datetime.now(),
                is_active=True
            ),
            TrustedSource(
                id="who",
                name="World Health Organization",
                source_type=SourceType.WHO,
                domain="who.int",
                url="https://www.who.int",
                description="International health organization",
                reliability_score=0.95,
                last_updated=datetime.now(),
                is_active=True
            ),
            TrustedSource(
                id="rbi",
                name="Reserve Bank of India",
                source_type=SourceType.RBI,
                domain="rbi.org.in",
                url="https://www.rbi.org.in",
                description="Central bank of India",
                reliability_score=0.95,
                last_updated=datetime.now(),
                is_active=True
            ),
            
            # Fact-checking sources
            TrustedSource(
                id="snopes",
                name="Snopes",
                source_type=SourceType.SNOPES,
                domain="snopes.com",
                url="https://www.snopes.com",
                description="Leading fact-checking website",
                reliability_score=0.90,
                last_updated=datetime.now(),
                is_active=True,
                rate_limit=60  # Conservative rate limit
            ),
            TrustedSource(
                id="politifact",
                name="PolitiFact",
                source_type=SourceType.POLITIFACT,
                domain="politifact.com",
                url="https://www.politifact.com",
                description="Fact-checking website by Poynter Institute",
                reliability_score=0.90,
                last_updated=datetime.now(),
                is_active=True,
                rate_limit=60
            ),
            TrustedSource(
                id="altnews",
                name="Alt News",
                source_type=SourceType.ALTNEWS,
                domain="altnews.in",
                url="https://www.altnews.in",
                description="Indian fact-checking website",
                reliability_score=0.85,
                last_updated=datetime.now(),
                is_active=True,
                rate_limit=30
            ),
            TrustedSource(
                id="boom_live",
                name="BOOM Live",
                source_type=SourceType.BOOM_LIVE,
                domain="boomlive.in",
                url="https://www.boomlive.in",
                description="Indian fact-checking and news verification platform",
                reliability_score=0.85,
                last_updated=datetime.now(),
                is_active=True,
                rate_limit=30
            ),
            TrustedSource(
                id="science_feedback",
                name="Science Feedback",
                source_type=SourceType.SCIENCE_FEEDBACK,
                domain="sciencefeedback.co",
                url="https://sciencefeedback.co",
                description="Fact-checking platform for scientific claims",
                reliability_score=0.90,
                last_updated=datetime.now(),
                is_active=True,
                rate_limit=30
            ),
            
            # News portals
            TrustedSource(
                id="reuters",
                name="Reuters",
                source_type=SourceType.NEWS_PORTAL,
                domain="reuters.com",
                url="https://www.reuters.com",
                description="International news agency",
                reliability_score=0.85,
                last_updated=datetime.now(),
                is_active=True
            ),
            TrustedSource(
                id="ap",
                name="Associated Press",
                source_type=SourceType.NEWS_PORTAL,
                domain="ap.org",
                url="https://www.ap.org",
                description="International news agency",
                reliability_score=0.85,
                last_updated=datetime.now(),
                is_active=True
            ),
            TrustedSource(
                id="bbc",
                name="BBC News",
                source_type=SourceType.NEWS_PORTAL,
                domain="bbc.com",
                url="https://www.bbc.com/news",
                description="British public service broadcaster",
                reliability_score=0.80,
                last_updated=datetime.now(),
                is_active=True
            )
        ]
        
        # Add sources that don't already exist
        for source in default_sources:
            if source.id not in self.sources:
                self.sources[source.id] = source
        
        self._save_sources()
    
    def _save_sources(self):
        """Save sources to file."""
        try:
            data = {}
            for source_id, source in self.sources.items():
                data[source_id] = asdict(source)
                # Convert datetime to string for JSON serialization
                data[source_id]["last_updated"] = source.last_updated.isoformat()
            
            with open(self.sources_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving sources: {e}")
    
    def add_source(self, source: TrustedSource):
        """Add a new trusted source."""
        self.sources[source.id] = source
        self._save_sources()
        logger.info(f"Added trusted source: {source.name}")
    
    def remove_source(self, source_id: str):
        """Remove a trusted source."""
        if source_id in self.sources:
            del self.sources[source_id]
            self._save_sources()
            logger.info(f"Removed trusted source: {source_id}")
    
    def get_sources_by_type(self, source_type: SourceType) -> List[TrustedSource]:
        """Get all sources of a specific type."""
        return [source for source in self.sources.values() if source.source_type == source_type]
    
    def get_fact_check_sources(self) -> List[TrustedSource]:
        """Get all fact-checking sources."""
        fact_check_types = [
            SourceType.FACT_CHECK, SourceType.SNOPES, SourceType.POLITIFACT,
            SourceType.ALTNEWS, SourceType.BOOM_LIVE, SourceType.SCIENCE_FEEDBACK
        ]
        return [source for source in self.sources.values() if source.source_type in fact_check_types]
    
    def search_content(self, query: str, source_types: Optional[List[SourceType]] = None, 
                      max_results: int = 10) -> List[TrustedContent]:
        """
        Search content from trusted sources.
        
        Args:
            query: Search query
            source_types: Filter by source types
            max_results: Maximum number of results
            
        Returns:
            List of relevant content
        """
        results = []
        
        # Filter content by source types if specified
        if source_types:
            filtered_content = [
                content for content in self.content.values()
                if self.sources.get(content.source_id, {}).source_type in source_types
            ]
        else:
            filtered_content = list(self.content.values())
        
        # Simple keyword-based search (can be enhanced with semantic search)
        query_lower = query.lower()
        
        for content in filtered_content:
            # Check if query appears in title, snippet, or full text
            if (query_lower in content.title.lower() or 
                query_lower in content.snippet.lower() or 
                query_lower in content.full_text.lower()):
                
                # Calculate relevance score
                relevance = 0.0
                if query_lower in content.title.lower():
                    relevance += 0.5
                if query_lower in content.snippet.lower():
                    relevance += 0.3
                if query_lower in content.full_text.lower():
                    relevance += 0.2
                
                # Boost score for fact-checking sources
                source = self.sources.get(content.source_id)
                if source and source.source_type in [
                    SourceType.FACT_CHECK, SourceType.SNOPES, SourceType.POLITIFACT,
                    SourceType.ALTNEWS, SourceType.BOOM_LIVE, SourceType.SCIENCE_FEEDBACK
                ]:
                    relevance += 0.2
                
                content.relevance_score = min(relevance, 1.0)
                results.append(content)
        
        # Sort by relevance score and recency
        results.sort(key=lambda x: (x.relevance_score, x.published_date), reverse=True)
        
        return results[:max_results]
    
    def get_fact_check_results(self, claim: str) -> List[TrustedContent]:
        """
        Get fact-check results for a claim from multiple fact-checking sources.
        
        Args:
            claim: The claim to fact-check
            
        Returns:
            List of fact-check results
        """
        fact_check_sources = self.get_fact_check_sources()
        results = []
        
        for source in fact_check_sources:
            try:
                # Rate limiting
                if source.rate_limit:
                    time.sleep(60 / source.rate_limit)  # Respect rate limit
                
                # Search for fact-check results
                source_results = self._search_fact_check_source(source, claim)
                results.extend(source_results)
                
            except Exception as e:
                logger.error(f"Error searching {source.name}: {e}")
                continue
        
        return results
    
    def _search_fact_check_source(self, source: TrustedSource, claim: str) -> List[TrustedContent]:
        """
        Search a specific fact-checking source for results.
        
        Args:
            source: The fact-checking source
            claim: The claim to search for
            
        Returns:
            List of fact-check results
        """
        # This is a placeholder implementation
        # In a real implementation, you would integrate with each source's API
        
        if source.source_type == SourceType.SNOPES:
            return self._search_snopes(claim)
        elif source.source_type == SourceType.POLITIFACT:
            return self._search_politifact(claim)
        elif source.source_type == SourceType.ALTNEWS:
            return self._search_altnews(claim)
        elif source.source_type == SourceType.BOOM_LIVE:
            return self._search_boom_live(claim)
        elif source.source_type == SourceType.SCIENCE_FEEDBACK:
            return self._search_science_feedback(claim)
        else:
            # Generic search for other sources
            return self.search_content(claim, [source.source_type])
    
    def _search_snopes(self, claim: str) -> List[TrustedContent]:
        """Search Snopes for fact-check results."""
        # Placeholder implementation
        # In reality, you would use Snopes API or web scraping
        logger.info(f"Searching Snopes for: {claim}")
        return []
    
    def _search_politifact(self, claim: str) -> List[TrustedContent]:
        """Search PolitiFact for fact-check results."""
        # Placeholder implementation
        logger.info(f"Searching PolitiFact for: {claim}")
        return []
    
    def _search_altnews(self, claim: str) -> List[TrustedContent]:
        """Search Alt News for fact-check results."""
        # Placeholder implementation
        logger.info(f"Searching Alt News for: {claim}")
        return []
    
    def _search_boom_live(self, claim: str) -> List[TrustedContent]:
        """Search BOOM Live for fact-check results."""
        # Placeholder implementation
        logger.info(f"Searching BOOM Live for: {claim}")
        return []
    
    def _search_science_feedback(self, claim: str) -> List[TrustedContent]:
        """Search Science Feedback for fact-check results."""
        # Placeholder implementation
        logger.info(f"Searching Science Feedback for: {claim}")
        return []
    
    def add_content(self, content: TrustedContent):
        """Add content from a trusted source."""
        self.content[content.id] = content
        self._save_content()
    
    def _save_content(self):
        """Save content to file."""
        try:
            data = {}
            for content_id, content in self.content.items():
                data[content_id] = asdict(content)
                # Convert datetime to string for JSON serialization
                data[content_id]["published_date"] = content.published_date.isoformat()
                data[content_id]["last_updated"] = content.last_updated.isoformat()
            
            with open(self.content_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving content: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        source_type_counts = {}
        for source_type in SourceType:
            source_type_counts[source_type.value] = len(self.get_sources_by_type(source_type))
        
        return {
            "total_sources": len(self.sources),
            "total_content": len(self.content),
            "source_type_distribution": source_type_counts,
            "fact_check_sources": len(self.get_fact_check_sources()),
            "active_sources": len([s for s in self.sources.values() if s.is_active])
        }


class TrustedSourcesAPI:
    """
    API wrapper for trusted sources.
    
    Enhanced with fact-checking source integration.
    """
    
    def __init__(self, database: TrustedSourcesDatabase):
        """
        Initialize the API wrapper.
        
        Args:
            database: Trusted sources database
        """
        self.database = database
    
    def search_claim(self, claim: str, include_fact_checks: bool = True) -> Dict[str, Any]:
        """
        Search for evidence related to a claim.
        
        Args:
            claim: The claim to search for
            include_fact_checks: Whether to include fact-checking sources
            
        Returns:
            Dictionary with search results
        """
        results = {
            "claim": claim,
            "total_results": 0,
            "fact_check_results": [],
            "news_results": [],
            "government_results": [],
            "research_results": []
        }
        
        # Search fact-checking sources
        if include_fact_checks:
            fact_check_results = self.database.get_fact_check_results(claim)
            results["fact_check_results"] = [asdict(content) for content in fact_check_results]
        
        # Search news sources
        news_results = self.database.search_content(
            claim, 
            [SourceType.NEWS_PORTAL]
        )
        results["news_results"] = [asdict(content) for content in news_results]
        
        # Search government sources
        gov_results = self.database.search_content(
            claim,
            [SourceType.PIB, SourceType.WHO, SourceType.RBI, SourceType.GOVERNMENT]
        )
        results["government_results"] = [asdict(content) for content in gov_results]
        
        # Search research sources
        research_results = self.database.search_content(
            claim,
            [SourceType.RESEARCH]
        )
        results["research_results"] = [asdict(content) for content in research_results]
        
        # Calculate total results
        results["total_results"] = (
            len(results["fact_check_results"]) +
            len(results["news_results"]) +
            len(results["government_results"]) +
            len(results["research_results"])
        )
        
        return results
