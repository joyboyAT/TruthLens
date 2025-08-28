"""
Trusted Sources Database for TruthLens Phase 3
Implements integration with trusted sources: PIB, WHO, RBI, major news portals
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

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


class TrustedSourcesDatabase:
    """
    Database of trusted sources and their content.
    
    Features:
    - Curated list of trusted sources
    - Content caching and freshness tracking
    - Relevance scoring
    - API integration for real-time updates
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
                            is_active=source_data.get("is_active", True)
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
                            relevance_score=content_data.get("relevance_score", 0.0)
                        )
                        self.content[content.id] = content
                        
        except Exception as e:
            logger.warning(f"Failed to load trusted sources data: {e}")
    
    def _save_data(self):
        """Save data to files."""
        try:
            # Save sources
            sources_data = {}
            for source in self.sources.values():
                sources_data[source.id] = asdict(source)
                sources_data[source.id]["source_type"] = source.source_type.value
                sources_data[source.id]["last_updated"] = source.last_updated.isoformat()
            
            with open(self.sources_file, 'w', encoding='utf-8') as f:
                json.dump(sources_data, f, indent=2, ensure_ascii=False)
            
            # Save content
            content_data = {}
            for content in self.content.values():
                content_data[content.id] = asdict(content)
                content_data[content.id]["published_date"] = content.published_date.isoformat()
                content_data[content.id]["last_updated"] = content.last_updated.isoformat()
            
            with open(self.content_file, 'w', encoding='utf-8') as f:
                json.dump(content_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save trusted sources data: {e}")
    
    def _initialize_default_sources(self):
        """Initialize default trusted sources if none exist."""
        if not self.sources:
            default_sources = [
                TrustedSource(
                    id="pib_main",
                    name="Press Information Bureau",
                    source_type=SourceType.PIB,
                    domain="pib.gov.in",
                    url="https://pib.gov.in",
                    description="Official press releases and announcements from Government of India",
                    reliability_score=0.95,
                    last_updated=datetime.now(),
                    is_active=True
                ),
                TrustedSource(
                    id="who_main",
                    name="World Health Organization",
                    source_type=SourceType.WHO,
                    domain="who.int",
                    url="https://www.who.int",
                    description="International health guidelines and advisories",
                    reliability_score=0.95,
                    last_updated=datetime.now(),
                    is_active=True
                ),
                TrustedSource(
                    id="rbi_main",
                    name="Reserve Bank of India",
                    source_type=SourceType.RBI,
                    domain="rbi.org.in",
                    url="https://www.rbi.org.in",
                    description="Official economic and financial data from RBI",
                    reliability_score=0.95,
                    last_updated=datetime.now(),
                    is_active=True
                ),
                TrustedSource(
                    id="factcheck_org",
                    name="FactCheck.org",
                    source_type=SourceType.FACT_CHECK,
                    domain="factcheck.org",
                    url="https://www.factcheck.org",
                    description="Non-partisan fact-checking organization",
                    reliability_score=0.90,
                    last_updated=datetime.now(),
                    is_active=True
                ),
                TrustedSource(
                    id="reuters",
                    name="Reuters",
                    source_type=SourceType.NEWS_PORTAL,
                    domain="reuters.com",
                    url="https://www.reuters.com",
                    description="International news agency with high editorial standards",
                    reliability_score=0.85,
                    last_updated=datetime.now(),
                    is_active=True
                ),
                TrustedSource(
                    id="bbc_news",
                    name="BBC News",
                    source_type=SourceType.NEWS_PORTAL,
                    domain="bbc.com",
                    url="https://www.bbc.com/news",
                    description="British public service broadcaster",
                    reliability_score=0.85,
                    last_updated=datetime.now(),
                    is_active=True
                )
            ]
            
            for source in default_sources:
                self.sources[source.id] = source
            
            self._save_data()
    
    def add_source(self, source: TrustedSource):
        """Add a new trusted source."""
        self.sources[source.id] = source
        self._save_data()
    
    def remove_source(self, source_id: str):
        """Remove a trusted source."""
        if source_id in self.sources:
            del self.sources[source_id]
            self._save_data()
    
    def get_sources_by_type(self, source_type: SourceType) -> List[TrustedSource]:
        """Get all sources of a specific type."""
        return [s for s in self.sources.values() if s.source_type == source_type and s.is_active]
    
    def get_high_reliability_sources(self, min_score: float = 0.8) -> List[TrustedSource]:
        """Get sources with reliability score above threshold."""
        return [s for s in self.sources.values() if s.reliability_score >= min_score and s.is_active]
    
    def add_content(self, content: TrustedContent):
        """Add content from a trusted source."""
        self.content[content.id] = content
        self._save_data()
    
    def search_content(self, query: str, source_types: Optional[List[SourceType]] = None, 
                      max_results: int = 10, min_relevance: float = 0.3) -> List[TrustedContent]:
        """
        Search content from trusted sources.
        
        Args:
            query: Search query
            source_types: Filter by source types
            max_results: Maximum number of results
            min_relevance: Minimum relevance score
            
        Returns:
            List of relevant content
        """
        # Simple keyword-based search (can be enhanced with vector search)
        query_lower = query.lower()
        results = []
        
        for content in self.content.values():
            # Check source type filter
            if source_types:
                source = self.sources.get(content.source_id)
                if not source or source.source_type not in source_types:
                    continue
            
            # Calculate relevance score
            relevance = 0.0
            
            # Title match
            if query_lower in content.title.lower():
                relevance += 0.5
            
            # Snippet match
            if query_lower in content.snippet.lower():
                relevance += 0.3
            
            # Full text match
            if query_lower in content.full_text.lower():
                relevance += 0.2
            
            # Tag match
            for tag in content.tags:
                if query_lower in tag.lower():
                    relevance += 0.1
            
            # Freshness bonus
            days_old = (datetime.now() - content.published_date).days
            if days_old <= 7:
                relevance += 0.1
            elif days_old <= 30:
                relevance += 0.05
            
            if relevance >= min_relevance:
                content.relevance_score = relevance
                results.append(content)
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:max_results]
    
    def get_recent_content(self, days: int = 7, source_types: Optional[List[SourceType]] = None) -> List[TrustedContent]:
        """Get recent content from trusted sources."""
        cutoff_date = datetime.now() - timedelta(days=days)
        results = []
        
        for content in self.content.values():
            if content.published_date >= cutoff_date:
                if source_types:
                    source = self.sources.get(content.source_id)
                    if source and source.source_type in source_types:
                        results.append(content)
                else:
                    results.append(content)
        
        results.sort(key=lambda x: x.published_date, reverse=True)
        return results
    
    def update_content_freshness(self):
        """Update content freshness and remove old content."""
        cutoff_date = datetime.now() - timedelta(days=365)  # Keep content for 1 year
        old_content_ids = []
        
        for content_id, content in self.content.items():
            if content.published_date < cutoff_date:
                old_content_ids.append(content_id)
        
        for content_id in old_content_ids:
            del self.content[content_id]
        
        if old_content_ids:
            self._save_data()
            logger.info(f"Removed {len(old_content_ids)} old content items")


class TrustedSourcesAPI:
    """
    API integration for trusted sources.
    
    Features:
    - Real-time content fetching
    - RSS feed parsing
    - API rate limiting
    - Content validation
    """
    
    def __init__(self, database: TrustedSourcesDatabase):
        """
        Initialize the API client.
        
        Args:
            database: TrustedSourcesDatabase instance
        """
        self.database = database
    
    def fetch_pib_content(self, max_items: int = 20) -> List[TrustedContent]:
        """Fetch content from Press Information Bureau."""
        # This would integrate with PIB's API or RSS feed
        # For now, return mock data
        mock_content = [
            TrustedContent(
                id=f"pib_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source_id="pib_main",
                title="Government Announces New Economic Package",
                url="https://pib.gov.in/newsite/PrintRelease.aspx?relid=123456",
                snippet="The Government of India has announced a comprehensive economic package to support various sectors.",
                full_text="The Government of India has announced a comprehensive economic package to support various sectors including agriculture, manufacturing, and services. The package includes measures for financial support, policy reforms, and infrastructure development.",
                published_date=datetime.now() - timedelta(hours=2),
                last_updated=datetime.now(),
                tags=["economy", "government", "policy"],
                relevance_score=0.8
            )
        ]
        
        for content in mock_content:
            self.database.add_content(content)
        
        return mock_content
    
    def fetch_who_content(self, max_items: int = 20) -> List[TrustedContent]:
        """Fetch content from World Health Organization."""
        # This would integrate with WHO's API or RSS feed
        mock_content = [
            TrustedContent(
                id=f"who_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source_id="who_main",
                title="WHO Updates COVID-19 Guidelines",
                url="https://www.who.int/news-room/detail/covid-19-guidelines",
                snippet="The World Health Organization has updated its COVID-19 prevention and treatment guidelines.",
                full_text="The World Health Organization has updated its COVID-19 prevention and treatment guidelines based on the latest scientific evidence. The new guidelines include updated recommendations for vaccination, treatment protocols, and preventive measures.",
                published_date=datetime.now() - timedelta(hours=4),
                last_updated=datetime.now(),
                tags=["health", "covid-19", "guidelines"],
                relevance_score=0.9
            )
        ]
        
        for content in mock_content:
            self.database.add_content(content)
        
        return mock_content
    
    def fetch_rbi_content(self, max_items: int = 20) -> List[TrustedContent]:
        """Fetch content from Reserve Bank of India."""
        # This would integrate with RBI's API or RSS feed
        mock_content = [
            TrustedContent(
                id=f"rbi_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source_id="rbi_main",
                title="RBI Announces Monetary Policy Decision",
                url="https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=12345",
                snippet="The Reserve Bank of India has announced its monetary policy decision for the current quarter.",
                full_text="The Reserve Bank of India has announced its monetary policy decision for the current quarter. The central bank has maintained the repo rate at current levels while providing forward guidance on inflation and growth outlook.",
                published_date=datetime.now() - timedelta(hours=6),
                last_updated=datetime.now(),
                tags=["economy", "monetary policy", "rbi"],
                relevance_score=0.85
            )
        ]
        
        for content in mock_content:
            self.database.add_content(content)
        
        return mock_content
    
    def update_all_sources(self):
        """Update content from all active sources."""
        logger.info("Updating content from all trusted sources...")
        
        # Update PIB content
        try:
            self.fetch_pib_content()
            logger.info("Updated PIB content")
        except Exception as e:
            logger.error(f"Failed to update PIB content: {e}")
        
        # Update WHO content
        try:
            self.fetch_who_content()
            logger.info("Updated WHO content")
        except Exception as e:
            logger.error(f"Failed to update WHO content: {e}")
        
        # Update RBI content
        try:
            self.fetch_rbi_content()
            logger.info("Updated RBI content")
        except Exception as e:
            logger.error(f"Failed to update RBI content: {e}")
        
        # Update content freshness
        self.database.update_content_freshness()
        logger.info("Completed trusted sources update")


def get_trusted_sources_database() -> TrustedSourcesDatabase:
    """Get or create the trusted sources database."""
    return TrustedSourcesDatabase()


def get_trusted_sources_api() -> TrustedSourcesAPI:
    """Get or create the trusted sources API client."""
    database = get_trusted_sources_database()
    return TrustedSourcesAPI(database)
