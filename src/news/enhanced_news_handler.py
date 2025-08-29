#!/usr/bin/env python3
"""
Enhanced News Handler for TruthLens
Integrates News API and Guardian API for comprehensive news sourcing and cross-referencing.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import sqlite3
import hashlib
import json

# Import existing news handler and new guardian handler
from .news_handler import NewsHandler, NewsArticle
from .guardian_api_handler import GuardianAPIHandler
from .currents_api_handler import CurrentsAPIHandler
from ..evidence_retrieval.semantic_cross_reference_scorer import SemanticCrossReferenceScorer

logger = logging.getLogger(__name__)

@dataclass
class EnhancedNewsArticle:
    """Enhanced news article with source information and cross-referencing data."""
    title: str
    description: str
    content: str
    url: str
    source: str
    source_name: str  # "NewsAPI" or "Guardian"
    published_at: str
    relevance_score: float
    api_type: str  # "newsapi" or "guardian"
    cross_reference_score: Optional[float] = None
    matching_headlines: List[str] = None

class EnhancedNewsHandler:
    """
    Enhanced news handler that integrates News API and Guardian API.
    
    Features:
    - Primary: News API for comprehensive coverage
    - Secondary: Guardian API for cross-referencing and credibility
    - Cross-reference scoring for increased confidence
    - Source diversity for better fact-checking
    """
    
    def __init__(self, news_api_key: str, guardian_api_key: str, currents_api_key: str = None):
        """
        Initialize the enhanced news handler.
        
        Args:
            news_api_key: News API key
            guardian_api_key: Guardian API key
            currents_api_key: Currents API key (optional)
        """
        self.news_handler = NewsHandler(news_api_key)
        self.guardian_handler = GuardianAPIHandler(guardian_api_key)
        self.currents_handler = CurrentsAPIHandler(currents_api_key)
        self.cross_reference_scorer = SemanticCrossReferenceScorer()
        
        # Initialize NewsAPI cache
        self._init_newsapi_cache()
        
        logger.info("Enhanced News Handler initialized with News API, Guardian API, and Currents API")
    
    def get_news_sources(self, query: str, max_results: int = 15, days_back: int = 30, 
                        prefer_sources: Optional[List[str]] = None) -> List[EnhancedNewsArticle]:
        """
        Get news from multiple sources with cross-referencing.
        
        Args:
            query: Search query
            max_results: Maximum total results to return
            days_back: Number of days back to search
            prefer_sources: Preferred source order (e.g., ["guardian", "newsapi"])
            
        Returns:
            List of enhanced news articles from multiple sources
        """
        try:
            logger.info(f"Fetching news for query: '{query}' from multiple sources")
            
            # Step 1: Get News API results (primary source)
            news_api_results = self._get_news_api_results(query, max_results // 3, days_back)
            logger.info(f"News API returned {len(news_api_results)} articles")
            
            # Step 2: Get Guardian API results (secondary source)
            guardian_results = self._get_guardian_results(query, max_results // 3, days_back)
            logger.info(f"Guardian API returned {len(guardian_results)} articles")
            
            # Step 3: Get Currents API results (tertiary source)
            currents_results = self._get_currents_results(query, max_results // 3, days_back)
            logger.info(f"Currents API returned {len(currents_results)} articles")
            
            # Step 4: Merge and cross-reference results with semantic scoring
            merged_results = self._merge_and_cross_reference(
                news_api_results, guardian_results, currents_results, query, prefer_sources
            )
            
            # Step 5: Sort by relevance and cross-reference score
            merged_results.sort(
                key=lambda x: (x.relevance_score, x.cross_reference_score or 0),
                reverse=True
            )
            
            # Step 6: Return top results
            final_results = merged_results[:max_results]
            
            logger.info(f"Final merged results: {len(final_results)} articles")
            logger.info(f"Source breakdown: NewsAPI={sum(1 for r in final_results if r.source_name == 'NewsAPI')}, "
                       f"Guardian={sum(1 for r in final_results if r.source_name == 'Guardian')}, "
                       f"Currents={sum(1 for r in final_results if r.source_name == 'Currents')}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in get_news_sources: {e}")
            # Fallback to News API only
            try:
                news_results = self._get_news_api_results(query, max_results, days_back)
                return self._convert_to_enhanced_articles(news_results, "NewsAPI")
            except Exception as fallback_error:
                logger.error(f"Fallback to News API also failed: {fallback_error}")
                return []
    
    def _get_news_api_results(self, query: str, max_results: int, days_back: int) -> List[NewsArticle]:
        """Get results from News API with caching and rate limit handling."""
        try:
            # Check if NewsAPI is rate limited
            if self._is_newsapi_rate_limited():
                logger.warning("NewsAPI is currently rate limited, using cached results if available")
                cached_results = self._get_cached_newsapi_results(query, days_back)
                if cached_results:
                    return cached_results[:max_results]
                else:
                    logger.info("No cached results available, skipping NewsAPI")
                    return []
            
            # Check cache first
            cached_results = self._get_cached_newsapi_results(query, days_back)
            if cached_results:
                logger.info(f"Using cached NewsAPI results for '{query}'")
                return cached_results[:max_results]
            
            # Make actual API call
            logger.info(f"Making NewsAPI call for '{query}'")
            results = self.news_handler.search_news(query, max_results, days_back)
            
            # Cache the results
            if results:
                self._cache_newsapi_results(query, days_back, results)
            
            return results
            
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                logger.warning("NewsAPI rate limit hit, marking as rate limited")
                self._mark_newsapi_rate_limited()
                
                # Try to get cached results
                cached_results = self._get_cached_newsapi_results(query, days_back)
                if cached_results:
                    logger.info("Using cached results after rate limit")
                    return cached_results[:max_results]
            
            logger.warning(f"News API search failed: {e}")
            return []
    
    def _get_guardian_results(self, query: str, max_results: int, days_back: int) -> List[Dict[str, Any]]:
        """Get results from Guardian API."""
        try:
            return self.guardian_handler.fetch_guardian_news(query, max_results, days_back)
        except Exception as e:
            logger.warning(f"Guardian API search failed: {e}")
            return []
    
    def _get_currents_results(self, query: str, max_results: int, days_back: int) -> List[Dict[str, Any]]:
        """Get results from Currents API."""
        try:
            if self.currents_handler.is_available():
                return self.currents_handler.fetch_currents_news(query, max_results, days_back)
            else:
                logger.info("Currents API not available, skipping")
                return []
        except Exception as e:
            logger.warning(f"Currents API search failed: {e}")
            return []
    
    def _convert_to_enhanced_articles(self, articles: List[Any], source_name: str) -> List[EnhancedNewsArticle]:
        """Convert articles to enhanced format."""
        enhanced_articles = []
        
        for article in articles:
            if source_name == "NewsAPI":
                # Convert NewsArticle to EnhancedNewsArticle
                enhanced_article = EnhancedNewsArticle(
                    title=article.title,
                    description=article.description,
                    content=article.content,
                    url=article.url,
                    source=article.source,
                    source_name=source_name,
                    published_at=article.published_at,
                    relevance_score=article.relevance_score,
                    api_type="newsapi",
                    cross_reference_score=None,
                    matching_headlines=[]
                )
            else:
                # Convert Guardian API result to EnhancedNewsArticle
                enhanced_article = EnhancedNewsArticle(
                    title=article.get('title', ''),
                    description=article.get('description', ''),
                    content=article.get('content', ''),
                    url=article.get('url', ''),
                    source=article.get('source', ''),
                    source_name=source_name,
                    published_at=article.get('published_at', ''),
                    relevance_score=article.get('relevance_score', 0.5),
                    api_type="guardian",
                    cross_reference_score=None,
                    matching_headlines=[]
                )
            
            enhanced_articles.append(enhanced_article)
        
        return enhanced_articles
    
    def _init_newsapi_cache(self):
        """Initialize SQLite cache for NewsAPI results."""
        try:
            self.cache_conn = sqlite3.connect(':memory:', check_same_thread=False)
            cursor = self.cache_conn.cursor()
            
            # Create cache table for NewsAPI results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS newsapi_cache (
                    query_hash TEXT PRIMARY KEY,
                    results_json TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    rate_limit_hit BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create table for rate limit tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rate_limit_tracker (
                    api_name TEXT PRIMARY KEY,
                    last_request_time DATETIME,
                    request_count INTEGER DEFAULT 0,
                    rate_limit_hit BOOLEAN DEFAULT FALSE
                )
            """)
            
            self.cache_conn.commit()
            logger.info("NewsAPI cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NewsAPI cache: {e}")
            self.cache_conn = None
    
    def _is_newsapi_rate_limited(self) -> bool:
        """Check if NewsAPI is currently rate limited."""
        if not self.cache_conn:
            return False
        
        try:
            cursor = self.cache_conn.cursor()
            cursor.execute("""
                SELECT rate_limit_hit, last_request_time FROM rate_limit_tracker 
                WHERE api_name = 'newsapi'
            """)
            
            result = cursor.fetchone()
            if result:
                rate_limit_hit, last_request_time = result
                if rate_limit_hit:
                    # Check if 24 hours have passed since rate limit
                    last_time = datetime.fromisoformat(last_request_time)
                    if datetime.now() - last_time < timedelta(hours=24):
                        return True
                    else:
                        # Reset rate limit flag
                        cursor.execute("""
                            UPDATE rate_limit_tracker 
                            SET rate_limit_hit = FALSE, request_count = 0 
                            WHERE api_name = 'newsapi'
                        """)
                        self.cache_conn.commit()
                        return False
            
            return False
            
        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}")
            return False
    
    def _mark_newsapi_rate_limited(self):
        """Mark NewsAPI as rate limited."""
        if not self.cache_conn:
            return
        
        try:
            cursor = self.cache_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO rate_limit_tracker 
                (api_name, last_request_time, request_count, rate_limit_hit)
                VALUES ('newsapi', datetime('now'), 0, TRUE)
            """)
            self.cache_conn.commit()
            logger.warning("NewsAPI marked as rate limited")
            
        except Exception as e:
            logger.error(f"Failed to mark NewsAPI as rate limited: {e}")
    
    def _get_cached_newsapi_results(self, query: str, days_back: int) -> Optional[List[NewsArticle]]:
        """Get cached NewsAPI results if available."""
        if not self.cache_conn:
            return None
        
        try:
            # Generate cache key
            cache_key = hashlib.md5(f"{query}_{days_back}".encode()).hexdigest()
            
            cursor = self.cache_conn.cursor()
            cursor.execute("""
                SELECT results_json, timestamp FROM newsapi_cache 
                WHERE query_hash = ? AND timestamp > datetime('now', '-6 hours')
            """, (cache_key,))
            
            result = cursor.fetchone()
            if result:
                results_json, timestamp = result
                # Convert back to NewsArticle objects
                articles_data = json.loads(results_json)
                cached_articles = []
                
                for article_data in articles_data:
                    article = NewsArticle(
                        title=article_data['title'],
                        description=article_data['description'],
                        content=article_data['content'],
                        url=article_data['url'],
                        source=article_data['source'],
                        published_at=article_data['published_at'],
                        relevance_score=article_data['relevance_score']
                    )
                    cached_articles.append(article)
                
                logger.info(f"Retrieved {len(cached_articles)} cached NewsAPI results for '{query}'")
                return cached_articles
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _cache_newsapi_results(self, query: str, days_back: int, results: List[NewsArticle]):
        """Cache NewsAPI results."""
        if not self.cache_conn:
            return
        
        try:
            # Generate cache key
            cache_key = hashlib.md5(f"{query}_{days_back}".encode()).hexdigest()
            
            # Convert NewsArticle objects to JSON-serializable format
            articles_data = []
            for article in results:
                article_data = {
                    'title': article.title,
                    'description': article.description,
                    'content': article.content,
                    'url': article.url,
                    'source': article.source,
                    'published_at': article.published_at,
                    'relevance_score': article.relevance_score
                }
                articles_data.append(article_data)
            
            results_json = json.dumps(articles_data)
            
            cursor = self.cache_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO newsapi_cache (query_hash, results_json, timestamp)
                VALUES (?, ?, datetime('now'))
            """, (cache_key, results_json))
            
            self.cache_conn.commit()
            logger.info(f"Cached {len(results)} NewsAPI results for '{query}'")
            
        except Exception as e:
            logger.warning(f"Failed to cache NewsAPI results: {e}")
    
    def _merge_and_cross_reference(self, news_api_results: List[NewsArticle], 
                                  guardian_results: List[Dict[str, Any]], 
                                  currents_results: List[Dict[str, Any]],
                                  query: str,
                                  prefer_sources: Optional[List[str]] = None) -> List[EnhancedNewsArticle]:
        """
        Merge results from three sources and calculate cross-reference scores using semantic similarity.
        
        Args:
            news_api_results: Results from News API
            guardian_results: Results from Guardian API
            currents_results: Results from Currents API
            query: Original search query
            prefer_sources: Preferred source order
            
        Returns:
            List of enhanced articles with cross-reference scores
        """
        # Convert all to enhanced format
        enhanced_news_api = self._convert_to_enhanced_articles(news_api_results, "NewsAPI")
        enhanced_guardian = self._convert_to_enhanced_articles(guardian_results, "Guardian")
        enhanced_currents = self._convert_to_enhanced_articles(currents_results, "Currents")
        
        # Combine all articles
        all_articles = enhanced_news_api + enhanced_guardian + enhanced_currents
        
        # Use semantic cross-reference scorer for advanced similarity calculation
        cross_reference_scores = self.cross_reference_scorer.calculate_cross_reference_scores(
            all_articles, query, prefer_sources
        )
        
        # Apply cross-reference scores to articles
        for i, article in enumerate(all_articles):
            if i < len(cross_reference_scores):
                score = cross_reference_scores[i]
                article.cross_reference_score = score.credibility_boost
                # Add verification badge and evidence strength
                setattr(article, 'verification_badge', score.verification_badge)
                setattr(article, 'evidence_strength', score.evidence_strength)
                setattr(article, 'matching_articles', score.matching_articles)
        
        return all_articles
    
    def _calculate_cross_reference_score(self, article: EnhancedNewsArticle, 
                                       all_articles: List[EnhancedNewsArticle], 
                                       query: str) -> float:
        """
        Calculate cross-reference score based on similar headlines/content.
        
        Args:
            article: Article to score
            all_articles: All articles from both sources
            query: Original search query
            
        Returns:
            Cross-reference score (0.0 to 1.0)
        """
        if not all_articles:
            return 0.0
        
        matching_headlines = []
        total_similarity = 0.0
        comparison_count = 0
        
        for other_article in all_articles:
            if other_article.url == article.url:  # Skip self-comparison
                continue
            
            # Calculate similarity between articles
            similarity = self._calculate_article_similarity(article, other_article)
            
            if similarity > 0.7:  # High similarity threshold
                matching_headlines.append(other_article.title)
                total_similarity += similarity
                comparison_count += 1
        
        # Store matching headlines
        article.matching_headlines = matching_headlines
        
        # Calculate cross-reference score
        if comparison_count > 0:
            avg_similarity = total_similarity / comparison_count
            # Boost score for articles with multiple sources covering similar content
            cross_ref_score = min(1.0, avg_similarity * (1 + len(matching_headlines) * 0.1))
        else:
            cross_ref_score = 0.0
        
        return cross_ref_score
    
    def _calculate_article_similarity(self, article1: EnhancedNewsArticle, 
                                    article2: EnhancedNewsArticle) -> float:
        """
        Calculate similarity between two articles.
        
        Args:
            article1: First article
            article2: Second article
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple text similarity using word overlap
        title1_words = set(article1.title.lower().split())
        title2_words = set(article2.title.lower().split())
        
        if not title1_words or not title2_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(title1_words.intersection(title2_words))
        union = len(title1_words.union(title2_words))
        
        if union == 0:
            return 0.0
        
        title_similarity = intersection / union
        
        # Also check content similarity for longer articles
        content_similarity = 0.0
        if len(article1.content) > 100 and len(article2.content) > 100:
            content1_words = set(article1.content.lower().split()[:50])  # First 50 words
            content2_words = set(article2.content.lower().split()[:50])
            
            if content1_words and content2_words:
                content_intersection = len(content1_words.intersection(content2_words))
                content_union = len(content1_words.union(content2_words))
                content_similarity = content_intersection / content_union if content_union > 0 else 0.0
        
        # Weighted combination (title more important than content)
        final_similarity = (title_similarity * 0.7) + (content_similarity * 0.3)
        
        return final_similarity
    
    def get_credibility_boost(self, articles: List[EnhancedNewsArticle]) -> Dict[str, Any]:
        """
        Calculate credibility boost based on cross-referencing.
        
        Args:
            articles: List of enhanced news articles
            
        Returns:
            Dictionary with credibility metrics
        """
        if not articles:
            return {"credibility_score": 0.0, "cross_references": 0, "source_diversity": 0.0}
        
        # Count cross-references
        cross_references = sum(1 for article in articles if article.cross_reference_score and article.cross_reference_score > 0.5)
        
        # Calculate source diversity
        source_names = set(article.source_name for article in articles)
        source_diversity = len(source_names) / 2.0  # Normalize to 0.0-1.0 (2 sources max)
        
        # Calculate average cross-reference score
        avg_cross_ref = sum(article.cross_reference_score or 0.0 for article in articles) / len(articles)
        
        # Overall credibility score
        credibility_score = (avg_cross_ref * 0.6) + (source_diversity * 0.4)
        
        # Get cross-reference summary if available
        cross_ref_summary = {}
        if hasattr(self, 'cross_reference_scorer'):
            try:
                cross_ref_summary = self.cross_reference_scorer.get_cross_reference_summary(
                    [getattr(article, 'cross_reference_score', 0) for article in articles]
                )
            except Exception as e:
                logger.warning(f"Could not get cross-reference summary: {e}")
        
        return {
            "credibility_score": min(1.0, credibility_score),
            "cross_references": cross_references,
            "source_diversity": source_diversity,
            "avg_cross_reference_score": avg_cross_ref,
            "total_articles": len(articles),
            "source_breakdown": {source: sum(1 for a in articles if a.source_name == source) for source in source_names},
            "cross_reference_summary": cross_ref_summary
        }
    
    def search_with_fallback(self, query: str, max_results: int = 10, days_back: int = 30) -> List[EnhancedNewsArticle]:
        """
        Search with fallback strategy: try News API first, then Guardian if needed.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            days_back: Number of days back to search
            
        Returns:
            List of enhanced news articles
        """
        try:
            # Try News API first
            news_results = self._get_news_api_results(query, max_results, days_back)
            
            if len(news_results) >= max_results // 2:
                # News API provided sufficient results, add some Guardian for diversity
                guardian_results = self._get_guardian_results(query, max_results // 4, days_back)
                merged = self._merge_and_cross_reference(news_results, guardian_results, query)
            else:
                # News API didn't provide enough results, get more from Guardian
                guardian_results = self._get_guardian_results(query, max_results - len(news_results), days_back)
                merged = self._merge_and_cross_reference(news_results, guardian_results, query)
            
            # Sort and return
            merged.sort(key=lambda x: (x.relevance_score, x.cross_reference_score or 0), reverse=True)
            return merged[:max_results]
            
        except Exception as e:
            logger.error(f"Error in search_with_fallback: {e}")
            # Return whatever we have
            return self._convert_to_enhanced_articles(news_results, "NewsAPI") if 'news_results' in locals() else []
