#!/usr/bin/env python3
"""
Semantic Cross-Reference Scorer for TruthLens
Uses Sentence-BERT to calculate similarity between Guardian and NewsAPI articles
and boost credibility scores for articles covered by multiple sources.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class CrossReferenceScore:
    """Result of cross-reference scoring between articles."""
    article_id: str
    source_name: str
    similarity_score: float
    matching_articles: List[Dict[str, Any]]
    credibility_boost: float
    verification_badge: str
    evidence_strength: str

@dataclass
class ArticleSimilarity:
    """Similarity between two articles."""
    article1_id: str
    article2_id: str
    source1: str
    source2: str
    title_similarity: float
    content_similarity: float
    semantic_similarity: float
    overall_similarity: float
    is_cross_source: bool

class SemanticCrossReferenceScorer:
    """
    Advanced semantic cross-reference scorer using Sentence-BERT.
    
    Features:
    - Semantic similarity using Sentence-BERT
    - Cross-source credibility boosting
    - Verification badges for multiple sources
    - SQLite caching for performance
    - Smart source preference handling
    """
    
    def __init__(self, cache_db_path: str = "cross_reference_cache.db"):
        """
        Initialize the semantic cross-reference scorer.
        
        Args:
            cache_db_path: Path to SQLite cache database
        """
        self.cache_db_path = cache_db_path
        self.sentence_transformer = None
        self.cache_conn = None
        
        # Initialize components
        self._initialize_sentence_transformer()
        self._initialize_cache()
        
        # Similarity thresholds
        self.high_similarity_threshold = 0.8
        self.medium_similarity_threshold = 0.6
        self.low_similarity_threshold = 0.4
        
        logger.info("Semantic Cross-Reference Scorer initialized")
    
    def _initialize_sentence_transformer(self):
        """Initialize Sentence-BERT model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence-BERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Sentence-BERT: {e}")
            self.sentence_transformer = None
    
    def _initialize_cache(self):
        """Initialize SQLite cache database."""
        try:
            self.cache_conn = sqlite3.connect(self.cache_db_path)
            self._create_cache_tables()
            logger.info("Cross-reference cache initialized")
        except Exception as e:
            logger.warning(f"Could not initialize cache: {e}")
            self.cache_conn = None
    
    def _create_cache_tables(self):
        """Create cache tables if they don't exist."""
        if not self.cache_conn:
            return
        
        cursor = self.cache_conn.cursor()
        
        # Cache for similarity calculations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS similarity_cache (
                query_hash TEXT PRIMARY KEY,
                article1_hash TEXT,
                article2_hash TEXT,
                similarity_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Cache for cross-reference results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cross_reference_cache (
                query_hash TEXT PRIMARY KEY,
                results_json TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.cache_conn.commit()
    
    def calculate_cross_reference_scores(self, 
                                       articles: List[Any], 
                                       query: str,
                                       prefer_sources: Optional[List[str]] = None) -> List[CrossReferenceScore]:
        """
        Calculate cross-reference scores for all articles.
        
        Args:
            articles: List of articles from multiple sources
            query: Original search query
            prefer_sources: Preferred source order (e.g., ["guardian", "newsapi"])
            
        Returns:
            List of cross-reference scores for each article
        """
        if not articles:
            return []
        
        # Check cache first
        cache_key = self._generate_cache_key(query, articles)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.info("Using cached cross-reference results")
            return cached_result
        
        logger.info(f"Calculating cross-reference scores for {len(articles)} articles")
        
        # Group articles by source
        source_groups = self._group_articles_by_source(articles)
        
        # Calculate cross-reference scores
        cross_reference_scores = []
        
        for article in articles:
            try:
                score = self._calculate_article_cross_reference(
                    article, articles, source_groups, prefer_sources
                )
                cross_reference_scores.append(score)
            except Exception as e:
                logger.warning(f"Error calculating cross-reference for article: {e}")
                # Create default score
                score = CrossReferenceScore(
                    article_id=self._get_article_id(article),
                    source_name=getattr(article, 'source_name', 'Unknown'),
                    similarity_score=0.0,
                    matching_articles=[],
                    credibility_boost=0.0,
                    verification_badge="⚠️ Single source",
                    evidence_strength="Weak"
                )
                cross_reference_scores.append(score)
        
        # Cache the results
        self._cache_result(cache_key, cross_reference_scores)
        
        return cross_reference_scores
    
    def _calculate_article_cross_reference(self, 
                                         article: Any, 
                                         all_articles: List[Any],
                                         source_groups: Dict[str, List[Any]],
                                         prefer_sources: Optional[List[str]] = None) -> CrossReferenceScore:
        """
        Calculate cross-reference score for a single article.
        
        Args:
            article: Article to score
            all_articles: All articles from all sources
            source_groups: Articles grouped by source
            prefer_sources: Preferred source order
            
        Returns:
            CrossReferenceScore for the article
        """
        article_id = self._get_article_id(article)
        source_name = getattr(article, 'source_name', 'Unknown')
        
        # Find similar articles from other sources
        similar_articles = []
        total_similarity = 0.0
        cross_source_count = 0
        
        for other_article in all_articles:
            if self._get_article_id(other_article) == article_id:
                continue
            
            # Calculate similarity
            similarity = self._calculate_article_similarity(article, other_article)
            
            if similarity > self.low_similarity_threshold:
                other_source = getattr(other_article, 'source_name', 'Unknown')
                is_cross_source = source_name != other_source
                
                similar_articles.append({
                    'id': self._get_article_id(other_article),
                    'title': getattr(other_article, 'title', ''),
                    'source': other_source,
                    'similarity': similarity,
                    'is_cross_source': is_cross_source
                })
                
                if is_cross_source:
                    cross_source_count += 1
                    total_similarity += similarity
        
        # Calculate credibility boost
        credibility_boost = self._calculate_credibility_boost(
            similar_articles, cross_source_count, total_similarity, prefer_sources
        )
        
        # Determine verification badge
        verification_badge = self._determine_verification_badge(
            similar_articles, cross_source_count, credibility_boost
        )
        
        # Determine evidence strength
        evidence_strength = self._determine_evidence_strength(
            similar_articles, cross_source_count, credibility_boost
        )
        
        return CrossReferenceScore(
            article_id=article_id,
            source_name=source_name,
            similarity_score=total_similarity / max(cross_source_count, 1),
            matching_articles=similar_articles,
            credibility_boost=credibility_boost,
            verification_badge=verification_badge,
            evidence_strength=evidence_strength
        )
    
    def _calculate_article_similarity(self, article1: Any, article2: Any) -> float:
        """
        Calculate semantic similarity between two articles.
        
        Args:
            article1: First article
            article2: Second article
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Check cache first
        cache_key = f"{self._get_article_id(article1)}_{self._get_article_id(article2)}"
        cached_similarity = self._get_cached_similarity(cache_key)
        if cached_similarity is not None:
            return cached_similarity
        
        # Calculate title similarity
        title1 = getattr(article1, 'title', '')
        title2 = getattr(article2, 'title', '')
        title_similarity = self._calculate_text_similarity(title1, title2)
        
        # Calculate content similarity
        content1 = getattr(article1, 'content', '')
        content2 = getattr(article2, 'content', '')
        content_similarity = self._calculate_text_similarity(content1, content2)
        
        # Calculate semantic similarity if Sentence-BERT is available
        semantic_similarity = 0.0
        if self.sentence_transformer and content1 and content2:
            try:
                # Use first 200 characters for semantic comparison
                text1 = content1[:200] if len(content1) > 200 else content1
                text2 = content2[:200] if len(content2) > 200 else content2
                
                embeddings = self.sentence_transformer.encode([text1, text2])
                semantic_similarity = cosine_similarity(
                    embeddings[0:1], embeddings[1:2]
                )[0][0]
                
                # Ensure semantic similarity is positive
                semantic_similarity = max(0.0, semantic_similarity)
                
            except Exception as e:
                logger.warning(f"Semantic similarity calculation failed: {e}")
                semantic_similarity = 0.0
        
        # Weighted combination
        if semantic_similarity > 0:
            # With semantic similarity: title (30%) + content (20%) + semantic (50%)
            overall_similarity = (title_similarity * 0.3) + (content_similarity * 0.2) + (semantic_similarity * 0.5)
        else:
            # Without semantic similarity: title (60%) + content (40%)
            overall_similarity = (title_similarity * 0.6) + (content_similarity * 0.4)
        
        # Cache the result
        self._cache_similarity(cache_key, overall_similarity)
        
        return overall_similarity
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using word overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_credibility_boost(self, 
                                   similar_articles: List[Dict[str, Any]], 
                                   cross_source_count: int,
                                   total_similarity: float,
                                   prefer_sources: Optional[List[str]] = None) -> float:
        """
        Calculate credibility boost based on cross-referencing.
        
        Args:
            similar_articles: List of similar articles
            cross_source_count: Number of cross-source matches
            total_similarity: Total similarity score
            prefer_sources: Preferred source order
            
        Returns:
            Credibility boost (0.0 to 1.0)
        """
        if cross_source_count == 0:
            return 0.0
        
        # Base boost from cross-source coverage
        base_boost = min(0.5, cross_source_count * 0.1)
        
        # Similarity boost
        avg_similarity = total_similarity / cross_source_count
        similarity_boost = avg_similarity * 0.3
        
        # Source preference boost
        preference_boost = 0.0
        if prefer_sources and similar_articles:
            preferred_matches = sum(1 for article in similar_articles 
                                  if article['source'] in prefer_sources)
            if preferred_matches > 0:
                preference_boost = min(0.2, preferred_matches * 0.05)
        
        # Total credibility boost
        total_boost = base_boost + similarity_boost + preference_boost
        
        return min(1.0, total_boost)
    
    def _determine_verification_badge(self, 
                                    similar_articles: List[Dict[str, Any]], 
                                    cross_source_count: int,
                                    credibility_boost: float) -> str:
        """
        Determine verification badge based on cross-referencing.
        
        Args:
            similar_articles: List of similar articles
            cross_source_count: Number of cross-source matches
            credibility_boost: Credibility boost score
            
        Returns:
            Verification badge string
        """
        if cross_source_count >= 3:
            return "✅ Verified by multiple sources"
        elif cross_source_count >= 2:
            return "✅ Verified by 2+ sources"
        elif cross_source_count == 1:
            return "⚠️ Partially verified"
        else:
            return "⚠️ Single source"
    
    def _determine_evidence_strength(self, 
                                   similar_articles: List[Dict[str, Any]], 
                                   cross_source_count: int,
                                   credibility_boost: float) -> str:
        """
        Determine evidence strength based on cross-referencing.
        
        Args:
            similar_articles: List of similar articles
            cross_source_count: Number of cross-source matches
            credibility_boost: Credibility boost score
            
        Returns:
            Evidence strength string
        """
        if credibility_boost >= 0.7:
            return "Strong"
        elif credibility_boost >= 0.4:
            return "Moderate"
        elif cross_source_count > 0:
            return "Weak"
        elif similar_articles:
            # Single source but has some similarity
            return "Weak (Single Source)"
        else:
            return "Very Weak (Single Source)"
    
    def _group_articles_by_source(self, articles: List[Any]) -> Dict[str, List[Any]]:
        """Group articles by source name."""
        source_groups = {}
        for article in articles:
            source = getattr(article, 'source_name', 'Unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(article)
        return source_groups
    
    def _get_article_id(self, article: Any) -> str:
        """Generate a unique ID for an article."""
        title = getattr(article, 'title', '')
        url = getattr(article, 'url', '')
        source = getattr(article, 'source_name', 'Unknown')
        
        # Create hash from title + url + source
        content = f"{title}{url}{source}".encode('utf-8')
        return hashlib.md5(content).hexdigest()[:16]
    
    def _generate_cache_key(self, query: str, articles: List[Any]) -> str:
        """Generate cache key for cross-reference results."""
        # Create hash from query and article IDs
        article_ids = [self._get_article_id(article) for article in articles]
        content = f"{query}{''.join(sorted(article_ids))}".encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[CrossReferenceScore]]:
        """Get cached cross-reference result."""
        if not self.cache_conn:
            return None
        
        try:
            cursor = self.cache_conn.cursor()
            cursor.execute("""
                SELECT results_json, timestamp FROM cross_reference_cache 
                WHERE query_hash = ? AND timestamp > datetime('now', '-1 hour')
            """, (cache_key,))
            
            result = cursor.fetchone()
            if result:
                import json
                return json.loads(result[0])
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, results: List[CrossReferenceScore]):
        """Cache cross-reference results."""
        if not self.cache_conn:
            return
        
        try:
            import json
            results_json = json.dumps([self._cross_reference_score_to_dict(score) for score in results])
            
            cursor = self.cache_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO cross_reference_cache (query_hash, results_json, timestamp)
                VALUES (?, ?, datetime('now'))
            """, (cache_key, results_json))
            
            self.cache_conn.commit()
            
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _get_cached_similarity(self, cache_key: str) -> Optional[float]:
        """Get cached similarity score."""
        if not self.cache_conn:
            return None
        
        try:
            cursor = self.cache_conn.cursor()
            cursor.execute("""
                SELECT similarity_score FROM similarity_cache 
                WHERE query_hash = ? AND timestamp > datetime('now', '-24 hours')
            """, (cache_key,))
            
            result = cursor.fetchone()
            if result:
                score = result[0]
                # Fix bytes vs float issue
                if isinstance(score, bytes):
                    score = float(score.decode("utf-8"))
                elif isinstance(score, (int, float)):
                    score = float(score)
                else:
                    logger.warning(f"Unexpected score type: {type(score)}, value: {score}")
                    return None
                return score
            
        except Exception as e:
            logger.warning(f"Similarity cache retrieval failed: {e}")
        
        return None
    
    def _cache_similarity(self, cache_key: str, similarity: float):
        """Cache similarity score."""
        if not self.cache_conn:
            return
        
        try:
            cursor = self.cache_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO similarity_cache (query_hash, similarity_score, timestamp)
                VALUES (?, ?, datetime('now'))
            """, (cache_key, similarity))
            
            self.cache_conn.commit()
            
        except Exception as e:
            logger.warning(f"Similarity cache storage failed: {e}")
    
    def _cross_reference_score_to_dict(self, score: CrossReferenceScore) -> Dict[str, Any]:
        """Convert CrossReferenceScore to dictionary for JSON serialization."""
        return {
            'article_id': score.article_id,
            'source_name': score.source_name,
            'similarity_score': float(score.similarity_score),  # Ensure float
            'matching_articles': score.matching_articles,
            'credibility_boost': float(score.credibility_boost),  # Ensure float
            'verification_badge': score.verification_badge,
            'evidence_strength': score.evidence_strength
        }
    
    def get_structured_output(self, scores: List[CrossReferenceScore]) -> List[Dict[str, Any]]:
        """
        Get structured output for cross-reference results.
        
        Args:
            scores: List of cross-reference scores
            
        Returns:
            List of structured dictionaries
        """
        structured_results = []
        
        for score in scores:
            # Get article details from matching articles
            article_details = []
            for match in score.matching_articles:
                article_details.append({
                    'title': match.get('title', ''),
                    'source': match.get('source', ''),
                    'similarity': float(match.get('similarity', 0.0)),  # Ensure float
                    'is_cross_source': match.get('is_cross_source', False)
                })
            
            structured_result = {
                'title': f"Article from {score.source_name}",  # Placeholder title
                'source': score.source_name,
                'similarity_score': float(score.similarity_score),
                'verification_badge': score.verification_badge,
                'evidence_strength': score.evidence_strength,
                'credibility_boost': float(score.credibility_boost),
                'matching_articles': article_details,
                'cross_source_count': len([m for m in score.matching_articles if m.get('is_cross_source', False)])
            }
            
            structured_results.append(structured_result)
        
        return structured_results
    
    def get_cross_reference_summary(self, scores: List[CrossReferenceScore]) -> Dict[str, Any]:
        """
        Generate summary of cross-referencing results.
        
        Args:
            scores: List of cross-reference scores
            
        Returns:
            Summary dictionary
        """
        if not scores:
            return {"message": "No cross-reference data available"}
        
        # Count verification badges
        badge_counts = {}
        strength_counts = {}
        total_boost = 0.0
        
        for score in scores:
            badge = score.verification_badge
            strength = score.evidence_strength
            
            badge_counts[badge] = badge_counts.get(badge, 0) + 1
            strength_counts[strength] = strength_counts.get(strength, 0) + 1
            total_boost += score.credibility_boost
        
        avg_boost = total_boost / len(scores)
        
        # Calculate source diversity
        sources = set(score.source_name for score in scores)
        source_diversity = len(sources)
        
        return {
            "total_articles": len(scores),
            "source_diversity": source_diversity,
            "average_credibility_boost": round(avg_boost, 3),
            "verification_badges": badge_counts,
            "evidence_strength": strength_counts,
            "cross_reference_coverage": sum(1 for s in scores if s.credibility_boost > 0),
            "strong_evidence_count": sum(1 for s in scores if s.evidence_strength == "Strong"),
            "moderate_evidence_count": sum(1 for s in scores if s.evidence_strength == "Moderate")
        }
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache entries."""
        if not self.cache_conn:
            return
        
        try:
            cursor = self.cache_conn.cursor()
            
            # Clean similarity cache
            cursor.execute("""
                DELETE FROM similarity_cache 
                WHERE timestamp < datetime('now', '-{} hours')
            """.format(max_age_hours))
            
            # Clean cross-reference cache
            cursor.execute("""
                DELETE FROM cross_reference_cache 
                WHERE timestamp < datetime('now', '-{} hours')
            """.format(max_age_hours))
            
            self.cache_conn.commit()
            logger.info("Cache cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.cache_conn:
            self.cache_conn.close()
