#!/usr/bin/env python3
"""
Enhanced Semantic Search for TruthLens
Implements improved article retrieval with semantic similarity and deduplication.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import re
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logging.warning("Sentence transformers not available, falling back to keyword search")

logger = logging.getLogger(__name__)

@dataclass
class EnhancedSearchResult:
    """Enhanced search result with semantic scoring and deduplication info."""
    article: Dict[str, Any]
    semantic_score: float
    content_hash: str
    cluster_id: Optional[str] = None
    similarity_group: Optional[str] = None

class EnhancedSemanticSearch:
    """
    Enhanced semantic search with improved article retrieval.
    
    Key improvements:
    - Semantic search using sentence-transformers/all-MiniLM-L6-v2
    - Article deduplication and clustering
    - Better relevance ranking
    - Content-based similarity detection
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the enhanced semantic search.
        
        Args:
            model_name: Sentence transformer model to use
        """
        self.model_name = model_name
        self.semantic_model = None
        
        if SEMANTIC_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer(model_name)
                logger.info(f"Loaded semantic model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load semantic model: {e}")
                self.semantic_model = None
        
        # Similarity thresholds
        self.similarity_threshold = 0.85  # Articles above this are considered similar
        self.min_content_length = 100  # Minimum content length for meaningful analysis
        
        # Clustering parameters
        self.max_cluster_size = 5  # Maximum articles per cluster
        self.min_cluster_similarity = 0.8  # Minimum similarity for clustering
    
    def search_and_rank_articles(self, 
                                claim: str, 
                                articles: List[Dict[str, Any]], 
                                max_results: int = 20) -> List[EnhancedSearchResult]:
        """
        Search and rank articles using enhanced semantic similarity.
        
        Args:
            claim: The claim to search for
            articles: List of articles to analyze
            max_results: Maximum number of results to return
            
        Returns:
            List of enhanced search results ranked by relevance
        """
        if not articles:
            return []
        
        # Step 1: Preprocess articles
        processed_articles = self._preprocess_articles(articles)
        
        # Step 2: Calculate semantic similarity if model available
        if self.semantic_model and processed_articles:
            processed_articles = self._add_semantic_scores(claim, processed_articles)
        
        # Step 3: Deduplicate articles
        unique_articles = self._deduplicate_articles(processed_articles)
        
        # Step 4: Cluster similar articles
        clustered_articles = self._cluster_articles(unique_articles)
        
        # Step 5: Rank by relevance
        ranked_articles = self._rank_by_relevance(clustered_articles)
        
        return ranked_articles[:max_results]
    
    def _preprocess_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess articles for better analysis."""
        processed = []
        
        for article in articles:
            try:
                # Extract and clean content
                title = article.get('title', '').strip()
                description = article.get('description', '').strip()
                content = article.get('content', '').strip()
                
                # Skip articles with insufficient content
                total_length = len(title) + len(description) + len(content)
                if total_length < self.min_content_length:
                    continue
                
                # Create content hash for deduplication
                content_text = f"{title} {description} {content}".lower()
                content_hash = hashlib.md5(content_text.encode()).hexdigest()
                
                # Clean and normalize text
                cleaned_title = self._clean_text(title)
                cleaned_description = self._clean_text(description)
                cleaned_content = self._clean_text(content)
                
                processed_article = {
                    'title': cleaned_title,
                    'description': cleaned_description,
                    'content': cleaned_content,
                    'original_article': article,
                    'content_hash': content_hash,
                    'total_length': total_length
                }
                
                processed.append(processed_article)
                
            except Exception as e:
                logger.warning(f"Error preprocessing article: {e}")
                continue
        
        logger.info(f"Preprocessed {len(articles)} articles to {len(processed)} valid articles")
        return processed
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better semantic analysis."""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:]', '', text)
        
        return text.strip()
    
    def _add_semantic_scores(self, claim: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add semantic similarity scores to articles."""
        try:
            # Prepare texts for embedding
            claim_text = claim.lower()
            article_texts = []
            
            for article in articles:
                text = f"{article['title']} {article['description']} {article['content']}"
                article_texts.append(text)
            
            # Get embeddings
            texts = [claim_text] + article_texts
            embeddings = self.semantic_model.encode(texts)
            
            # Calculate similarities
            claim_embedding = embeddings[0].reshape(1, -1)
            article_embeddings = embeddings[1:]
            
            similarities = cosine_similarity(claim_embedding, article_embeddings)[0]
            
            # Add semantic scores to articles
            for i, article in enumerate(articles):
                article['semantic_score'] = float(similarities[i])
            
            logger.info(f"Added semantic scores to {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error calculating semantic scores: {e}")
            # Fallback: assign random scores
            for article in articles:
                article['semantic_score'] = 0.5
            return articles
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on content hash."""
        seen_hashes = set()
        unique_articles = []
        
        for article in articles:
            content_hash = article.get('content_hash', '')
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_articles.append(article)
        
        logger.info(f"Deduplicated {len(articles)} articles to {len(unique_articles)} unique articles")
        return unique_articles
    
    def _cluster_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster similar articles to avoid redundancy."""
        if not articles or len(articles) < 2:
            return articles
        
        # Sort by semantic score for better clustering
        sorted_articles = sorted(articles, key=lambda x: x.get('semantic_score', 0), reverse=True)
        
        clusters = []
        used_indices = set()
        
        for i, article in enumerate(sorted_articles):
            if i in used_indices:
                continue
            
            # Start new cluster
            cluster = [article]
            used_indices.add(i)
            cluster_id = f"cluster_{len(clusters)}"
            
            # Find similar articles
            for j, other_article in enumerate(sorted_articles[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Check similarity
                if self._are_articles_similar(article, other_article):
                    cluster.append(other_article)
                    used_indices.add(j)
                    
                    # Limit cluster size
                    if len(cluster) >= self.max_cluster_size:
                        break
            
            # Mark cluster members
            for cluster_article in cluster:
                cluster_article['cluster_id'] = cluster_id
                cluster_article['cluster_size'] = len(cluster)
            
            clusters.append(cluster)
        
        # Select best article from each cluster
        clustered_articles = []
        for cluster in clusters:
            # Select article with highest semantic score
            best_article = max(cluster, key=lambda x: x.get('semantic_score', 0))
            best_article['similarity_group'] = f"Similar to {len(cluster)-1} other articles"
            clustered_articles.append(best_article)
        
        logger.info(f"Clustered {len(articles)} articles into {len(clusters)} clusters")
        return clustered_articles
    
    def _are_articles_similar(self, article1: Dict[str, Any], article2: Dict[str, Any]) -> bool:
        """Check if two articles are similar enough to cluster."""
        # Check semantic similarity if available
        if 'semantic_score' in article1 and 'semantic_score' in article2:
            # Use semantic scores as a proxy for similarity
            score1 = article1.get('semantic_score', 0)
            score2 = article2.get('semantic_score', 0)
            
            # If both have high scores, they might be similar
            if score1 > 0.7 and score2 > 0.7:
                return True
        
        # Check content similarity
        title1 = article1.get('title', '').lower()
        title2 = article2.get('title', '').lower()
        
        # Simple title similarity check
        if self._calculate_text_similarity(title1, title2) > 0.8:
            return True
        
        # Check description similarity
        desc1 = article1.get('description', '').lower()
        desc2 = article2.get('description', '').lower()
        
        if self._calculate_text_similarity(desc1, desc2) > 0.7:
            return True
        
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using word overlap."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _rank_by_relevance(self, articles: List[Dict[str, Any]]) -> List[EnhancedSearchResult]:
        """Rank articles by relevance to the claim."""
        ranked_results = []
        
        for article in articles:
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(article)
            
            result = EnhancedSearchResult(
                article=article['original_article'],
                semantic_score=article.get('semantic_score', 0.5),
                content_hash=article.get('content_hash', ''),
                cluster_id=article.get('cluster_id'),
                similarity_group=article.get('similarity_group')
            )
            
            # Add relevance score to the result
            result.article['relevance_score'] = relevance_score
            
            ranked_results.append(result)
        
        # Sort by relevance score
        ranked_results.sort(key=lambda x: x.article.get('relevance_score', 0), reverse=True)
        
        logger.info(f"Ranked {len(ranked_results)} articles by relevance")
        return ranked_results
    
    def _calculate_relevance_score(self, article: Dict[str, Any]) -> float:
        """Calculate overall relevance score for an article."""
        base_score = 0.0
        
        # Semantic score (if available)
        if 'semantic_score' in article:
            base_score += article['semantic_score'] * 0.6
        
        # Content length bonus
        content_length = article.get('total_length', 0)
        if content_length > 500:
            base_score += 0.1
        elif content_length > 200:
            base_score += 0.05
        
        # Cluster size penalty (avoid too many similar articles)
        cluster_size = article.get('cluster_size', 1)
        if cluster_size > 3:
            base_score -= 0.1
        
        # Normalize score
        return max(0.0, min(1.0, base_score))
    
    def get_search_summary(self, results: List[EnhancedSearchResult]) -> Dict[str, Any]:
        """Generate a summary of the search results."""
        if not results:
            return {"message": "No results found"}
        
        total_articles = len(results)
        avg_semantic_score = sum(r.semantic_score for r in results) / total_articles
        clustered_count = sum(1 for r in results if r.cluster_id)
        
        # Count similarity groups
        similarity_groups = defaultdict(int)
        for result in results:
            if result.similarity_group:
                similarity_groups[result.similarity_group] += 1
        
        return {
            "total_articles": total_articles,
            "average_semantic_score": round(avg_semantic_score, 3),
            "clustered_articles": clustered_count,
            "similarity_groups": dict(similarity_groups),
            "top_scores": [round(r.semantic_score, 3) for r in results[:5]],
            "search_quality": "High" if avg_semantic_score > 0.7 else "Medium" if avg_semantic_score > 0.5 else "Low"
        }
