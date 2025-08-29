#!/usr/bin/env python3
"""
News Handler for TruthLens
Uses News API to fetch and process news articles for fact-checking.
"""

import requests
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """News article structure."""
    title: str
    description: str
    content: str
    url: str
    source: str
    published_at: str
    relevance_score: float

class NewsHandler:
    """Handler for News API integration."""
    
    def __init__(self, api_key: str):
        """
        Initialize the News API handler.
        
        Args:
            api_key: News API key
        """
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.session = requests.Session()
        
        # Test API key
        self._test_api_key()
    
    def _test_api_key(self):
        """Test if the News API key is valid."""
        try:
            test_url = f"{self.base_url}/top-headlines"
            params = {
                'apiKey': self.api_key,
                'country': 'us',
                'pageSize': 1
            }
            
            response = self.session.get(test_url, params=params)
            
            if response.status_code == 200:
                logger.info("News API key is valid")
            elif response.status_code == 401:
                logger.error("News API key is invalid")
                raise ValueError("Invalid News API key")
            else:
                logger.warning(f"News API test returned status code: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error testing News API key: {e}")
            raise
    
    def search_news(self, query: str, max_results: int = 10, days_back: int = 30) -> List[NewsArticle]:
        """
        Search for news articles using the News API.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            days_back: Number of days back to search
            
        Returns:
            List of NewsArticle objects
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = f"{self.base_url}/everything"
            params = {
                'apiKey': self.api_key,
                'q': query,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'pageSize': min(max_results, 100),  # API limit is 100
                'language': 'en'
            }
            
            logger.info(f"Searching News API for: {query}")
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                total_results = data.get('totalResults', 0)
                
                logger.info(f"Found {len(articles)} news articles (total: {total_results})")
                
                # Convert to NewsArticle objects
                news_articles = []
                for article in articles:
                    news_article = NewsArticle(
                        title=article.get('title', ''),
                        description=article.get('description', ''),
                        content=article.get('content', ''),
                        url=article.get('url', ''),
                        source=article.get('source', {}).get('name', 'Unknown'),
                        published_at=article.get('publishedAt', ''),
                        relevance_score=self._calculate_relevance_score(query, article)
                    )
                    news_articles.append(news_article)
                
                # Sort by relevance score
                news_articles.sort(key=lambda x: x.relevance_score, reverse=True)
                return news_articles[:max_results]
                
            else:
                logger.error(f"News API request failed with status {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching news: {e}")
            return []
    
    def _calculate_relevance_score(self, query: str, article: Dict[str, Any]) -> float:
        """Calculate relevance score for an article based on the query."""
        query_lower = query.lower()
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = article.get('content', '').lower()
        
        # Count query terms in different fields
        title_matches = sum(1 for term in query_lower.split() if term in title)
        desc_matches = sum(1 for term in query_lower.split() if term in description)
        content_matches = sum(1 for term in query_lower.split() if term in content)
        
        # Weighted scoring
        score = (title_matches * 3) + (desc_matches * 2) + (content_matches * 1)
        
        # Normalize score
        max_possible = len(query_lower.split()) * 6  # 3+2+1 weights
        if max_possible > 0:
            score = score / max_possible
        
        return min(score, 1.0)
    
    def extract_claims_from_news(self, news_articles: List[NewsArticle]) -> List[str]:
        """
        Extract potential claims from news articles.
        
        Args:
            news_articles: List of news articles
            
        Returns:
            List of potential claims
        """
        claims = []
        
        for article in news_articles:
            # Extract claims from title and description
            text_to_analyze = f"{article.title}. {article.description}"
            
            # Simple claim extraction using keyword patterns
            claim_indicators = [
                "claims", "alleges", "reports", "says", "announces", "reveals",
                "discovers", "finds", "confirms", "denies", "admits", "confesses",
                "according to", "sources say", "officials say", "experts say"
            ]
            
            sentences = text_to_analyze.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if any(indicator in sentence.lower() for indicator in claim_indicators):
                    if len(sentence) > 10:  # Minimum length for a claim
                        claims.append(sentence)
            
            # If no claims found, use the title as a potential claim
            if not claims and len(article.title) > 10:
                claims.append(article.title)
        
        return claims[:5]  # Limit to top 5 claims
    
    def get_news_context(self, query: str, max_articles: int = 5) -> Dict[str, Any]:
        """
        Get news context for a query.
        
        Args:
            query: Search query
            max_articles: Maximum number of articles to fetch
            
        Returns:
            Dictionary with news context
        """
        try:
            # Search for news articles
            articles = self.search_news(query, max_articles)
            
            if not articles:
                return {
                    "query": query,
                    "articles_found": 0,
                    "claims": [],
                    "context": "No recent news articles found for this query."
                }
            
            # Extract claims
            claims = self.extract_claims_from_news(articles)
            
            # Create context summary
            context_parts = []
            for i, article in enumerate(articles[:3], 1):
                context_parts.append(f"{i}. {article.title} ({article.source})")
            
            context = f"Recent news coverage:\n" + "\n".join(context_parts)
            
            return {
                "query": query,
                "articles_found": len(articles),
                "claims": claims,
                "context": context,
                "articles": [
                    {
                        "title": article.title,
                        "description": article.description,
                        "url": article.url,
                        "source": article.source,
                        "published_at": article.published_at,
                        "relevance_score": article.relevance_score
                    }
                    for article in articles
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting news context: {e}")
            return {
                "query": query,
                "articles_found": 0,
                "claims": [],
                "context": f"Error retrieving news: {str(e)}"
            }

def create_news_handler(api_key: str) -> NewsHandler:
    """Factory function to create a News API handler."""
    return NewsHandler(api_key)
