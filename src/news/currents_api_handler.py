#!/usr/bin/env python3
"""
Currents API Handler for TruthLens
Provides access to Currents API as a free alternative news source.
"""

import logging
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CurrentsAPIHandler:
    """
    Handler for Currents API (free tier available).
    
    Features:
    - Free tier with 100 requests per day
    - Global news coverage
    - Multiple languages support
    - Real-time news updates
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Currents API handler.
        
        Args:
            api_key: Currents API key (optional for free tier)
        """
        self.api_key = api_key
        self.base_url = "https://api.currentsapi.services/v1"
        self.free_tier = api_key is None
        
        if self.free_tier:
            logger.info("Currents API handler initialized in free tier mode")
        else:
            logger.info("Currents API handler initialized with API key")
    
    def fetch_currents_news(self, query: str, max_results: int = 10, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch news from Currents API.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            days_back: Number of days back to search
            
        Returns:
            List of news articles
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Prepare parameters
            params = {
                'keywords': query,
                'language': 'en',
                'limit': min(max_results, 100),  # Free tier limit
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'published_desc'
            }
            
            # Add API key if available
            if self.api_key:
                params['apiKey'] = self.api_key
            
            # Make API request
            response = requests.get(f"{self.base_url}/search/latest", params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('news', [])
                
                # Convert to standard format
                converted_articles = []
                for article in articles[:max_results]:
                    converted_article = {
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'source': article.get('author', 'Currents'),
                        'published_at': article.get('published', ''),
                        'relevance_score': 0.7,  # Default relevance for Currents
                        'api_type': 'currents',
                        'language': article.get('language', 'en'),
                        'category': article.get('category', []),
                        'country': article.get('country', [])
                    }
                    converted_articles.append(converted_article)
                
                logger.info(f"Currents API returned {len(converted_articles)} articles for '{query}'")
                return converted_articles
                
            elif response.status_code == 429:
                logger.warning("Currents API rate limit hit")
                return []
            else:
                logger.warning(f"Currents API request failed with status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching Currents news: {e}")
            return []
    
    def get_trending_topics(self, category: str = None) -> List[str]:
        """
        Get trending topics from Currents API.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of trending topics
        """
        try:
            params = {
                'language': 'en',
                'limit': 20
            }
            
            if category:
                params['category'] = category
            
            if self.api_key:
                params['apiKey'] = self.api_key
            
            response = requests.get(f"{self.base_url}/trending", params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                topics = data.get('trending', [])
                return [topic.get('title', '') for topic in topics if topic.get('title')]
            else:
                logger.warning(f"Failed to get trending topics: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if Currents API is available."""
        try:
            # Simple availability check
            response = requests.get(f"{self.base_url}/search/latest", 
                                 params={'keywords': 'test', 'limit': 1}, 
                                 timeout=10)
            return response.status_code in [200, 429]  # 429 means rate limited but available
        except:
            return False
