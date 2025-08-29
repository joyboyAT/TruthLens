#!/usr/bin/env python3
"""
Guardian API Handler for TruthLens
Fetches news articles from The Guardian API for cross-referencing with News API.
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class GuardianAPIHandler:
    """
    Handler for The Guardian API to fetch news articles.
    
    API Documentation: https://open-platform.theguardian.com/documentation/
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Guardian API handler.
        
        Args:
            api_key: Guardian API key
        """
        self.api_key = api_key
        self.base_url = "https://content.guardianapis.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TruthLens/1.0 (Fact-Checking System)'
        })
        
        # Rate limiting
        self.request_delay = 0.5  # 500ms between requests
        self.last_request_time = 0
        
        # Test API key
        self._test_api_key()
    
    def _test_api_key(self):
        """Test if the Guardian API key is valid."""
        try:
            test_url = f"{self.base_url}/search"
            params = {
                'api-key': self.api_key,
                'q': 'test',
                'page-size': 1
            }
            
            response = self.session.get(test_url, params=params)
            if response.status_code != 200:
                raise ValueError(f"Guardian API test failed with status {response.status_code}")
            
            data = response.json()
            if data.get('response', {}).get('status') != 'ok':
                raise ValueError("Guardian API returned error status")
            
            logger.info("Guardian API key validated successfully")
            
        except Exception as e:
            logger.error(f"Guardian API key validation failed: {e}")
            raise
    
    def _rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def fetch_guardian_news(self, query: str, max_results: int = 10, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch news articles from The Guardian API.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            days_back: Number of days back to search
            
        Returns:
            List of Guardian news articles
        """
        try:
            self._rate_limit()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = f"{self.base_url}/search"
            params = {
                'api-key': self.api_key,
                'q': query,
                'page-size': min(max_results, 50),  # Guardian allows up to 50 per page
                'from-date': start_date.strftime('%Y-%m-%d'),
                'to-date': end_date.strftime('%Y-%m-%d'),
                'show-fields': 'headline,bodyText,lastModified,sectionName,webUrl,standfirst',
                'show-tags': 'contributor',
                'show-blocks': 'all',  # Get full article content
                'order-by': 'relevance'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Guardian API returned status {response.status_code}")
                return []
            
            data = response.json()
            response_data = data.get('response', {})
            
            if response_data.get('status') != 'ok':
                logger.warning(f"Guardian API error: {response_data.get('message', 'Unknown error')}")
                return []
            
            results = response_data.get('results', [])
            logger.info(f"Guardian API: found {len(results)} results for query '{query}'")
            
            # Parse and format results
            formatted_results = []
            for result in results[:max_results]:
                formatted_result = self._parse_guardian_result(result)
                if formatted_result:
                    formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error fetching Guardian news for query '{query}': {e}")
            return []
    
    def _parse_guardian_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a Guardian API result into our standard format.
        
        Args:
            result: Raw result from Guardian API
            
        Returns:
            Formatted article data or None if parsing fails
        """
        try:
            # Extract fields
            fields = result.get('fields', {})
            
            # Get headline (prefer webTitle, fallback to fields.headline)
            headline = result.get('webTitle') or fields.get('headline', 'No title')
            
            # Get body text
            body_text = fields.get('bodyText', '')
            
            # Get section
            section = result.get('sectionName', 'Unknown')
            
            # Get URL
            url = result.get('webUrl', '')
            
            # Get publication date
            pub_date = result.get('webPublicationDate', '')
            
            # Get contributors/authors
            tags = result.get('tags', [])
            contributors = [tag.get('webTitle', '') for tag in tags if tag.get('type') == 'contributor']
            author = ', '.join(contributors) if contributors else 'Guardian Staff'
            
            # Create formatted result
            formatted_result = {
                'title': headline,
                'description': body_text[:200] + '...' if len(body_text) > 200 else body_text,
                'content': body_text,
                'url': url,
                'source': 'The Guardian',
                'section': section,
                'author': author,
                'published_at': pub_date,
                'source_name': 'Guardian',  # For pipeline integration
                'guardian_id': result.get('id', ''),
                'api_type': 'guardian'
            }
            
            return formatted_result
            
        except Exception as e:
            logger.warning(f"Error parsing Guardian result: {e}")
            return None
    
    def search_multiple_queries(self, queries: List[str], max_results_per_query: int = 5) -> List[Dict[str, Any]]:
        """
        Search multiple queries and return combined results.
        
        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query
            
        Returns:
            Combined list of Guardian articles
        """
        all_results = []
        
        for query in queries:
            try:
                results = self.fetch_guardian_news(query, max_results_per_query)
                all_results.extend(results)
                
                # Add small delay between queries
                time.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"Error searching query '{query}': {e}")
                continue
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        logger.info(f"Combined {len(all_results)} Guardian results into {len(unique_results)} unique articles")
        return unique_results
    
    def get_article_content(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full article content by ID.
        
        Args:
            article_id: Guardian article ID
            
        Returns:
            Full article content or None if not found
        """
        try:
            self._rate_limit()
            
            url = f"{self.base_url}/{article_id}"
            params = {
                'api-key': self.api_key,
                'show-fields': 'headline,bodyText,lastModified,sectionName,webUrl,standfirst',
                'show-tags': 'contributor,keyword'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            response_data = data.get('response', {})
            
            if response_data.get('status') != 'ok':
                return None
            
            content = response_data.get('content')
            if not content:
                return None
            
            return self._parse_guardian_result(content)
            
        except Exception as e:
            logger.error(f"Error fetching article content for ID '{article_id}': {e}")
            return None
