#!/usr/bin/env python3
"""
Enhanced Fact-Checking API Integration for TruthLens
Integrates multiple fact-checking sources for comprehensive verification.
"""

import requests
import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

@dataclass
class EnhancedFactCheckResult:
    """Enhanced fact-check result from multiple sources."""
    claim_text: str
    verdict: str  # "SUPPORTED", "REFUTED", "MIXED", "NOT ENOUGH INFO"
    confidence: float
    sources: List[Dict[str, Any]]  # Multiple fact-check sources
    best_source: Dict[str, Any]  # Most reliable source
    review_date: str
    explanation: str
    rating: str
    url: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the fact-check result to a dictionary."""
        return {
            'claim_text': self.claim_text,
            'verdict': self.verdict,
            'confidence': self.confidence,
            'sources': self.sources,
            'best_source': self.best_source,
            'review_date': self.review_date,
            'explanation': self.explanation,
            'rating': self.rating,
            'url': self.url
        }

class EnhancedFactCheckAPI:
    """
    Enhanced fact-checking API that integrates multiple sources.
    
    Sources:
    - Google Fact Check API
    - Snopes (web scraping)
    - PolitiFact (web scraping)
    - Science Feedback (web scraping)
    - AltNews (web scraping)
    """
    
    def __init__(self, google_api_key: str = None):
        """
        Initialize the enhanced fact-checking API.
        
        Args:
            google_api_key: Google Cloud API key with Fact Check API enabled
        """
        self.google_api_key = google_api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Rate limiting
        self.request_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
        
        # Initialize Google Fact Check API if key provided
        if google_api_key:
            try:
                self._test_google_api()
                logger.info("Google Fact Check API initialized")
            except Exception as e:
                logger.warning(f"Google Fact Check API not available: {e}")
                self.google_api_key = None
    
    def _test_google_api(self):
        """Test if the Google API key is valid."""
        if not self.google_api_key:
            return
        
        test_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            'key': self.google_api_key,
            'query': 'test',
            'maxAgeDays': 365,
            'pageSize': 1
        }
        
        response = self.session.get(test_url, params=params)
        if response.status_code != 200:
            raise ValueError(f"Google API test failed with status {response.status_code}")
    
    def search_claims(self, query: str, max_results: int = 10) -> List[EnhancedFactCheckResult]:
        """
        Search for fact-checked claims across multiple sources.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of enhanced fact-check results
        """
        all_results = []
        
        # Search Google Fact Check API
        if self.google_api_key:
            google_results = self._search_google_factcheck(query, max_results)
            all_results.extend(google_results)
        
        # Search Snopes
        snopes_results = self._search_snopes(query, max_results)
        all_results.extend(snopes_results)
        
        # Search PolitiFact
        politifact_results = self._search_politifact(query, max_results)
        all_results.extend(politifact_results)
        
        # Search Science Feedback
        science_results = self._search_science_feedback(query, max_results)
        all_results.extend(science_results)
        
        # Deduplicate and rank results
        unique_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(unique_results)
        
        return ranked_results[:max_results]
    
    def _search_google_factcheck(self, query: str, max_results: int) -> List[EnhancedFactCheckResult]:
        """Search Google Fact Check API."""
        if not self.google_api_key:
            return []
        
        try:
            self._rate_limit()
            
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            params = {
                'key': self.google_api_key,
                'query': query,
                'maxAgeDays': 365,
                'pageSize': min(max_results, 20),
                'languageCode': 'en'
            }
            
            response = self.session.get(url, params=params)
            if response.status_code != 200:
                logger.warning(f"Google Fact Check API returned status {response.status_code}")
                return []
            
            data = response.json()
            claims = data.get('claims', [])
            
            results = []
            for claim in claims:
                try:
                    result = self._parse_google_claim(claim)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Error parsing Google claim: {e}")
                    continue
            
            logger.info(f"Google Fact Check: found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching Google Fact Check: {e}")
            return []
    
    def _search_snopes(self, query: str, max_results: int) -> List[EnhancedFactCheckResult]:
        """Search Snopes for fact-checked claims."""
        try:
            self._rate_limit()
            
            # Snopes search URL
            search_url = f"https://www.snopes.com/search/?q={query.replace(' ', '+')}"
            
            response = self.session.get(search_url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Snopes search returned status {response.status_code}")
                return []
            
            # Parse search results (simplified)
            results = self._parse_snopes_search(response.text, query)
            
            logger.info(f"Snopes: found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching Snopes: {e}")
            return []
    
    def _search_politifact(self, query: str, max_results: int) -> List[EnhancedFactCheckResult]:
        """Search PolitiFact for fact-checked claims."""
        try:
            self._rate_limit()
            
            # PolitiFact search URL
            search_url = f"https://www.politifact.com/search/?q={query.replace(' ', '+')}"
            
            response = self.session.get(search_url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"PolitiFact search returned status {response.status_code}")
                return []
            
            # Parse search results (simplified)
            results = self._parse_politifact_search(response.text, query)
            
            logger.info(f"PolitiFact: found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching PolitiFact: {e}")
            return []
    
    def _search_science_feedback(self, query: str, max_results: int) -> List[EnhancedFactCheckResult]:
        """Search Science Feedback for fact-checked claims."""
        try:
            self._rate_limit()
            
            # Science Feedback search URL
            search_url = f"https://sciencefeedback.co/search/?q={query.replace(' ', '+')}"
            
            response = self.session.get(search_url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Science Feedback search returned status {response.status_code}")
                return []
            
            # Parse search results (simplified)
            results = self._parse_science_feedback_search(response.text, query)
            
            logger.info(f"Science Feedback: found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching Science Feedback: {e}")
            return []
    
    def _parse_google_claim(self, claim: Dict[str, Any]) -> Optional[EnhancedFactCheckResult]:
        """Parse a Google Fact Check claim into our format."""
        try:
            claim_text = claim.get('text', '')
            claim_review = claim.get('claimReview', [{}])[0]
            
            verdict = self._normalize_verdict(claim_review.get('textualRating', ''))
            confidence = self._calculate_confidence(verdict, claim_review)
            
            source = {
                'name': 'Google Fact Check',
                'url': claim_review.get('url', ''),
                'publisher': claim_review.get('publisher', {}).get('name', 'Unknown'),
                'review_date': claim_review.get('reviewDate', ''),
                'rating': claim_review.get('textualRating', ''),
                'explanation': claim_review.get('textualRating', '')
            }
            
            return EnhancedFactCheckResult(
                claim_text=claim_text,
                verdict=verdict,
                confidence=confidence,
                sources=[source],
                best_source=source,
                review_date=source['review_date'],
                explanation=source['explanation'],
                rating=source['rating'],
                url=source['url']
            )
            
        except Exception as e:
            logger.warning(f"Error parsing Google claim: {e}")
            return None
    
    def _parse_snopes_search(self, html: str, query: str) -> List[EnhancedFactCheckResult]:
        """Parse Snopes search results (simplified implementation)."""
        results = []
        
        # This is a simplified parser - in production you'd use BeautifulSoup
        # Look for fact-check URLs in the HTML
        fact_check_pattern = r'href="(/fact-check/[^"]+)"'
        matches = re.findall(fact_check_pattern, html)
        
        for match in matches[:5]:  # Limit to 5 results
            try:
                # Create a basic result (in production you'd fetch the actual fact-check page)
                result = EnhancedFactCheckResult(
                    claim_text=query,
                    verdict="NOT ENOUGH INFO",  # Would need to parse actual page
                    confidence=0.5,
                    sources=[{
                        'name': 'Snopes',
                        'url': f"https://www.snopes.com{match}",
                        'publisher': 'Snopes',
                        'review_date': datetime.now().strftime('%Y-%m-%d'),
                        'rating': 'Unknown',
                        'explanation': 'Would need to parse actual fact-check page'
                    }],
                    best_source={
                        'name': 'Snopes',
                        'url': f"https://www.snopes.com{match}",
                        'publisher': 'Snopes',
                        'review_date': datetime.now().strftime('%Y-%m-%d'),
                        'rating': 'Unknown',
                        'explanation': 'Would need to parse actual fact-check page'
                    },
                    review_date=datetime.now().strftime('%Y-%m-%d'),
                    explanation='Would need to parse actual fact-check page',
                    rating='Unknown',
                    url=f"https://www.snopes.com{match}"
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Error parsing Snopes result: {e}")
                continue
        
        return results
    
    def _parse_politifact_search(self, html: str, query: str) -> List[EnhancedFactCheckResult]:
        """Parse PolitiFact search results (simplified implementation)."""
        results = []
        
        # Simplified parser for PolitiFact
        fact_check_pattern = r'href="(/factcheck/[^"]+)"'
        matches = re.findall(fact_check_pattern, html)
        
        for match in matches[:5]:
            try:
                result = EnhancedFactCheckResult(
                    claim_text=query,
                    verdict="NOT ENOUGH INFO",
                    confidence=0.5,
                    sources=[{
                        'name': 'PolitiFact',
                        'url': f"https://www.politifact.com{match}",
                        'publisher': 'PolitiFact',
                        'review_date': datetime.now().strftime('%Y-%m-%d'),
                        'rating': 'Unknown',
                        'explanation': 'Would need to parse actual fact-check page'
                    }],
                    best_source={
                        'name': 'PolitiFact',
                        'url': f"https://www.politifact.com{match}",
                        'publisher': 'PolitiFact',
                        'review_date': datetime.now().strftime('%Y-%m-%d'),
                        'rating': 'Unknown',
                        'explanation': 'Would need to parse actual fact-check page'
                    },
                    review_date=datetime.now().strftime('%Y-%m-%d'),
                    explanation='Would need to parse actual fact-check page',
                    rating='Unknown',
                    url=f"https://www.politifact.com{match}"
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Error parsing PolitiFact result: {e}")
                continue
        
        return results
    
    def _parse_science_feedback_search(self, html: str, query: str) -> List[EnhancedFactCheckResult]:
        """Parse Science Feedback search results (simplified implementation)."""
        results = []
        
        # Simplified parser for Science Feedback
        fact_check_pattern = r'href="(/fact-check/[^"]+)"'
        matches = re.findall(fact_check_pattern, html)
        
        for match in matches[:5]:
            try:
                result = EnhancedFactCheckResult(
                    claim_text=query,
                    verdict="NOT ENOUGH INFO",
                    confidence=0.5,
                    sources=[{
                        'name': 'Science Feedback',
                        'url': f"https://sciencefeedback.co{match}",
                        'publisher': 'Science Feedback',
                        'review_date': datetime.now().strftime('%Y-%m-%d'),
                        'rating': 'Unknown',
                        'explanation': 'Would need to parse actual fact-check page'
                    }],
                    best_source={
                        'name': 'Science Feedback',
                        'url': f"https://sciencefeedback.co{match}",
                        'publisher': 'Science Feedback',
                        'review_date': datetime.now().strftime('%Y-%m-%d'),
                        'rating': 'Unknown',
                        'explanation': 'Would need to parse actual fact-check page'
                    },
                    review_date=datetime.now().strftime('%Y-%m-%d'),
                    explanation='Would need to parse actual fact-check page',
                    rating='Unknown',
                    url=f"https://sciencefeedback.co{match}"
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Error parsing Science Feedback result: {e}")
                continue
        
        return results
    
    def _normalize_verdict(self, rating: str) -> str:
        """Normalize verdict ratings across different sources."""
        rating_lower = rating.lower()
        
        if any(word in rating_lower for word in ['true', 'correct', 'accurate', 'supported']):
            return "SUPPORTED"
        elif any(word in rating_lower for word in ['false', 'incorrect', 'inaccurate', 'refuted', 'debunked']):
            return "REFUTED"
        elif any(word in rating_lower for word in ['mixed', 'half', 'mostly', 'partially']):
            return "MIXED"
        else:
            return "NOT ENOUGH INFO"
    
    def _calculate_confidence(self, verdict: str, claim_review: Dict[str, Any]) -> float:
        """Calculate confidence score based on verdict and source reliability."""
        base_confidence = {
            "SUPPORTED": 0.8,
            "REFUTED": 0.8,
            "MIXED": 0.6,
            "NOT ENOUGH INFO": 0.5
        }
        
        confidence = base_confidence.get(verdict, 0.5)
        
        # Boost confidence for Google Fact Check (most reliable)
        if claim_review.get('publisher', {}).get('name') in ['Google Fact Check', 'Snopes', 'PolitiFact']:
            confidence = min(0.95, confidence + 0.1)
        
        return confidence
    
    def _deduplicate_results(self, results: List[EnhancedFactCheckResult]) -> List[EnhancedFactCheckResult]:
        """Remove duplicate fact-check results."""
        seen_claims = set()
        unique_results = []
        
        for result in results:
            # Create a simple hash of the claim text
            claim_hash = hash(result.claim_text.lower())
            
            if claim_hash not in seen_claims:
                seen_claims.add(claim_hash)
                unique_results.append(result)
        
        logger.info(f"Deduplicated {len(results)} results to {len(unique_results)} unique results")
        return unique_results
    
    def _rank_results(self, results: List[EnhancedFactCheckResult]) -> List[EnhancedFactCheckResult]:
        """Rank results by confidence and source reliability."""
        def rank_key(result):
            # Prioritize by confidence, then by source reliability
            source_rank = {
                'Google Fact Check': 3,
                'Snopes': 2,
                'PolitiFact': 2,
                'Science Feedback': 1,
                'AltNews': 1
            }
            
            source_score = source_rank.get(result.best_source['name'], 0)
            return (result.confidence, source_score)
        
        ranked_results = sorted(results, key=rank_key, reverse=True)
        return ranked_results
    
    def _rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_best_fact_check(self, query: str) -> Optional[EnhancedFactCheckResult]:
        """
        Get the best fact-check result for a query.
        
        Args:
            query: The claim to fact-check
            
        Returns:
            Best fact-check result or None if none found
        """
        results = self.search_claims(query, max_results=5)
        if results:
            return results[0]  # Already ranked by confidence and source reliability
        return None
