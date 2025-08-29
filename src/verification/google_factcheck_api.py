#!/usr/bin/env python3
"""
Google Fact Check API Integration
Uses Google's Fact Check API to verify claims with real fact-checking data.
"""

import requests
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class FactCheckResult:
    """Result from Google Fact Check API."""
    claim_text: str
    verdict: str
    confidence: float
    publisher: str
    review_date: str
    url: str
    explanation: str
    rating: str
    claim_review_url: str

class GoogleFactCheckAPI:
    """Google Fact Check API client."""
    
    def __init__(self, api_key: str):
        """
        Initialize the Google Fact Check API client.
        
        Args:
            api_key: Google Cloud API key with Fact Check API enabled
        """
        self.api_key = api_key
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1"
        self.session = requests.Session()
        
        # Test API key
        self._test_api_key()
    
    def _test_api_key(self):
        """Test if the API key is valid."""
        try:
            test_url = f"{self.base_url}/claims:search"
            params = {
                'key': self.api_key,
                'query': 'test',
                'maxAgeDays': 365,
                'pageSize': 1
            }
            
            response = self.session.get(test_url, params=params)
            
            if response.status_code == 200:
                logger.info("Google Fact Check API key is valid")
            elif response.status_code == 403:
                logger.error("Google Fact Check API key is invalid or API not enabled")
                raise ValueError("Invalid API key or Fact Check API not enabled")
            else:
                logger.warning(f"API test returned status code: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error testing API key: {e}")
            raise
    
    def search_claims(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for fact-checked claims.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of fact-check results
        """
        try:
            url = f"{self.base_url}/claims:search"
            params = {
                'key': self.api_key,
                'query': query,
                'maxAgeDays': 365,  # Claims from the last year
                'pageSize': min(max_results, 20),  # API limit is 20
                'languageCode': 'en'
            }
            
            logger.info(f"Searching Google Fact Check API for: {query}")
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                claims = data.get('claims', [])
                logger.info(f"Found {len(claims)} fact-checked claims")
                return claims
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching claims: {e}")
            return []
    
    def verify_claim(self, claim_text: str) -> Optional[FactCheckResult]:
        """
        Verify a claim using Google Fact Check API.
        
        Args:
            claim_text: The claim to verify
            
        Returns:
            FactCheckResult if found, None otherwise
        """
        try:
            # Search for the claim
            claims = self.search_claims(claim_text, max_results=5)
            
            if not claims:
                logger.info("No fact-checked claims found")
                return None
            
            # Get the most relevant claim (first result)
            claim = claims[0]
            
            # Extract claim review
            claim_reviews = claim.get('claimReview', [])
            if not claim_reviews:
                logger.warning("No claim review found in the result")
                return None
            
            review = claim_reviews[0]
            
            # Extract publisher info
            publisher = review.get('publisher', {})
            publisher_name = publisher.get('name', 'Unknown')
            publisher_site = publisher.get('site', '')
            
            # Extract review details
            review_date = review.get('reviewDate', '')
            review_url = review.get('url', '')
            rating = review.get('textualRating', 'Unknown')
            
            # Determine verdict based on rating
            verdict = self._map_rating_to_verdict(rating)
            
            # Generate explanation
            explanation = self._generate_explanation(claim_text, rating, publisher_name)
            
            # Calculate confidence based on publisher reliability
            confidence = self._calculate_confidence(publisher_name)
            
            return FactCheckResult(
                claim_text=claim_text,
                verdict=verdict,
                confidence=confidence,
                publisher=publisher_name,
                review_date=review_date,
                url=review_url,
                explanation=explanation,
                rating=rating,
                claim_review_url=review_url
            )
            
        except Exception as e:
            logger.error(f"Error verifying claim: {e}")
            return None
    
    def _map_rating_to_verdict(self, rating: str) -> str:
        """Map Google's rating to our verdict format."""
        rating_lower = rating.lower()
        
        # False ratings
        if any(word in rating_lower for word in ['false', 'pants on fire', 'mostly false', 'false']):
            return "REFUTED"
        
        # True ratings
        elif any(word in rating_lower for word in ['true', 'mostly true', 'half true']):
            return "SUPPORTED"
        
        # Mixed or unclear ratings
        elif any(word in rating_lower for word in ['mixture', 'half true', 'unproven', 'unclear']):
            return "NOT ENOUGH INFO"
        
        # Default
        else:
            return "NOT ENOUGH INFO"
    
    def _generate_explanation(self, claim_text: str, rating: str, publisher: str) -> str:
        """Generate explanation for the verdict."""
        if rating.lower() in ['false', 'pants on fire', 'mostly false']:
            return f"This claim has been fact-checked and rated as '{rating}' by {publisher}. The claim appears to be false or misleading."
        elif rating.lower() in ['true', 'mostly true']:
            return f"This claim has been fact-checked and rated as '{rating}' by {publisher}. The claim appears to be supported by evidence."
        elif rating.lower() in ['mixture', 'half true']:
            return f"This claim has been fact-checked and rated as '{rating}' by {publisher}. The claim contains both true and false elements."
        else:
            return f"This claim has been reviewed by {publisher} with a rating of '{rating}'. More context may be needed for a complete assessment."
    
    def _calculate_confidence(self, publisher: str) -> float:
        """Calculate confidence based on publisher reliability."""
        # High-confidence publishers
        high_confidence_publishers = [
            'snopes', 'politifact', 'factcheck.org', 'reuters fact check',
            'associated press', 'bbc', 'cnn fact check'
        ]
        
        # Medium-confidence publishers
        medium_confidence_publishers = [
            'lead stories', 'usa today', 'washington post', 'new york times'
        ]
        
        publisher_lower = publisher.lower()
        
        if any(pub in publisher_lower for pub in high_confidence_publishers):
            return 0.9
        elif any(pub in publisher_lower for pub in medium_confidence_publishers):
            return 0.7
        else:
            return 0.5  # Default confidence for unknown publishers
    
    def batch_verify_claims(self, claims: List[str]) -> List[Optional[FactCheckResult]]:
        """
        Verify multiple claims in batch.
        
        Args:
            claims: List of claims to verify
            
        Returns:
            List of FactCheckResult objects (None for claims not found)
        """
        results = []
        
        for i, claim in enumerate(claims):
            logger.info(f"Verifying claim {i+1}/{len(claims)}: {claim[:50]}...")
            
            result = self.verify_claim(claim)
            results.append(result)
            
            # Rate limiting - wait between requests
            if i < len(claims) - 1:
                time.sleep(0.5)
        
        return results

def create_google_factcheck_client(api_key: str) -> GoogleFactCheckAPI:
    """Factory function to create a Google Fact Check API client."""
    return GoogleFactCheckAPI(api_key)
