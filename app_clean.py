#!/usr/bin/env python3
"""
Clean TruthLens FastAPI App
A working FastAPI wrapper for fact-checking claims.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import requests
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TruthLens Fact-Checking API",
    description="AI-powered fact-checking API using multiple news sources",
    version="1.0.0"
)

# Pydantic Models
class ClaimInput(BaseModel):
    claim: str = Field(..., description="The claim to verify", min_length=10, max_length=500)

class ArticleDetail(BaseModel):
    title: str
    url: str
    source: str
    source_name: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    published_at: Optional[str] = None
    relevance_score: Optional[float] = None

class VerificationResponse(BaseModel):
    claim: str
    sources_checked: List[str]
    verification_badge: str
    evidence_strength: str
    details: List[ArticleDetail]
    processing_time: float
    timestamp: str
    total_articles: int
    source_breakdown: Dict[str, int]

# API Keys
NEWS_API_KEY = "9c7e59e19af34bb8adb97d0a8bec458d"
GUARDIAN_API_KEY = "0b8f5a8d-d3a0-49e1-8472-f943dae59338"

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "TruthLens Fact-Checking API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /verify": "Verify a claim",
            "GET /health": "Health check",
            "GET /sources": "List available news sources"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "news_api": NEWS_API_KEY is not None,
            "guardian_api": GUARDIAN_API_KEY is not None
        }
    }

@app.get("/sources")
async def list_sources():
    """List available news sources."""
    return {
        "available_sources": [
            {
                "name": "News API",
                "status": "available" if NEWS_API_KEY else "not configured",
                "description": "Primary news source with comprehensive coverage"
            },
            {
                "name": "The Guardian",
                "status": "available" if GUARDIAN_API_KEY else "not configured",
                "description": "Fallback news source for cross-referencing"
            }
        ]
    }

async def search_newsapi(claim: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search News API for articles related to the claim."""
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': claim,
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': max_results
        }
        
        logger.info(f"Searching News API for: {claim}")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            
            logger.info(f"News API returned {len(articles)} articles")
            
            return [
                {
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'source_name': 'NewsAPI',
                    'published_at': article.get('publishedAt', ''),
                    'relevance_score': 0.8,
                    'content': article.get('description', '')
                }
                for article in articles if article.get('title') and article.get('url')
            ]
        elif response.status_code == 429:
            logger.warning("News API rate limit hit")
            return []
        else:
            logger.warning(f"News API request failed with status {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"News API search failed: {e}")
        return []

async def search_guardian(claim: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search Guardian API as fallback if News API fails."""
    try:
        url = "https://content.guardianapis.com/search"
        params = {
            'api-key': GUARDIAN_API_KEY,
            'q': claim,
            'section': 'news',
            'show-fields': 'headline,trailText,lastModified',
            'page-size': max_results
        }
        
        logger.info(f"Searching Guardian API for: {claim}")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('response', {}).get('results', [])
            
            logger.info(f"Guardian API returned {len(articles)} articles")
            
            return [
                {
                    'title': article.get('webTitle', ''),
                    'url': article.get('webUrl', ''),
                    'source': 'The Guardian',
                    'source_name': 'Guardian',
                    'published_at': article.get('webPublicationDate', ''),
                    'relevance_score': 0.7,
                    'content': article.get('fields', {}).get('trailText', '')
                }
                for article in articles if article.get('webTitle') and article.get('webUrl')
            ]
        else:
            logger.warning(f"Guardian API request failed with status {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Guardian API search failed: {e}")
        return []

def calculate_similarity(claim: str, title: str) -> float:
    """Calculate similarity between claim and article title using Jaccard similarity."""
    try:
        # Convert to lowercase and split into words
        claim_words = set(claim.lower().split())
        title_words = set(title.lower().split())
        
        if not claim_words or not title_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(claim_words.intersection(title_words))
        union = len(claim_words.union(title_words))
        
        if union == 0:
            return 0.0
        
        base_similarity = intersection / union
        
        # Boost score for exact word matches
        exact_matches = sum(1 for word in claim_words if word in title_words)
        boost = min(0.2, exact_matches * 0.1)
        
        return min(1.0, base_similarity + boost)
        
    except Exception as e:
        logger.warning(f"Similarity calculation failed: {e}")
        return 0.0

def determine_evidence_strength(articles: List[Dict[str, Any]]) -> str:
    """Determine evidence strength based on number of articles and similarity scores."""
    if not articles:
        return "No evidence"
    
    if len(articles) == 1:
        return "Weak"
    
    if len(articles) == 2:
        return "Moderate"
    
    # Calculate average similarity score for 3+ articles
    total_similarity = sum(article.get('similarity_score', 0.0) for article in articles)
    avg_similarity = total_similarity / len(articles)
    
    if avg_similarity >= 0.7:
        return "Strong"
    elif avg_similarity >= 0.5:
        return "Moderate"
    else:
        return "Weak"

def assign_verification_badge(articles: List[Dict[str, Any]], evidence_strength: str) -> str:
    """Assign verification badge based on evidence strength and source diversity."""
    if evidence_strength == "No evidence":
        return "⚠️ No sources found"
    
    if evidence_strength == "Weak":
        return "⚠️ Limited verification"
    
    # Check source diversity
    sources = set(article.get('source_name', 'Unknown') for article in articles)
    
    if len(sources) >= 2:
        if evidence_strength == "Strong":
            return "✅ Verified by multiple sources"
        else:
            return "✅ Verified by multiple sources"
    else:
        if evidence_strength == "Strong":
            return "✅ Verified by single source"
        else:
            return "⚠️ Single source verification"

@app.post("/verify", response_model=VerificationResponse)
async def verify_claim(claim_input: ClaimInput):
    """
    Verify a claim using multiple news sources and semantic analysis.
    
    Args:
        claim_input: The claim to verify
        
    Returns:
        VerificationResponse with structured verification results
    """
    start_time = time.time()
    
    try:
        claim = claim_input.claim
        logger.info(f"Verifying claim: {claim}")
        
        # Step 1: Search News API
        newsapi_articles = await search_newsapi(claim, max_results=10)
        logger.info(f"News API returned {len(newsapi_articles)} articles")
        
        # Step 2: Fallback to Guardian if News API has no results
        guardian_articles = []
        if len(newsapi_articles) < 3:
            logger.info("News API returned insufficient results, trying Guardian API")
            guardian_articles = await search_guardian(claim, max_results=10)
            logger.info(f"Guardian API returned {len(guardian_articles)} articles")
        
        # Step 3: Combine and normalize results
        all_articles = newsapi_articles + guardian_articles
        
        if not all_articles:
            # Return no evidence response
            processing_time = time.time() - start_time
            return VerificationResponse(
                claim=claim,
                sources_checked=["News API", "Guardian API"],
                verification_badge="⚠️ No sources found",
                evidence_strength="No evidence",
                details=[],
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                total_articles=0,
                source_breakdown={}
            )
        
        # Step 4: Calculate similarity scores for each article
        for article in all_articles:
            similarity = calculate_similarity(claim, article['title'])
            article['similarity_score'] = similarity
        
        # Step 5: Sort by similarity score
        all_articles.sort(key=lambda x: x.get('similarity_score', 0.0), reverse=True)
        
        # Step 6: Determine evidence strength and verification badge
        evidence_strength = determine_evidence_strength(all_articles)
        verification_badge = assign_verification_badge(all_articles, evidence_strength)
        
        # Step 7: Prepare response details
        details = []
        for article in all_articles[:10]:  # Limit to top 10 results
            detail = ArticleDetail(
                title=article['title'],
                url=article['url'],
                source=article['source'],
                source_name=article['source_name'],
                similarity_score=article['similarity_score'],
                published_at=article.get('published_at'),
                relevance_score=article.get('relevance_score')
            )
            details.append(detail)
        
        # Step 8: Calculate source breakdown
        source_breakdown = {}
        for article in all_articles:
            source = article.get('source_name', 'Unknown')
            source_breakdown[source] = source_breakdown.get(source, 0) + 1
        
        # Step 9: Prepare sources checked list
        sources_checked = []
        if newsapi_articles:
            sources_checked.append("News API")
        if guardian_articles:
            sources_checked.append("Guardian API")
        
        processing_time = time.time() - start_time
        
        logger.info(f"Claim verification completed in {processing_time:.2f}s")
        logger.info(f"Evidence strength: {evidence_strength}")
        logger.info(f"Total articles: {len(all_articles)}")
        
        return VerificationResponse(
            claim=claim,
            sources_checked=sources_checked,
            verification_badge=verification_badge,
            evidence_strength=evidence_strength,
            details=details,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            total_articles=len(all_articles),
            source_breakdown=source_breakdown
        )
        
    except Exception as e:
        logger.error(f"Error verifying claim: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Claim verification failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting TruthLens FastAPI Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
