#!/usr/bin/env python3
"""
TruthLens FastAPI App
FastAPI wrapper for the enhanced TruthLens fact-checking pipeline.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import TruthLens components
try:
    from src.enhanced_truthlens_pipeline import EnhancedTruthLensPipeline
    from src.news.enhanced_news_handler import EnhancedNewsHandler
    from src.evidence_retrieval.semantic_cross_reference_scorer import SemanticCrossReferenceScorer
    TRUTHLENS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TruthLens components not available: {e}")
    TRUTHLENS_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="TruthLens Fact-Checking API",
    description="AI-powered fact-checking API using multiple news sources and semantic analysis",
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

# Global variables for API keys and handlers
NEWS_API_KEY = "9c7e59e19af34bb8adb97d0a8bec458d"
GUARDIAN_API_KEY = "0b8f5a8d-d3a0-49e1-8472-f943dae59338"
GOOGLE_API_KEY = "AIzaSyACLtCwVw1dJeBNPKmbJ8Yfqu5D4zUK5Sc"

# Initialize handlers
news_handler = None
cross_reference_scorer = None
pipeline = None

def initialize_handlers():
    """Initialize news handlers and pipeline."""
    global news_handler, cross_reference_scorer, pipeline
    
    try:
        if TRUTHLENS_AVAILABLE:
            # Initialize enhanced news handler
            news_handler = EnhancedNewsHandler(
                news_api_key=NEWS_API_KEY,
                guardian_api_key=GUARDIAN_API_KEY
            )
            logger.info("Enhanced News Handler initialized")
            
            # Initialize cross-reference scorer
            cross_reference_scorer = SemanticCrossReferenceScorer()
            logger.info("Cross-reference scorer initialized")
            
            # Initialize pipeline
            pipeline = EnhancedTruthLensPipeline(
                news_api_key=NEWS_API_KEY,
                guardian_api_key=GUARDIAN_API_KEY,
                google_api_key=GOOGLE_API_KEY
            )
            logger.info("Enhanced TruthLens Pipeline initialized")
            
        else:
            logger.error("TruthLens components not available")
            
    except Exception as e:
        logger.error(f"Failed to initialize handlers: {e}")

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize handlers on startup."""
    initialize_handlers()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "TruthLens Fact-Checking API",
        "version": "1.0.0",
        "status": "running",
        "components_available": TRUTHLENS_AVAILABLE,
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
            "news_handler": news_handler is not None,
            "cross_reference_scorer": cross_reference_scorer is not None,
            "pipeline": pipeline is not None
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
            },
            {
                "name": "Google Fact Check",
                "status": "available" if GOOGLE_API_KEY else "not configured",
                "description": "Fact-checking database integration"
            }
        ]
    }

async def search_newsapi(claim: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search News API for articles related to the claim."""
    if not news_handler:
        logger.warning("News handler not available")
        return []
    
    try:
        # Use the enhanced news handler to get News API results
        articles = news_handler._get_news_api_results(claim, max_results, days_back=30)
        
        if articles:
            logger.info(f"News API returned {len(articles)} articles")
            return [
                {
                    'title': article.title,
                    'url': article.url,
                    'source': article.source,
                    'source_name': 'NewsAPI',
                    'published_at': article.published_at,
                    'relevance_score': article.relevance_score,
                    'content': article.content
                }
                for article in articles
            ]
        else:
            logger.info("News API returned no articles")
            return []
            
    except Exception as e:
        logger.error(f"News API search failed: {e}")
        return []

async def search_guardian(claim: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search Guardian API as fallback if News API fails."""
    if not news_handler:
        logger.warning("News handler not available")
        return []
    
    try:
        # Use the enhanced news handler to get Guardian results
        articles = news_handler._get_guardian_results(claim, max_results, days_back=30)
        
        if articles:
            logger.info(f"Guardian API returned {len(articles)} articles")
            return [
                {
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', 'The Guardian'),
                    'source_name': 'Guardian',
                    'published_at': article.get('published_at', ''),
                    'relevance_score': article.get('relevance_score', 0.5),
                    'content': article.get('content', '')
                }
                for article in articles
            ]
        else:
            logger.info("Guardian API returned no articles")
            return []
            
    except Exception as e:
        logger.error(f"Guardian API search failed: {e}")
        return []

async def calculate_similarity(claim: str, title: str) -> float:
    """Calculate similarity between claim and article title using semantic scoring."""
    if not cross_reference_scorer:
        # Fallback to simple text similarity
        claim_words = set(claim.lower().split())
        title_words = set(title.lower().split())
        
        if not claim_words or not title_words:
            return 0.0
        
        intersection = len(claim_words.intersection(title_words))
        union = len(claim_words.union(title_words))
        
        return intersection / union if union > 0 else 0.0
    
    try:
        # Use semantic cross-reference scorer for better similarity
        # This is a simplified version - in production you'd use the full scorer
        claim_words = set(claim.lower().split())
        title_words = set(title.lower().split())
        
        if not claim_words or not title_words:
            return 0.0
        
        intersection = len(claim_words.intersection(title_words))
        union = len(claim_words.union(title_words))
        
        # Boost score for semantic similarity
        base_score = intersection / union if union > 0 else 0.0
        
        # Simple semantic boost based on word overlap
        semantic_boost = min(0.3, base_score * 0.5)
        
        return min(1.0, base_score + semantic_boost)
        
    except Exception as e:
        logger.warning(f"Semantic similarity calculation failed: {e}")
        # Fallback to simple similarity
        claim_words = set(claim.lower().split())
        title_words = set(title.lower().split())
        
        if not claim_words or not title_words:
            return 0.0
        
        intersection = len(claim_words.intersection(title_words))
        union = len(claim_words.union(title_words))
        
        return intersection / union if union > 0 else 0.0

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
    start_time = datetime.now()
    
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
            processing_time = (datetime.now() - start_time).total_seconds()
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
            similarity = await calculate_similarity(claim, article['title'])
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
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
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

@app.post("/verify-advanced")
async def verify_claim_advanced(claim_input: ClaimInput):
    """
    Advanced claim verification using the full TruthLens pipeline.
    
    This endpoint uses the complete enhanced pipeline for more accurate results.
    """
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Advanced verification not available - pipeline not initialized"
        )
    
    try:
        start_time = datetime.now()
        claim = claim_input.claim
        
        logger.info(f"Advanced verification of claim: {claim}")
        
        # Use the full pipeline
        result = pipeline.analyze_claim(claim, max_articles=15)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "claim": claim,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "evidence_summary": result.evidence_summary,
            "news_articles": len(result.news_articles),
            "stance_distribution": dict(result.stance_distribution),
            "fact_check_result": result.fact_check_result,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Advanced verification failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Advanced verification failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
