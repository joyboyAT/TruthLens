#!/usr/bin/env python3
"""
Enhanced TruthLens FastAPI App
Integrates with the complete TruthLens pipeline for comprehensive fact-checking.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import sys
import os
from datetime import datetime
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import TruthLens components
try:
    from src.enhanced_truthlens_pipeline import EnhancedTruthLensPipeline, EnhancedAnalysisResult
    from src.verification.enhanced_stance_classifier import EnhancedStanceClassifier
    from src.verification.enhanced_verdict_aggregator import EnhancedVerdictAggregator
    from src.verification.enhanced_factcheck_api import EnhancedFactCheckAPI
    from src.evidence_retrieval.enhanced_semantic_search import EnhancedSemanticSearch
    from src.news.enhanced_news_handler import EnhancedNewsHandler
    TRUTHLENS_AVAILABLE = True
    logger.info("Enhanced TruthLens components imported successfully")
except ImportError as e:
    logger.warning(f"Enhanced TruthLens components not available: {e}")
    TRUTHLENS_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="TruthLens Enhanced Fact-Checking API",
    description="Complete TruthLens pipeline integration for comprehensive fact-checking",
    version="2.0.0"
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
    stance: Optional[str] = None
    stance_confidence: Optional[float] = None

class VerificationResponse(BaseModel):
    claim: str
    verdict: str
    confidence: float
    reasoning: str
    sources_checked: List[str]
    verification_badge: str
    evidence_strength: str
    stance_distribution: Dict[str, int]
    stance_percentages: Dict[str, float]
    fact_check_result: Optional[Dict[str, Any]]
    details: List[ArticleDetail]
    processing_time: float
    timestamp: str
    total_articles: int
    source_breakdown: Dict[str, int]
    evidence_summary: str
    rule_based_overrides: List[str]

# API Keys
NEWS_API_KEY = "9c7e59e19af34bb8adb97d0a8bec458d"
GUARDIAN_API_KEY = "0b8f5a8d-d3a0-49e1-8472-f943dae59338"
GOOGLE_API_KEY = "AIzaSyACLtCwVw1dJeBNPKmbJ8Yfqu5D4zUK5Sc"

# Global pipeline instance
pipeline = None

def initialize_truthlens_pipeline():
    """Initialize the complete TruthLens pipeline."""
    global pipeline
    
    if not TRUTHLENS_AVAILABLE:
        logger.error("Enhanced TruthLens components not available")
        return False
    
    try:
        pipeline = EnhancedTruthLensPipeline(
            news_api_key=NEWS_API_KEY,
            guardian_api_key=GUARDIAN_API_KEY,
            google_api_key=GOOGLE_API_KEY
        )
        logger.info("Enhanced TruthLens Pipeline initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize TruthLens pipeline: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize TruthLens pipeline on startup."""
    logger.info("Starting TruthLens Enhanced FastAPI App...")
    success = initialize_truthlens_pipeline()
    if success:
        logger.info("✅ TruthLens pipeline ready")
    else:
        logger.warning("⚠️ TruthLens pipeline not available - using fallback mode")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "TruthLens Enhanced Fact-Checking API",
        "version": "2.0.0",
        "status": "running",
        "truthlens_available": TRUTHLENS_AVAILABLE,
        "pipeline_ready": pipeline is not None,
        "endpoints": {
            "POST /verify": "Verify a claim with complete TruthLens pipeline",
            "POST /verify-basic": "Basic verification (fallback mode)",
            "GET /health": "Health check",
            "GET /sources": "List available news sources",
            "GET /pipeline-status": "Check TruthLens pipeline status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "truthlens_available": TRUTHLENS_AVAILABLE,
            "pipeline_ready": pipeline is not None,
            "news_api": NEWS_API_KEY is not None,
            "guardian_api": GUARDIAN_API_KEY is not None,
            "google_fact_check": GOOGLE_API_KEY is not None
        }
    }

@app.get("/pipeline-status")
async def pipeline_status():
    """Check TruthLens pipeline status."""
    if not TRUTHLENS_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Enhanced TruthLens components not installed",
            "components": {
                "enhanced_stance_classifier": False,
                "enhanced_verdict_aggregator": False,
                "enhanced_factcheck_api": False,
                "enhanced_semantic_search": False,
                "enhanced_news_handler": False
            }
        }
    
    if not pipeline:
        return {
            "status": "not_initialized",
            "message": "Pipeline failed to initialize",
            "components": {
                "enhanced_stance_classifier": False,
                "enhanced_verdict_aggregator": False,
                "enhanced_factcheck_api": False,
                "enhanced_semantic_search": False,
                "enhanced_news_handler": False
            }
        }
    
    return {
        "status": "ready",
        "message": "Enhanced TruthLens pipeline is ready",
        "components": {
            "enhanced_stance_classifier": pipeline.stance_classifier is not None,
            "enhanced_verdict_aggregator": pipeline.verdict_aggregator is not None,
            "enhanced_factcheck_api": pipeline.fact_check_api is not None,
            "enhanced_semantic_search": pipeline.semantic_search is not None,
            "enhanced_news_handler": pipeline.news_handler is not None
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

def determine_evidence_strength(articles: List[Any]) -> str:
    """Determine evidence strength based on number of articles and similarity scores."""
    if not articles:
        return "No evidence"
    
    if len(articles) == 1:
        return "Weak"
    
    if len(articles) == 2:
        return "Moderate"
    
    # Calculate average similarity score for 3+ articles
    total_similarity = 0.0
    for article in articles:
        if hasattr(article, 'similarity_score'):
            # ArticleDetail object
            total_similarity += article.similarity_score
        elif isinstance(article, dict):
            # Dictionary object
            total_similarity += article.get('similarity_score', 0.0)
        else:
            total_similarity += 0.0
    
    avg_similarity = total_similarity / len(articles)
    
    if avg_similarity >= 0.7:
        return "Strong"
    elif avg_similarity >= 0.5:
        return "Moderate"
    else:
        return "Weak"

def assign_verification_badge(verdict: str, evidence_strength: str, stance_distribution: Dict[str, int]) -> str:
    """Assign verification badge based on verdict and evidence."""
    if verdict == "🔴 Likely False":
        return "❌ Claim appears to be false"
    elif verdict == "🟢 Likely True":
        return "✅ Claim appears to be true"
    elif verdict == "🟡 Unclear":
        if evidence_strength == "Strong":
            return "⚠️ Mixed evidence - unclear"
        else:
            return "⚠️ Insufficient evidence"
    else:
        return "❓ Unable to determine"

@app.post("/verify", response_model=VerificationResponse)
async def verify_claim_enhanced(claim_input: ClaimInput):
    """
    Verify a claim using the complete TruthLens pipeline.
    
    This endpoint uses the full enhanced pipeline with:
    - Enhanced stance classification (thresholds: 0.6)
    - Rule-based signals for contradictions
    - Scientific consensus handling
    - Enhanced verdict aggregation (40% thresholds)
    - Multiple fact-checking sources
    - Semantic search and ranking
    """
    start_time = time.time()
    
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Enhanced TruthLens pipeline not available. Use /verify-basic for basic verification."
        )
    
    try:
        claim = claim_input.claim
        logger.info(f"Enhanced verification of claim: {claim}")
        
        # Use the complete TruthLens pipeline
        result: EnhancedAnalysisResult = pipeline.analyze_claim(claim, max_articles=20)
        
        # Convert to API response format
        details = []
        for i, article in enumerate(result.news_articles[:10]):  # Top 10 articles
            # Find corresponding stance result
            stance_result = None
            if i < len(result.stance_results):
                stance_result = result.stance_results[i]
            
            # Handle both dict and object types
            if isinstance(article, dict):
                title = article.get('title', '')
                url = article.get('url', '')
                source = article.get('source', '')
                source_name = article.get('source_name', 'Unknown')
                similarity_score = article.get('relevance_score', 0.5)
                published_at = article.get('published_at')
                relevance_score = article.get('relevance_score', 0.5)
            else:
                # Handle object attributes
                title = getattr(article, 'title', '')
                url = getattr(article, 'url', '')
                source = getattr(article, 'source', '')
                source_name = getattr(article, 'source_name', 'Unknown')
                similarity_score = getattr(article, 'relevance_score', 0.5)
                published_at = getattr(article, 'published_at', None)
                relevance_score = getattr(article, 'relevance_score', 0.5)
            
            detail = ArticleDetail(
                title=title,
                url=url,
                source=source,
                source_name=source_name,
                similarity_score=similarity_score,
                published_at=published_at,
                relevance_score=relevance_score,
                stance=stance_result.get('stance', 'neutral') if stance_result else 'neutral',
                stance_confidence=stance_result.get('confidence', 0.0) if stance_result else 0.0
            )
            details.append(detail)
        
        # Calculate source breakdown
        source_breakdown = {}
        for article in result.news_articles:
            if isinstance(article, dict):
                source = article.get('source_name', 'Unknown')
            else:
                source = getattr(article, 'source_name', 'Unknown')
            source_breakdown[source] = source_breakdown.get(source, 0) + 1
        
        # Determine sources checked
        sources_checked = []
        if source_breakdown.get('NewsAPI', 0) > 0:
            sources_checked.append("News API")
        if source_breakdown.get('Guardian', 0) > 0:
            sources_checked.append("Guardian API")
        if source_breakdown.get('Currents', 0) > 0:
            sources_checked.append("Currents API")
        
        # Determine evidence strength
        evidence_strength = determine_evidence_strength(details)
        
        # Assign verification badge
        verification_badge = assign_verification_badge(
            result.verdict, 
            evidence_strength, 
            result.stance_distribution
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"Enhanced verification completed in {processing_time:.2f}s")
        logger.info(f"Final verdict: {result.verdict} (confidence: {result.confidence:.1%})")
        
        return VerificationResponse(
            claim=claim,
            verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
            sources_checked=sources_checked,
            verification_badge=verification_badge,
            evidence_strength=evidence_strength,
            stance_distribution=result.stance_distribution,
            stance_percentages=result.stance_percentages,
            fact_check_result=result.fact_check_result,
            details=details,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            total_articles=len(result.news_articles),
            source_breakdown=source_breakdown,
            evidence_summary=result.evidence_summary,
            rule_based_overrides=result.rule_based_overrides
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced verification: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced verification failed: {str(e)}"
        )

@app.post("/verify-basic")
async def verify_claim_basic(claim_input: ClaimInput):
    """
    Basic claim verification (fallback mode when TruthLens pipeline is not available).
    
    This endpoint provides basic functionality similar to the original app.
    """
    start_time = time.time()
    
    try:
        claim = claim_input.claim
        logger.info(f"Basic verification of claim: {claim}")
        
        # This would implement basic verification logic
        # For now, return a simple response
        return {
            "claim": claim,
            "verdict": "🟡 Unclear",
            "confidence": 0.5,
            "reasoning": "Basic verification mode - TruthLens pipeline not available",
            "sources_checked": ["News API"],
            "verification_badge": "⚠️ Basic verification only",
            "evidence_strength": "Weak",
            "stance_distribution": {"neutral": 1},
            "stance_percentages": {"neutral": 100.0},
            "fact_check_result": None,
            "details": [],
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "total_articles": 0,
            "source_breakdown": {},
            "evidence_summary": "Basic verification mode - limited functionality",
            "rule_based_overrides": []
        }
        
    except Exception as e:
        logger.error(f"Error in basic verification: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Basic verification failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting TruthLens Enhanced FastAPI Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    print("🔧 Enhanced TruthLens pipeline integration: ", "✅ Available" if TRUTHLENS_AVAILABLE else "❌ Not available")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
