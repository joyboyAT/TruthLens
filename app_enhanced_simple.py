#!/usr/bin/env python3
"""
Simplified Enhanced TruthLens FastAPI App
Integrates with TruthLens components while avoiding import conflicts.
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

# Import TruthLens components selectively to avoid conflicts
TRUTHLENS_AVAILABLE = False
pipeline = None

try:
    # Try to import core components without problematic dependencies
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

# Component instances
news_handler = None
stance_classifier = None
verdict_aggregator = None
fact_check_api = None
semantic_search = None

def initialize_truthlens_components():
    """Initialize TruthLens components individually."""
    global news_handler, stance_classifier, verdict_aggregator, fact_check_api, semantic_search
    
    if not TRUTHLENS_AVAILABLE:
        logger.error("Enhanced TruthLens components not available")
        return False
    
    try:
        # Initialize components one by one
        news_handler = EnhancedNewsHandler(NEWS_API_KEY, GUARDIAN_API_KEY)
        logger.info("✅ Enhanced News Handler initialized")
        
        stance_classifier = EnhancedStanceClassifier()
        logger.info("✅ Enhanced Stance Classifier initialized")
        
        verdict_aggregator = EnhancedVerdictAggregator()
        logger.info("✅ Enhanced Verdict Aggregator initialized")
        
        if GOOGLE_API_KEY:
            fact_check_api = EnhancedFactCheckAPI(GOOGLE_API_KEY)
            logger.info("✅ Enhanced Fact Check API initialized")
        
        semantic_search = EnhancedSemanticSearch()
        logger.info("✅ Enhanced Semantic Search initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize TruthLens components: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize TruthLens components on startup."""
    logger.info("Starting TruthLens Enhanced FastAPI App...")
    success = initialize_truthlens_components()
    if success:
        logger.info("✅ TruthLens components ready")
    else:
        logger.warning("⚠️ TruthLens components not available - using fallback mode")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "TruthLens Enhanced Fact-Checking API",
        "version": "2.0.0",
        "status": "running",
        "truthlens_available": TRUTHLENS_AVAILABLE,
        "components_ready": all([news_handler, stance_classifier, verdict_aggregator]),
        "endpoints": {
            "POST /verify": "Verify a claim with enhanced TruthLens components",
            "POST /verify-basic": "Basic verification (fallback mode)",
            "GET /health": "Health check",
            "GET /sources": "List available news sources",
            "GET /components-status": "Check TruthLens components status"
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
            "news_handler": news_handler is not None,
            "stance_classifier": stance_classifier is not None,
            "verdict_aggregator": verdict_aggregator is not None,
            "fact_check_api": fact_check_api is not None,
            "semantic_search": semantic_search is not None,
            "news_api": NEWS_API_KEY is not None,
            "guardian_api": GUARDIAN_API_KEY is not None,
            "google_fact_check": GOOGLE_API_KEY is not None
        }
    }

@app.get("/components-status")
async def components_status():
    """Check TruthLens components status."""
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
    
    return {
        "status": "ready" if all([news_handler, stance_classifier, verdict_aggregator]) else "partial",
        "message": "Enhanced TruthLens components status",
        "components": {
            "enhanced_stance_classifier": stance_classifier is not None,
            "enhanced_verdict_aggregator": verdict_aggregator is not None,
            "enhanced_factcheck_api": fact_check_api is not None,
            "enhanced_semantic_search": semantic_search is not None,
            "enhanced_news_handler": news_handler is not None
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
    Verify a claim using enhanced TruthLens components.
    
    This endpoint uses enhanced components with:
    - Enhanced stance classification (thresholds: 0.6)
    - Rule-based signals for contradictions
    - Scientific consensus handling
    - Enhanced verdict aggregation (40% thresholds)
    - Multiple fact-checking sources
    - Semantic search and ranking
    """
    start_time = time.time()
    
    if not all([news_handler, stance_classifier, verdict_aggregator]):
        raise HTTPException(
            status_code=503,
            detail="Enhanced TruthLens components not available. Use /verify-basic for basic verification."
        )
    
    try:
        claim = claim_input.claim
        logger.info(f"Enhanced verification of claim: {claim}")
        
        # Step 1: Search for news articles using enhanced news handler
        news_articles = news_handler.get_news_sources(claim, max_articles=20, days_back=30)
        logger.info(f"Found {len(news_articles)} news articles")
        
        if not news_articles:
            # Return no evidence response
            processing_time = time.time() - start_time
            return VerificationResponse(
                claim=claim,
                verdict="🟡 Unclear",
                confidence=0.0,
                reasoning="No news articles found",
                sources_checked=["News API", "Guardian API"],
                verification_badge="⚠️ No sources found",
                evidence_strength="No evidence",
                stance_distribution={},
                stance_percentages={},
                fact_check_result=None,
                details=[],
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                total_articles=0,
                source_breakdown={},
                evidence_summary="No news articles found",
                rule_based_overrides=[]
            )
        
        # Step 2: Apply semantic search if available
        if semantic_search:
            try:
                search_results = semantic_search.search_and_rank_articles(claim, news_articles, 20)
                ranked_articles = [result.article for result in search_results]
                logger.info("Applied enhanced semantic search and ranking")
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
                ranked_articles = news_articles
        else:
            ranked_articles = news_articles
        
        # Step 3: Check fact-checking sources
        fact_check_result = None
        if fact_check_api:
            try:
                fact_check_result = fact_check_api.get_best_fact_check(claim)
                if fact_check_result:
                    logger.info(f"Found fact-check result: {fact_check_result.verdict}")
            except Exception as e:
                logger.warning(f"Fact-check API failed: {e}")
        
        # Step 4: Perform enhanced stance detection
        stance_results = []
        stance_counts = {"support": 0, "contradict": 0, "neutral": 0}
        
        for article in ranked_articles:
            try:
                # Convert article to the format expected by stance classifier
                # Handle both EnhancedSearchResult and regular dict objects
                if hasattr(article, 'article'):
                    # EnhancedSearchResult object
                    article_data = article.article
                else:
                    # Regular dict object
                    article_data = article
                
                # Debug: Check article type
                logger.info(f"Article type: {type(article_data)}")
                if hasattr(article_data, '__dict__'):
                    logger.info(f"Article attributes: {list(article_data.__dict__.keys())}")
                
                article_dict = {
                    'title': article_data.get('title', '') if isinstance(article_data, dict) else getattr(article_data, 'title', ''),
                    'content': (article_data.get('content', '') or article_data.get('description', '')) if isinstance(article_data, dict) else (getattr(article_data, 'content', '') or getattr(article_data, 'description', '')),
                    'url': article_data.get('url', '') if isinstance(article_data, dict) else getattr(article_data, 'url', ''),
                    'description': article_data.get('description', '') if isinstance(article_data, dict) else getattr(article_data, 'description', '')
                }
                
                stance_result = stance_classifier.classify_stance(claim, article_dict)
                stance_results.append(stance_result)
                stance_counts[stance_result.stance] += 1
                
            except Exception as e:
                logger.warning(f"Error in stance detection for article: {e}")
                # Create neutral stance as fallback
                stance_results.append(type('StanceResult', (), {
                    'stance': 'neutral',
                    'confidence': 0.5,
                    'evidence_sentences': [],
                    'reasoning': 'Stance detection failed, defaulting to neutral'
                })())
                stance_counts["neutral"] += 1
        
        # Step 5: Aggregate verdict using enhanced logic
        try:
            # Convert stance results to the format expected by verdict aggregator
            stance_dicts = []
            for sr in stance_results:
                stance_dict = {
                    'stance': sr.stance,
                    'confidence': sr.confidence,
                    'evidence_sentences': getattr(sr, 'evidence_sentences', []),
                    'reasoning': getattr(sr, 'reasoning', '')
                }
                stance_dicts.append(stance_dict)
            
            verdict_result = verdict_aggregator.aggregate_verdict(
                claim, stance_dicts, fact_check_result, len(ranked_articles)
            )
            
            logger.info(f"Verdict: {verdict_result.verdict} (confidence: {verdict_result.confidence:.1%})")
            
        except Exception as e:
            logger.error(f"Error in verdict aggregation: {e}")
            # Fallback verdict
            verdict_result = type('VerdictResult', (), {
                'verdict': '🟡 Unclear',
                'confidence': 0.5,
                'reasoning': f'Verdict aggregation failed: {str(e)}',
                'stance_distribution': stance_counts,
                'stance_percentages': {k: (v/len(stance_results)*100) for k, v in stance_counts.items()}
            })()
        
        # Step 6: Generate evidence summary
        evidence_summary = f"Found {len(ranked_articles)} articles with stance distribution: {stance_counts}"
        
        # Step 7: Extract rule-based overrides
        rule_based_overrides = []
        for sr in stance_results:
            if hasattr(sr, 'rule_based_override') and sr.rule_based_override:
                rule_based_overrides.append(sr.rule_based_override)
        
        # Convert to API response format
        details = []
        for i, article in enumerate(ranked_articles[:10]):  # Top 10 articles
            stance_result = stance_results[i] if i < len(stance_results) else None
            
            # Handle both EnhancedSearchResult and regular dict objects
            if hasattr(article, 'article'):
                # EnhancedSearchResult object
                article_data = article.article
            else:
                # Regular dict object
                article_data = article
            
            detail = ArticleDetail(
                title=article_data.get('title', ''),
                url=article_data.get('url', ''),
                source=article_data.get('source', ''),
                source_name=article_data.get('source_name', 'Unknown'),
                similarity_score=article_data.get('relevance_score', 0.5) or article_data.get('semantic_score', 0.5),
                published_at=article_data.get('published_at'),
                relevance_score=article_data.get('relevance_score', 0.5) or article_data.get('semantic_score', 0.5),
                stance=stance_result.stance if stance_result else 'neutral',
                stance_confidence=stance_result.confidence if stance_result else 0.0
            )
            details.append(detail)
        
        # Calculate source breakdown
        source_breakdown = {}
        for article in ranked_articles:
            # Handle both EnhancedSearchResult and regular dict objects
            if hasattr(article, 'article'):
                # EnhancedSearchResult object
                article_data = article.article
            else:
                # Regular dict object
                article_data = article
            
            source = article_data.get('source_name', 'Unknown')
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
            verdict_result.verdict, 
            evidence_strength, 
            verdict_result.stance_distribution
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"Enhanced verification completed in {processing_time:.2f}s")
        logger.info(f"Final verdict: {verdict_result.verdict} (confidence: {verdict_result.confidence:.1%})")
        
        return VerificationResponse(
            claim=claim,
            verdict=verdict_result.verdict,
            confidence=verdict_result.confidence,
            reasoning=verdict_result.reasoning,
            sources_checked=sources_checked,
            verification_badge=verification_badge,
            evidence_strength=evidence_strength,
            stance_distribution=verdict_result.stance_distribution,
            stance_percentages=verdict_result.stance_percentages,
            fact_check_result=fact_check_result.to_dict() if fact_check_result else None,
            details=details,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            total_articles=len(ranked_articles),
            source_breakdown=source_breakdown,
            evidence_summary=evidence_summary,
            rule_based_overrides=rule_based_overrides
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
    Basic claim verification (fallback mode when TruthLens components are not available).
    
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
            "reasoning": "Basic verification mode - TruthLens components not available",
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
    print("🔧 Enhanced TruthLens components: ", "✅ Available" if TRUTHLENS_AVAILABLE else "❌ Not available")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
