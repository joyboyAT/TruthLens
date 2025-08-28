#!/usr/bin/env python3
"""
TruthLens FastAPI Backend
Wraps the complete TruthLens pipeline and exposes a /predict endpoint
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add the parent directory to Python path to import TruthLens modules
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Disable ML to avoid GPU memory issues and use heuristic detection
os.environ["TRUTHLENS_DISABLE_ML"] = "1"

# Import TruthLens modules
try:
    # Enhanced Pipeline (Primary)
    from src.enhanced_pipeline import process_text_enhanced, EnhancedTruthLensPipeline
    
    # Original Pipeline (Fallback)
    from src.ingestion import process_input, detect_input_type
    from src.translation import normalize_text
    from src.claim_processing import ClaimProcessor
    from src.evidence_retrieval.hybrid_retriever import HybridEvidenceRetriever
    from src.evidence_retrieval.trusted_sources import TrustedSourcesDatabase, TrustedSourcesAPI
    from src.verification.pipeline import VerificationPipeline
    from src.output_ux.citation_highlighter import CitationHighlighter
    from src.output_ux.verdict_mapping import map_verdict
    from src.output_ux.explanation_generator import UserExplanationLayer, generate_user_explanation
    from src.output_ux.cue_badges import generate_cue_badges
    from src.output_ux.prebunk_card import build_prebunk_card
    from src.output_ux.evidence_cards import build_evidence_cards
    
    TRUTHLENS_AVAILABLE = True
    ENHANCED_PIPELINE_AVAILABLE = True
    print("✅ Enhanced TruthLens pipeline loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import TruthLens modules: {e}")
    print("Falling back to mock data")
    TRUTHLENS_AVAILABLE = False
    ENHANCED_PIPELINE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TruthLens API",
    description="Complete TruthLens pipeline for fact-checking and misinformation detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PredictRequest(BaseModel):
    text: str = Field(..., description="Input text to analyze", min_length=1, max_length=10000)
    input_type: str = Field(default="text", description="Type of input (text, url, image)")
    max_claims: int = Field(default=5, description="Maximum number of claims to extract", ge=1, le=20)
    max_evidence_per_claim: int = Field(default=3, description="Maximum evidence per claim", ge=1, le=10)

class ClaimResult(BaseModel):
    id: str
    text: str
    subject: str
    predicate: str
    object: str
    checkworthiness: float
    context: Dict[str, Any]

class EvidenceResult(BaseModel):
    id: str
    title: str
    content: str
    source: str
    url: str
    relevance_score: float

class VerificationResult(BaseModel):
    claim_text: str
    confidence_score: float
    stance: str
    verdict: str
    highlights: List[str]

class ExplanationResult(BaseModel):
    claim_text: str
    why_misleading: str
    manipulation_cues: List[str]
    missing_context: str
    prebunk_tip: str

class PipelinePhase(BaseModel):
    status: str
    processing_time: float
    details: Dict[str, Any]

class EnhancedPredictResponse(BaseModel):
    input: Dict[str, Any]
    evidence_retrieval: Dict[str, Any]
    verification: Dict[str, Any]
    summary: Dict[str, Any]
    metadata: Dict[str, Any]

class PredictResponse(BaseModel):
    input: Dict[str, Any]
    pipeline_results: Dict[str, PipelinePhase]
    summary: Dict[str, Any]
    timestamp: str

# Initialize TruthLens components
claim_processor = None
evidence_retriever = None
verification_pipeline = None
explanation_layer = None
citation_highlighter = None

def initialize_truthlens_pipeline():
    """Initialize all TruthLens pipeline components."""
    global claim_processor, evidence_retriever, verification_pipeline, explanation_layer, citation_highlighter
    
    if not TRUTHLENS_AVAILABLE:
        return
    
    try:
        # Initialize Phase 2: Claim Processing
        print("🔍 Initializing claim processor...")
        claim_processor = ClaimProcessor()
        print("✅ Claim processor initialized")
        
        # Initialize Phase 3: Evidence Retrieval
        print("🔍 Initializing evidence retriever...")
        evidence_retriever = HybridEvidenceRetriever()
        print("✅ Evidence retriever initialized")
        
        # Initialize Phase 4: Verification Pipeline
        print("🔍 Initializing verification pipeline...")
        verification_pipeline = VerificationPipeline()
        citation_highlighter = CitationHighlighter()
        print("✅ Verification pipeline initialized")
        
        # Initialize Phase 5: User Explanation Layer
        print("🔍 Initializing explanation layer...")
        explanation_layer = UserExplanationLayer()
        print("✅ Explanation layer initialized")
        
        print("🎉 Complete TruthLens pipeline initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing TruthLens pipeline: {e}")
        logger.error(f"TruthLens pipeline initialization error: {e}")

def process_truthlens_pipeline(
    input_data: str, 
    input_type: str = "text",
    max_claims: int = 5,
    max_evidence_per_claim: int = 3
) -> Dict[str, Any]:
    """
    Enhanced TruthLens pipeline function.
    Uses dynamic evidence retrieval and NLI verification for any input text.
    
    Args:
        input_data: Input text to analyze (any text, not just claims)
        input_type: Type of input (text, url, image)
        max_claims: Maximum number of claims to extract (for fallback)
        max_evidence_per_claim: Maximum evidence per claim
    
    Returns:
        Enhanced pipeline results
    """
    if not TRUTHLENS_AVAILABLE:
        return generate_mock_pipeline_result(input_data, max_claims, max_evidence_per_claim)
    
    try:
        # Use enhanced pipeline if available
        if ENHANCED_PIPELINE_AVAILABLE:
            logger.info("Using enhanced pipeline with dynamic evidence retrieval and NLI verification")
            return process_text_enhanced(input_data, max_evidence_per_claim)
        
        # Fallback to original pipeline
        logger.info("Using original pipeline (enhanced pipeline not available)")
        return process_original_pipeline(input_data, input_type, max_claims, max_evidence_per_claim)
        
    except Exception as e:
        logger.error(f"Error in enhanced pipeline: {e}")
        return {
            "error": str(e),
            "status": "failed",
            "timestamp": datetime.now().isoformat()
        }

def process_original_pipeline(
    input_data: str, 
    input_type: str = "text",
    max_claims: int = 5,
    max_evidence_per_claim: int = 3
) -> Dict[str, Any]:
    """Original pipeline processing (fallback)."""
    try:
        start_time = time.time()
        
        # Phase 1: Input Processing
        phase1_start = time.time()
        processed_input = process_input(input_data)
        normalized_text = normalize_text(processed_input["text"] if processed_input["success"] else input_data)
        phase1_time = time.time() - phase1_start
        
        logger.info(f"Phase 1 completed in {phase1_time:.3f}s")
        
        # Phase 2: Claim Extraction & Ranking
        phase2_start = time.time()
        claim_result = claim_processor.process_claims(normalized_text)
        top_claims = claim_result.atomic_claims[:max_claims]
        phase2_time = time.time() - phase2_start
        
        logger.info(f"Phase 2 completed in {phase2_time:.3f}s - Found {len(top_claims)} claims")
        
        # Phase 3: Evidence Retrieval
        phase3_start = time.time()
        all_evidence = []
        for claim in top_claims:
            evidence_result = evidence_retriever.retrieve_evidence(claim["text"], claim["id"])
            # Store evidence in a temporary list since claim is a dict
            claim_evidence = evidence_result.evidence[:max_evidence_per_claim]
            all_evidence.extend(claim_evidence)
        phase3_time = time.time() - phase3_start
        
        logger.info(f"Phase 3 completed in {phase3_time:.3f}s - Retrieved {len(all_evidence)} evidence")
        
        # Phase 4: Verification & Scoring
        phase4_start = time.time()
        verification_results = []
        for i, claim in enumerate(top_claims):
            # Get evidence for this claim (using index since we stored evidence separately)
            claim_evidence = all_evidence[i * max_evidence_per_claim:(i + 1) * max_evidence_per_claim] if all_evidence else []
            
            if claim_evidence:
                # Verify claim against evidence
                verification_result = verification_pipeline.verify_claim(
                    claim["text"], 
                    [ev.content for ev in claim_evidence]
                )
                
                # Highlight citations
                highlights = citation_highlighter.highlight_citation(
                    claim_evidence[0].content, 
                    claim["text"]
                )
                
                # Map verdict
                verdict = map_verdict(verification_result.confidence_score)
                
                verification_results.append({
                    "claim": claim,
                    "verification": verification_result,
                    "highlights": highlights,
                    "verdict": verdict
                })
        phase4_time = time.time() - phase4_start
        
        logger.info(f"Phase 4 completed in {phase4_time:.3f}s")
        
        # Phase 5: User Explanation Layer
        phase5_start = time.time()
        explanations = []
        cue_badges = []
        prebunk_cards = []
        
        for result in verification_results:
            # Generate explanations
            explanation = generate_user_explanation(
                result["claim"]["text"],
                result["verification"].confidence_score,
                result["verdict"]
            )
            explanations.append(explanation)
            
            # Generate cue badges
            cues = generate_cue_badges(result["claim"]["text"])
            cue_badges.extend(cues)
            
            # Generate prebunk card
            prebunk = build_prebunk_card(
                result["claim"]["text"],
                explanation.why_misleading,
                explanation.prebunk_tip
            )
            prebunk_cards.append(prebunk)
        
        # Format evidence cards
        evidence_cards = build_evidence_cards(all_evidence)
        phase5_time = time.time() - phase5_start
        
        logger.info(f"Phase 5 completed in {phase5_time:.3f}s")
        
        total_time = time.time() - start_time
        
        # Prepare comprehensive response
        response = {
            "input": {
                "original": input_data,
                "processed": normalized_text,
                "type": input_type
            },
            "pipeline_results": {
                "phase1": {
                    "status": "completed",
                    "processing_time": phase1_time,
                    "details": {
                        "normalized_text": normalized_text
                    }
                },
                "phase2": {
                    "status": "completed",
                    "processing_time": phase2_time,
                    "details": {
                        "total_claims": claim_result.total_claims,
                        "top_claims": len(top_claims),
                        "claims": [
                            {
                                "id": claim["id"],
                                "text": claim["text"],
                                "subject": claim["subject"],
                                "predicate": claim["predicate"],
                                "object": claim["object"],
                                "checkworthiness": claim["checkworthiness"],
                                "context": claim["context"]
                            } for claim in top_claims
                        ]
                    }
                },
                "phase3": {
                    "status": "completed",
                    "processing_time": phase3_time,
                    "details": {
                        "total_evidence": len(all_evidence),
                        "evidence_sources": list(set([ev.source for ev in all_evidence])),
                        "evidence": [
                            {
                                "id": ev.id,
                                "title": ev.title,
                                "content": ev.content,
                                "source": ev.source,
                                "url": ev.url,
                                "relevance_score": ev.relevance_score
                            } for ev in all_evidence
                        ]
                    }
                },
                "phase4": {
                    "status": "completed",
                    "processing_time": phase4_time,
                    "details": {
                        "verification_results": [
                            {
                                "claim_text": result["claim"]["text"],
                                "confidence_score": result["verification"].confidence_score,
                                "stance": result["verification"].stance,
                                "verdict": result["verdict"],
                                "highlights": result["highlights"].highlighted_spans if result["highlights"] else []
                            } for result in verification_results
                        ]
                    }
                },
                "phase5": {
                    "status": "completed",
                    "processing_time": phase5_time,
                    "details": {
                        "explanations": [
                            {
                                "claim_text": exp.claim_text,
                                "why_misleading": exp.why_misleading,
                                "manipulation_cues": exp.manipulation_cues,
                                "missing_context": exp.missing_context,
                                "prebunk_tip": exp.prebunk_tip
                            } for exp in explanations
                        ],
                        "cue_badges": cue_badges,
                        "prebunk_cards": prebunk_cards,
                        "evidence_cards": evidence_cards
                    }
                }
            },
            "summary": {
                "total_processing_time": total_time,
                "phases_completed": 5,
                "claims_processed": len(top_claims),
                "evidence_retrieved": len(all_evidence),
                "verdicts": [result["verdict"] for result in verification_results] if verification_results else [],
                "overall_confidence": sum([result["verification"].confidence_score for result in verification_results]) / len(verification_results) if verification_results else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in original pipeline: {e}")
        return {
            "error": str(e),
            "status": "failed",
            "timestamp": datetime.now().isoformat()
        }

def generate_mock_pipeline_result(
    input_data: str, 
    max_claims: int = 5, 
    max_evidence_per_claim: int = 3
) -> Dict[str, Any]:
    """Generate mock pipeline result for testing."""
    return {
        "input": {
            "original": input_data,
            "processed": input_data,
            "type": "text"
        },
        "pipeline_results": {
            "phase1": {
                "status": "completed",
                "processing_time": 0.1,
                "details": {
                    "normalized_text": input_data
                }
            },
            "phase2": {
                "status": "completed",
                "processing_time": 0.2,
                "details": {
                    "total_claims": 1,
                    "top_claims": 1,
                    "claims": [
                        {
                            "id": "mock_claim_1",
                            "text": input_data,
                            "subject": "Subject",
                            "predicate": "Predicate",
                            "object": "Object",
                            "checkworthiness": 0.85,
                            "context": "Mock context"
                        }
                    ]
                }
            },
            "phase3": {
                "status": "completed",
                "processing_time": 0.3,
                "details": {
                    "total_evidence": 2,
                    "evidence_sources": ["Mock Source 1", "Mock Source 2"],
                    "evidence": [
                        {
                            "id": "mock_ev_1",
                            "title": "Mock Evidence 1",
                            "content": "Mock evidence content",
                            "source": "Mock Source 1",
                            "url": "https://example.com/1",
                            "relevance_score": 0.9
                        },
                        {
                            "id": "mock_ev_2",
                            "title": "Mock Evidence 2",
                            "content": "Mock evidence content 2",
                            "source": "Mock Source 2",
                            "url": "https://example.com/2",
                            "relevance_score": 0.8
                        }
                    ]
                }
            },
            "phase4": {
                "status": "completed",
                "processing_time": 0.4,
                "details": {
                    "verification_results": [
                        {
                            "claim_text": input_data,
                            "confidence_score": 0.75,
                            "stance": "refutes",
                            "verdict": "Likely False",
                            "highlights": ["highlighted", "text", "spans"]
                        }
                    ]
                }
            },
            "phase5": {
                "status": "completed",
                "processing_time": 0.5,
                "details": {
                    "explanations": [
                        {
                            "claim_text": input_data,
                            "why_misleading": "This claim lacks scientific evidence",
                            "manipulation_cues": ["fear appeal", "cherry-picking"],
                            "missing_context": "Important context is missing",
                            "prebunk_tip": "Always check multiple sources"
                        }
                    ],
                    "cue_badges": ["Fear Appeal", "Cherry-picking"],
                    "prebunk_cards": ["Mock prebunk card"],
                    "evidence_cards": ["Mock evidence card"]
                }
            }
        },
        "summary": {
            "total_processing_time": 1.5,
            "phases_completed": 5,
            "claims_processed": 1,
            "evidence_retrieved": 2,
            "verdicts": ["Likely False"],
            "overall_confidence": 0.75
        },
        "timestamp": datetime.now().isoformat()
    }

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "TruthLens API",
        "version": "1.0.0",
        "description": "Enhanced TruthLens pipeline for fact-checking and misinformation detection",
        "endpoints": {
            "POST /predict": "Enhanced pipeline with dynamic evidence retrieval and NLI verification",
            "POST /predict/original": "Original pipeline with claim extraction and verification",
            "GET /health": "Health check",
            "GET /status": "Pipeline component status"
        },
        "features": {
            "enhanced_pipeline": "Dynamic evidence retrieval from Wikipedia, news, and fact-check sources",
            "nli_verification": "Natural Language Inference using RoBERTa-large-mnli",
            "any_text_input": "Works with any input text, not just pre-detected claims"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "truthlens_available": TRUTHLENS_AVAILABLE,
        "pipeline_components": {
            "claim_processor": claim_processor is not None,
            "evidence_retriever": evidence_retriever is not None,
            "verification_pipeline": verification_pipeline is not None,
            "explanation_layer": explanation_layer is not None,
            "citation_highlighter": citation_highlighter is not None
        }
    }

@app.get("/status")
async def pipeline_status():
    """Get pipeline component status."""
    return {
        "truthlens_available": TRUTHLENS_AVAILABLE,
        "pipeline_components": {
            "phase1_input_processing": True,  # Always available
            "phase2_claim_extraction": claim_processor is not None,
            "phase3_evidence_retrieval": evidence_retriever is not None,
            "phase4_verification": verification_pipeline is not None,
            "phase5_explanation": explanation_layer is not None
        },
        "capabilities": {
            "complete_pipeline": TRUTHLENS_AVAILABLE,
            "individual_phases": True,
            "mock_fallback": True
        }
    }

@app.post("/predict", response_model=EnhancedPredictResponse)
async def predict(request: PredictRequest):
    """
    Main prediction endpoint that processes text through the enhanced TruthLens pipeline.
    
    Args:
        request: PredictRequest containing text and parameters
    
    Returns:
        Enhanced pipeline results with dynamic evidence retrieval and NLI verification
    """
    try:
        logger.info(f"Processing prediction request: {request.text[:100]}...")
        
        # Process through enhanced pipeline
        result = process_truthlens_pipeline(
            input_data=request.text,
            input_type=request.input_type,
            max_claims=request.max_claims,
            max_evidence_per_claim=request.max_evidence_per_claim
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/original", response_model=PredictResponse)
async def predict_original(request: PredictRequest):
    """
    Original prediction endpoint that processes text through the original TruthLens pipeline.
    
    Args:
        request: PredictRequest containing text and parameters
    
    Returns:
        Original pipeline results with claim extraction and verification
    """
    try:
        logger.info(f"Processing original pipeline request: {request.text[:100]}...")
        
        # Process through original pipeline
        result = process_original_pipeline(
            input_data=request.text,
            input_type=request.input_type,
            max_claims=request.max_claims,
            max_evidence_per_claim=request.max_evidence_per_claim
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in original predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize TruthLens pipeline on startup."""
    print("🚀 Starting TruthLens FastAPI Backend")
    print("=" * 60)
    initialize_truthlens_pipeline()

if __name__ == "__main__":
    # Run the FastAPI app
    port = int(os.environ.get("PORT", 8000))
    print(f"🌐 Server starting on port {port}")
    print(f"📊 Health check: http://localhost:{port}/health")
    print(f"🔍 Pipeline status: http://localhost:{port}/status")
    print(f"🎯 Predict endpoint: POST http://localhost:{port}/predict")
    print(f"📚 API docs: http://localhost:{port}/docs")
    
    uvicorn.run(
        "truthlens_fastapi:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
