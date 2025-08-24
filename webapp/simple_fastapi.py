#!/usr/bin/env python3
"""
Simplified TruthLens FastAPI Backend for Testing
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import time

app = FastAPI(
    title="TruthLens API (Simple)",
    description="Simplified TruthLens pipeline for testing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str = Field(..., description="Input text to analyze", min_length=1, max_length=10000)
    input_type: str = Field(default="text", description="Type of input")
    max_claims: int = Field(default=5, description="Maximum claims to extract", ge=1, le=20)
    max_evidence_per_claim: int = Field(default=3, description="Maximum evidence per claim", ge=1, le=10)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "TruthLens API (Simple)",
        "version": "1.0.0",
        "description": "Simplified TruthLens pipeline for testing",
        "endpoints": {
            "POST /predict": "Analyze text through pipeline",
            "GET /health": "Health check",
            "GET /status": "Pipeline status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "truthlens_available": True,
        "pipeline_components": {
            "claim_processor": True,
            "evidence_retriever": True,
            "verification_pipeline": True,
            "explanation_layer": True,
            "citation_highlighter": True
        }
    }

@app.get("/status")
async def pipeline_status():
    """Get pipeline component status."""
    return {
        "truthlens_available": True,
        "pipeline_components": {
            "phase1_input_processing": True,
            "phase2_claim_extraction": True,
            "phase3_evidence_retrieval": True,
            "phase4_verification": True,
            "phase5_explanation": True
        },
        "capabilities": {
            "complete_pipeline": True,
            "individual_phases": True,
            "mock_fallback": True
        }
    }

@app.post("/predict")
async def predict(request: PredictRequest):
    """Main prediction endpoint with mock data."""
    try:
        start_time = time.time()
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Generate mock response
        mock_response = {
            "input": {
                "original": request.text,
                "processed": request.text,
                "type": request.input_type
            },
            "pipeline_results": {
                "phase1": {
                    "status": "completed",
                    "processing_time": 0.1,
                    "details": {
                        "normalized_text": request.text
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
                                "text": request.text,
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
                                "content": "This is mock evidence content for testing purposes.",
                                "source": "Mock Source 1",
                                "url": "https://example.com/1",
                                "relevance_score": 0.9
                            },
                            {
                                "id": "mock_ev_2",
                                "title": "Mock Evidence 2",
                                "content": "Another mock evidence content for testing.",
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
                                "claim_text": request.text,
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
                                "claim_text": request.text,
                                "why_misleading": "This claim lacks scientific evidence and has been debunked by multiple studies.",
                                "manipulation_cues": ["fear appeal", "cherry-picking"],
                                "missing_context": "Important context about vaccine safety studies is missing.",
                                "prebunk_tip": "Always check multiple sources and look for peer-reviewed studies."
                            }
                        ],
                        "cue_badges": ["Fear Appeal", "Cherry-picking"],
                        "prebunk_cards": ["Mock prebunk card"],
                        "evidence_cards": ["Mock evidence card"]
                    }
                }
            },
            "summary": {
                "total_processing_time": time.time() - start_time,
                "phases_completed": 5,
                "claims_processed": 1,
                "evidence_retrieved": 2,
                "verdicts": ["Likely False"],
                "overall_confidence": 0.75
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return mock_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simplified TruthLens FastAPI Backend")
    print("=" * 60)
    print("🌐 Server starting on port 8000")
    print("📊 Health check: http://localhost:8000/health")
    print("🎯 Predict endpoint: POST http://localhost:8000/predict")
    print("📚 API docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "simple_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
