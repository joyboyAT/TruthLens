#!/usr/bin/env python3
"""
TruthLens Complete Pipeline API
Integrates all phases: Input Processing → Claim Extraction → Evidence Retrieval → Verification → User Explanation
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
import uuid
import time

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add the parent directory to Python path to import TruthLens modules
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import TruthLens modules
try:
    # Phase 1: Input Processing
    from src.ingestion import process_input, detect_input_type
    from src.translation import normalize_text
    
    # Phase 2: Claim Extraction & Ranking
    from src.claim_processing import ClaimProcessor
    
    # Phase 3: Evidence Retrieval
    from src.evidence_retrieval.hybrid_retriever import HybridEvidenceRetriever
    from src.evidence_retrieval.trusted_sources import TrustedSourcesDatabase, TrustedSourcesAPI
    
    # Phase 4: Verification & Scoring
    from src.verification.pipeline import VerificationPipeline
    from src.output_ux.citation_highlighter import CitationHighlighter
    from src.output_ux.verdict_mapping import map_verdict
    
    # Phase 5: User Explanation Layer
    from src.output_ux.explanation_generator import UserExplanationLayer, generate_user_explanation
    from src.output_ux.cue_badges import generate_cue_badges
    from src.output_ux.prebunk_card import build_prebunk_card
    from src.output_ux.evidence_cards import format_evidence_cards
    
    TRUTHLENS_AVAILABLE = True
    print("✅ TruthLens complete pipeline loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import TruthLens modules: {e}")
    print("Falling back to mock data")
    TRUTHLENS_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def process_complete_pipeline(input_data: str, input_type: str = "text") -> Dict[str, Any]:
    """Process input through the complete TruthLens pipeline."""
    if not TRUTHLENS_AVAILABLE:
        return generate_mock_pipeline_result(input_data)
    
    try:
        start_time = time.time()
        
        # Phase 1: Input Processing
        phase1_start = time.time()
        processed_input = process_input(input_data, input_type)
        normalized_text = normalize_text(processed_input)
        phase1_time = time.time() - phase1_start
        
        logger.info(f"Phase 1 completed in {phase1_time:.3f}s")
        
        # Phase 2: Claim Extraction & Ranking
        phase2_start = time.time()
        claim_result = claim_processor.process_claims(normalized_text)
        top_claims = claim_result.claims[:5]  # Get top 5 claims
        phase2_time = time.time() - phase2_start
        
        logger.info(f"Phase 2 completed in {phase2_time:.3f}s - Found {len(top_claims)} claims")
        
        # Phase 3: Evidence Retrieval
        phase3_start = time.time()
        all_evidence = []
        for claim in top_claims:
            evidence_result = evidence_retriever.retrieve_evidence(claim.text, claim.id)
            claim.evidence = evidence_result.evidence[:3]  # Top 3 evidence per claim
            all_evidence.extend(evidence_result.evidence[:3])
        phase3_time = time.time() - phase3_start
        
        logger.info(f"Phase 3 completed in {phase3_time:.3f}s - Retrieved {len(all_evidence)} evidence")
        
        # Phase 4: Verification & Scoring
        phase4_start = time.time()
        verification_results = []
        for claim in top_claims:
            if claim.evidence:
                # Verify claim against evidence
                verification_result = verification_pipeline.verify_claim(
                    claim.text, 
                    [ev.content for ev in claim.evidence]
                )
                
                # Highlight citations
                highlights = citation_highlighter.highlight_citation(
                    claim.evidence[0].content, 
                    claim.text
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
                result["claim"].text,
                result["verification"].confidence_score,
                result["verdict"]
            )
            explanations.append(explanation)
            
            # Generate cue badges
            cues = generate_cue_badges(result["claim"].text)
            cue_badges.extend(cues)
            
            # Generate prebunk card
            prebunk = build_prebunk_card(
                result["claim"].text,
                explanation.why_misleading,
                explanation.prebunk_tip
            )
            prebunk_cards.append(prebunk)
        
        # Format evidence cards
        evidence_cards = format_evidence_cards(all_evidence)
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
                    "normalized_text": normalized_text
                },
                "phase2": {
                    "status": "completed",
                    "processing_time": phase2_time,
                    "total_claims": len(claim_result.claims),
                    "top_claims": len(top_claims),
                    "claims": [
                        {
                            "id": claim.id,
                            "text": claim.text,
                            "subject": claim.subject,
                            "predicate": claim.predicate,
                            "object": claim.object,
                            "checkworthiness": claim.checkworthiness,
                            "context": claim.context
                        } for claim in top_claims
                    ]
                },
                "phase3": {
                    "status": "completed",
                    "processing_time": phase3_time,
                    "total_evidence": len(all_evidence),
                    "evidence_sources": list(set([ev.source for ev in all_evidence]))
                },
                "phase4": {
                    "status": "completed",
                    "processing_time": phase4_time,
                    "verification_results": [
                        {
                            "claim_text": result["claim"].text,
                            "confidence_score": result["verification"].confidence_score,
                            "stance": result["verification"].stance,
                            "verdict": result["verdict"],
                            "highlights": result["highlights"].highlighted_spans if result["highlights"] else []
                        } for result in verification_results
                    ]
                },
                "phase5": {
                    "status": "completed",
                    "processing_time": phase5_time,
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
            },
            "summary": {
                "total_processing_time": total_time,
                "phases_completed": 5,
                "claims_processed": len(top_claims),
                "evidence_retrieved": len(all_evidence),
                "verdicts": [result["verdict"] for result in verification_results],
                "overall_confidence": sum([result["verification"].confidence_score for result in verification_results]) / len(verification_results) if verification_results else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in complete pipeline: {e}")
        return {
            "error": str(e),
            "status": "failed",
            "timestamp": datetime.now().isoformat()
        }

def generate_mock_pipeline_result(input_data: str) -> Dict[str, Any]:
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
                "normalized_text": input_data
            },
            "phase2": {
                "status": "completed",
                "processing_time": 0.2,
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
            },
            "phase3": {
                "status": "completed",
                "processing_time": 0.3,
                "total_evidence": 2,
                "evidence_sources": ["Mock Source 1", "Mock Source 2"]
            },
            "phase4": {
                "status": "completed",
                "processing_time": 0.4,
                "verification_results": [
                    {
                        "claim_text": input_data,
                        "confidence_score": 0.75,
                        "stance": "refutes",
                        "verdict": "Likely False",
                        "highlights": ["highlighted", "text", "spans"]
                    }
                ]
            },
            "phase5": {
                "status": "completed",
                "processing_time": 0.5,
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

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
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
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_complete():
    """Complete pipeline analysis endpoint."""
    try:
        data = request.get_json()
        input_data = data.get('text', '').strip()
        input_type = data.get('type', 'text')
        
        if not input_data:
            return jsonify({"error": "No input provided"}), 400
        
        logger.info(f"Starting complete pipeline analysis for: {input_data[:100]}...")
        
        # Process through complete pipeline
        result = process_complete_pipeline(input_data, input_type)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/pipeline/phase1', methods=['POST'])
def process_input_phase():
    """Phase 1: Input Processing"""
    try:
        data = request.get_json()
        input_data = data.get('text', '').strip()
        input_type = data.get('type', 'text')
        
        if not input_data:
            return jsonify({"error": "No input provided"}), 400
        
        if not TRUTHLENS_AVAILABLE:
            return jsonify({
                "phase": "input_processing",
                "status": "mock",
                "processed_text": input_data,
                "input_type": input_type
            })
        
        start_time = time.time()
        processed_input = process_input(input_data, input_type)
        normalized_text = normalize_text(processed_input)
        processing_time = time.time() - start_time
        
        return jsonify({
            "phase": "input_processing",
            "status": "completed",
            "original_input": input_data,
            "processed_text": normalized_text,
            "input_type": input_type,
            "processing_time": processing_time
        })
        
    except Exception as e:
        logger.error(f"Error in Phase 1: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/pipeline/phase2', methods=['POST'])
def extract_claims_phase():
    """Phase 2: Claim Extraction & Ranking"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        if not TRUTHLENS_AVAILABLE or not claim_processor:
            return jsonify({
                "phase": "claim_extraction",
                "status": "mock",
                "claims": [{"text": text, "checkworthiness": 0.8}]
            })
        
        start_time = time.time()
        result = claim_processor.process_claims(text)
        processing_time = time.time() - start_time
        
        return jsonify({
            "phase": "claim_extraction",
            "status": "completed",
            "total_claims": len(result.claims),
            "claims": [
                {
                    "id": claim.id,
                    "text": claim.text,
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object": claim.object,
                    "checkworthiness": claim.checkworthiness,
                    "context": claim.context
                } for claim in result.claims
            ],
            "processing_time": processing_time
        })
        
    except Exception as e:
        logger.error(f"Error in Phase 2: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/pipeline/phase3', methods=['POST'])
def retrieve_evidence_phase():
    """Phase 3: Evidence Retrieval"""
    try:
        data = request.get_json()
        claim_text = data.get('claim', '').strip()
        
        if not claim_text:
            return jsonify({"error": "No claim provided"}), 400
        
        if not TRUTHLENS_AVAILABLE or not evidence_retriever:
            return jsonify({
                "phase": "evidence_retrieval",
                "status": "mock",
                "evidence": [
                    {
                        "title": f"Evidence for: {claim_text}",
                        "content": "Mock evidence content",
                        "source": "Mock Source",
                        "url": "https://example.com"
                    }
                ]
            })
        
        start_time = time.time()
        result = evidence_retriever.retrieve_evidence(claim_text, "api_claim")
        processing_time = time.time() - start_time
        
        return jsonify({
            "phase": "evidence_retrieval",
            "status": "completed",
            "claim": claim_text,
            "total_evidence": len(result.evidence),
            "evidence": [
                {
                    "id": ev.id,
                    "title": ev.title,
                    "content": ev.content,
                    "source": ev.source,
                    "url": ev.url,
                    "relevance_score": ev.relevance_score
                } for ev in result.evidence
            ],
            "processing_time": processing_time
        })
        
    except Exception as e:
        logger.error(f"Error in Phase 3: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/pipeline/phase4', methods=['POST'])
def verify_claims_phase():
    """Phase 4: Verification & Scoring"""
    try:
        data = request.get_json()
        claim_text = data.get('claim', '').strip()
        evidence_list = data.get('evidence', [])
        
        if not claim_text:
            return jsonify({"error": "No claim provided"}), 400
        
        if not TRUTHLENS_AVAILABLE or not verification_pipeline:
            return jsonify({
                "phase": "verification",
                "status": "mock",
                "confidence_score": 0.75,
                "stance": "refutes",
                "verdict": "Likely False"
            })
        
        start_time = time.time()
        
        # Extract evidence content
        evidence_content = [ev.get('content', '') for ev in evidence_list]
        
        # Verify claim
        verification_result = verification_pipeline.verify_claim(claim_text, evidence_content)
        
        # Map verdict
        verdict = map_verdict(verification_result.confidence_score)
        
        # Highlight citations
        highlights = None
        if evidence_content and citation_highlighter:
            highlights = citation_highlighter.highlight_citation(evidence_content[0], claim_text)
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "phase": "verification",
            "status": "completed",
            "claim": claim_text,
            "confidence_score": verification_result.confidence_score,
            "stance": verification_result.stance,
            "verdict": verdict,
            "highlights": highlights.highlighted_spans if highlights else [],
            "processing_time": processing_time
        })
        
    except Exception as e:
        logger.error(f"Error in Phase 4: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/pipeline/phase5', methods=['POST'])
def generate_explanations_phase():
    """Phase 5: User Explanation Layer"""
    try:
        data = request.get_json()
        claim_text = data.get('claim', '').strip()
        confidence_score = data.get('confidence_score', 0.5)
        verdict = data.get('verdict', 'Unclear')
        
        if not claim_text:
            return jsonify({"error": "No claim provided"}), 400
        
        if not TRUTHLENS_AVAILABLE or not explanation_layer:
            return jsonify({
                "phase": "user_explanation",
                "status": "mock",
                "explanation": {
                    "why_misleading": "This claim lacks scientific evidence",
                    "manipulation_cues": ["fear appeal"],
                    "prebunk_tip": "Always check multiple sources"
                }
            })
        
        start_time = time.time()
        
        # Generate explanation
        explanation = generate_user_explanation(claim_text, confidence_score, verdict)
        
        # Generate cue badges
        cues = generate_cue_badges(claim_text)
        
        # Generate prebunk card
        prebunk = build_prebunk_card(claim_text, explanation.why_misleading, explanation.prebunk_tip)
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "phase": "user_explanation",
            "status": "completed",
            "claim": claim_text,
            "explanation": {
                "why_misleading": explanation.why_misleading,
                "manipulation_cues": explanation.manipulation_cues,
                "missing_context": explanation.missing_context,
                "prebunk_tip": explanation.prebunk_tip
            },
            "cue_badges": cues,
            "prebunk_card": prebunk,
            "processing_time": processing_time
        })
        
    except Exception as e:
        logger.error(f"Error in Phase 5: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/pipeline/status', methods=['GET'])
def pipeline_status():
    """Get pipeline component status."""
    return jsonify({
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
    })

if __name__ == '__main__':
    print("🚀 Starting TruthLens Complete Pipeline API")
    print("=" * 60)
    
    # Initialize pipeline components
    initialize_truthlens_pipeline()
    
    # Start the server
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Server starting on port {port}")
    print(f"📊 Health check: http://localhost:{port}/api/health")
    print(f"🔍 Pipeline status: http://localhost:{port}/api/pipeline/status")
    print(f"🎯 Complete analysis: POST http://localhost:{port}/api/analyze")
    
    app.run(host='0.0.0.0', port=port, debug=True)
