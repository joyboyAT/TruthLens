#!/usr/bin/env python3
"""
TruthLens Backend with Fine-tuned RoBERTa-base Model
Integrates the fine-tuned model for improved claim detection accuracy
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
    from extractor.pipeline import process_text
    from extractor.claim_detector import is_claim
    from extractor.claim_extractor import extract_claim_spans
    from extractor.atomicizer import to_atomic
    from extractor.context import analyze_context
    from extractor.ranker import score_claim
    
    # Import evidence retrieval modules
    from src.evidence_retrieval.vector_search import VectorEvidenceRetriever
    from src.evidence_retrieval.grounded_search import SerperClient, BingClient
    from src.schemas.evidence import Evidence, SourceType
    
    # Import input processing modules
    from src.ingestion import process_input, detect_input_type
    from src.translation import normalize_text
    
    TRUTHLENS_AVAILABLE = True
    print("✅ TruthLens models loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import TruthLens models: {e}")
    print("Falling back to enhanced mock data")
    TRUTHLENS_AVAILABLE = False

# Import fine-tuned model
try:
    from integrate_fine_tuned_roberta_base import FineTunedRoBERTaBaseDetector
    FINE_TUNED_AVAILABLE = True
    print("✅ Fine-tuned RoBERTa-base model available")
except ImportError as e:
    print(f"⚠️ Warning: Could not import fine-tuned model: {e}")
    FINE_TUNED_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize TruthLens components
vector_retriever = None
search_client = None
fine_tuned_detector = None

def initialize_truthlens():
    """Initialize TruthLens components if available."""
    global vector_retriever, search_client, fine_tuned_detector
    
    if not TRUTHLENS_AVAILABLE:
        return
    
    try:
        # Initialize fine-tuned model
        if FINE_TUNED_AVAILABLE:
            print("🤖 Loading fine-tuned RoBERTa-base model...")
            fine_tuned_detector = FineTunedRoBERTaBaseDetector()
            print("✅ Fine-tuned model loaded successfully")
        
        # Initialize vector retriever
        print("🔍 Initializing vector evidence retriever...")
        vector_retriever = VectorEvidenceRetriever(
            model_name="all-MiniLM-L6-v2",
            index_path=parent_dir / "data/processed/faiss_index.bin",
            embeddings_path=parent_dir / "data/processed/embeddings_cache.pkl"
        )
        print("✅ Vector retriever initialized")
        
        # Initialize search client (if API key available)
        serper_key = os.getenv('SERPER_API_KEY')
        bing_key = os.getenv('BING_API_KEY')
        
        if serper_key:
            search_client = SerperClient(serper_key)
            print("✅ Serper search client initialized")
        elif bing_key:
            search_client = BingClient(bing_key)
            print("✅ Bing search client initialized")
        else:
            print("⚠️ No search API keys found - will use vector search only")
            
    except Exception as e:
        print(f"❌ Error initializing TruthLens components: {e}")
        logger.error(f"TruthLens initialization error: {e}")

def extract_claims_with_fine_tuned_model(text: str) -> List[Dict[str, Any]]:
    """Extract claims using fine-tuned model for better accuracy."""
    if not TRUTHLENS_AVAILABLE:
        return []
    
    try:
        # Import components directly
        from extractor.claim_extractor import extract_claim_spans
        from extractor.atomicizer import to_atomic
        from extractor.context import analyze_context
        from extractor.ranker import score_claim
        
        results = []
        
        # Split into sentences
        def _normalize(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").strip())
        
        def _split_sentences(doc: str):
            text = _normalize(doc)
            if not text:
                return []
            parts = re.split(r"(?<=[.!?])\s+", text)
            return [p.strip() for p in parts if p.strip()]
        
        sentences = _split_sentences(text)
        if not sentences:
            sentences = [text]  # If no sentence splitting, use the whole text
        
        for sent in sentences:
            # Use fine-tuned model for claim detection if available
            is_checkworthy = True
            confidence = 0.5
            
            if fine_tuned_detector:
                try:
                    is_checkworthy, confidence = fine_tuned_detector.predict(sent)
                    logger.info(f"Fine-tuned model: '{sent[:50]}...' -> Check-worthy: {is_checkworthy}, Confidence: {confidence:.3f}")
                except Exception as e:
                    logger.warning(f"Fine-tuned model failed for sentence: {e}")
                    # Fallback to base model
                    is_checkworthy, confidence = is_claim(sent)
            else:
                # Fallback to base model
                is_checkworthy, confidence = is_claim(sent)
            
            # Only process if it's a check-worthy claim
            if is_checkworthy:
                # Extract claim spans
                spans = extract_claim_spans(sent)
                if not spans:
                    spans = [{"text": sent, "start": 0, "end": len(sent), "conf": confidence}]
                
                for sp in spans:
                    atomic_claims = to_atomic(sp.get("text", ""), None) or []
                    if not atomic_claims:
                        atomic_claims = [{"text": sp.get("text", ""), "subject": "", "predicate": "", "object": ""}]
                    
                    for ac in atomic_claims:
                        claim_text = _normalize(ac.get("text", ""))
                        ctx = analyze_context(claim_text, sent)
                        score = score_claim(claim_text)
                        
                        # Use fine-tuned confidence if available
                        final_confidence = confidence if fine_tuned_detector else float(max(0.0, min(1.0, score)))
                        
                        results.append({
                            "id": str(uuid.uuid4()),
                            "text": claim_text,
                            "subject": _normalize(ac.get("subject", "")),
                            "predicate": _normalize(ac.get("predicate", "")),
                            "object": _normalize(ac.get("object", "")),
                            "checkworthiness": final_confidence,
                            "context": ctx,
                            "detection_method": "fine_tuned" if fine_tuned_detector else "base_model"
                        })
        
        return results
    except Exception as e:
        logger.error(f"Error extracting claims: {e}")
        return []

def search_evidence_with_truthlens(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search for evidence using TruthLens components."""
    evidence_results = []
    
    try:
        # Try vector search first
        if vector_retriever:
            try:
                vector_results = vector_retriever.search(query, max_results=max_results)
                for result in vector_results:
                    evidence_results.append({
                        "id": f"vector_{len(evidence_results)}",
                        "title": result.evidence.title,
                        "content": result.evidence.content,
                        "url": result.evidence.url,
                        "source_type": result.evidence.source_type.value,
                        "relevance_score": result.similarity_score,
                        "stance": "supports",  # Default stance
                        "confidence": min(0.95, result.similarity_score)
                    })
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        
        # Try web search if available
        if search_client and len(evidence_results) < max_results:
            try:
                search_results = search_client.search(query, num_results=max_results - len(evidence_results))
                for result in search_results:
                    evidence_results.append({
                        "id": f"web_{len(evidence_results)}",
                        "title": result.title,
                        "content": result.snippet,
                        "url": result.url,
                        "source_type": "news",  # Default to news
                        "relevance_score": 0.8,  # Default score
                        "stance": "supports",  # Default stance
                        "confidence": 0.7  # Default confidence
                    })
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
        
    except Exception as e:
        logger.error(f"Error searching evidence: {e}")
    
    return evidence_results

def generate_mock_evidence(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Generate mock evidence for testing."""
    mock_evidence = [
        {
            "id": "mock_1",
            "title": f"Fact-check: {query}",
            "content": f"This is a comprehensive fact-check of the claim: '{query}'. Multiple sources have been consulted to verify this information.",
            "url": "https://example.com/fact-check",
            "source_type": "fact_check",
            "relevance_score": 0.95,
            "stance": "refutes",
            "confidence": 0.9
        },
        {
            "id": "mock_2",
            "title": f"Scientific Study on {query}",
            "content": f"A peer-reviewed study published in a reputable journal examines the claim: '{query}'. The findings provide evidence-based analysis.",
            "url": "https://example.com/study",
            "source_type": "academic",
            "relevance_score": 0.88,
            "stance": "supports",
            "confidence": 0.85
        },
        {
            "id": "mock_3",
            "title": f"Expert Analysis: {query}",
            "content": f"Leading experts in the field have analyzed the claim: '{query}'. Their professional assessment provides valuable insights.",
            "url": "https://example.com/expert-analysis",
            "source_type": "expert",
            "relevance_score": 0.82,
            "stance": "unclear",
            "confidence": 0.75
        }
    ]
    
    return mock_evidence[:max_results]

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "truthlens_available": TRUTHLENS_AVAILABLE,
        "fine_tuned_available": FINE_TUNED_AVAILABLE,
        "vector_retriever": vector_retriever is not None,
        "search_client": search_client is not None
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Analyze text for claims and retrieve evidence."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        logger.info(f"Analyzing text: {text[:100]}...")
        
        # Extract claims
        start_time = time.time()
        claims = extract_claims_with_fine_tuned_model(text)
        extraction_time = time.time() - start_time
        
        logger.info(f"Extracted {len(claims)} claims in {extraction_time:.3f}s")
        
        # Get evidence for each claim
        all_evidence = []
        for claim in claims:
            claim_text = claim.get('text', '')
            if claim_text:
                # Search for evidence
                evidence = search_evidence_with_truthlens(claim_text, max_results=3)
                if not evidence and not TRUTHLENS_AVAILABLE:
                    # Fallback to mock evidence
                    evidence = generate_mock_evidence(claim_text, max_results=3)
                
                claim['evidence'] = evidence
                all_evidence.extend(evidence)
        
        # Prepare response
        response = {
            "input_text": text,
            "claims": claims,
            "total_claims": len(claims),
            "total_evidence": len(all_evidence),
            "processing_time": extraction_time,
            "model_info": {
                "truthlens_available": TRUTHLENS_AVAILABLE,
                "fine_tuned_available": FINE_TUNED_AVAILABLE,
                "detection_method": "fine_tuned" if fine_tuned_detector else "base_model"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/claims', methods=['POST'])
def extract_claims():
    """Extract claims from text."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        logger.info(f"Extracting claims from: {text[:100]}...")
        
        start_time = time.time()
        claims = extract_claims_with_fine_tuned_model(text)
        extraction_time = time.time() - start_time
        
        response = {
            "input_text": text,
            "claims": claims,
            "total_claims": len(claims),
            "processing_time": extraction_time,
            "model_info": {
                "truthlens_available": TRUTHLENS_AVAILABLE,
                "fine_tuned_available": FINE_TUNED_AVAILABLE,
                "detection_method": "fine_tuned" if fine_tuned_detector else "base_model"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in claims endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/evidence', methods=['POST'])
def search_evidence():
    """Search for evidence for a claim."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        max_results = data.get('max_results', 5)
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        logger.info(f"Searching evidence for: {query}")
        
        start_time = time.time()
        evidence = search_evidence_with_truthlens(query, max_results=max_results)
        search_time = time.time() - start_time
        
        if not evidence and not TRUTHLENS_AVAILABLE:
            evidence = generate_mock_evidence(query, max_results=max_results)
        
        response = {
            "query": query,
            "evidence": evidence,
            "total_results": len(evidence),
            "processing_time": search_time,
            "model_info": {
                "truthlens_available": TRUTHLENS_AVAILABLE,
                "vector_retriever": vector_retriever is not None,
                "search_client": search_client is not None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in evidence endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Get model status and capabilities."""
    return jsonify({
        "truthlens_available": TRUTHLENS_AVAILABLE,
        "fine_tuned_available": FINE_TUNED_AVAILABLE,
        "fine_tuned_model": {
            "name": "RoBERTa-base (Fine-tuned)",
            "dataset": "Nithiwat/claim-detection",
            "accuracy": "91.94% F1 Score",
            "status": "loaded" if fine_tuned_detector else "not_available"
        },
        "vector_retriever": {
            "status": "loaded" if vector_retriever else "not_available",
            "model": "all-MiniLM-L6-v2"
        },
        "search_client": {
            "status": "loaded" if search_client else "not_available",
            "type": "Serper" if hasattr(search_client, 'serper_key') else "Bing" if hasattr(search_client, 'bing_key') else "none"
        },
        "capabilities": {
            "claim_detection": True,
            "evidence_retrieval": vector_retriever is not None or search_client is not None,
            "fine_tuned_detection": fine_tuned_detector is not None
        }
    })

if __name__ == '__main__':
    print("🚀 Starting TruthLens Backend with Fine-tuned Model")
    print("=" * 60)
    
    # Initialize components
    initialize_truthlens()
    
    # Start the server
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Server starting on port {port}")
    print(f"📊 Health check: http://localhost:{port}/api/health")
    print(f"🔍 Model status: http://localhost:{port}/api/model-status")
    
    app.run(host='0.0.0.0', port=port, debug=True)
