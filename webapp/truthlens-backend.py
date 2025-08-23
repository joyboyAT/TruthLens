#!/usr/bin/env python3
"""
TruthLens Model Backend Server
Integrates actual TruthLens models for claim extraction and evidence retrieval
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

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
    
    TRUTHLENS_AVAILABLE = True
    print("✅ TruthLens models loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import TruthLens models: {e}")
    print("Falling back to enhanced mock data")
    TRUTHLENS_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize TruthLens components
vector_retriever = None
search_client = None

def initialize_truthlens():
    """Initialize TruthLens components if available."""
    global vector_retriever, search_client
    
    if not TRUTHLENS_AVAILABLE:
        return
    
    try:
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

def extract_claims_with_truthlens(text: str) -> List[Dict[str, Any]]:
    """Extract claims using TruthLens pipeline."""
    if not TRUTHLENS_AVAILABLE:
        return []
    
    try:
        # Use the actual TruthLens pipeline
        atomic_claims = process_text(text)
        
        # Convert to the format expected by frontend
        claims = []
        for claim in atomic_claims:
            claims.append({
                "id": claim["id"],
                "text": claim["text"],
                "subject": claim["subject"],
                "predicate": claim["predicate"],
                "object": claim["object"],
                "checkworthiness": claim["checkworthiness"],
                "context": claim["context"]
            })
        
        return claims
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

def analyze_manipulation(claims: List[Dict[str, Any]], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze potential manipulation based on claims and evidence."""
    manipulation_detected = False
    manipulation_types = []
    
    if not claims:
        return {
            "manipulation_detected": False,
            "manipulation_types": []
        }
    
    # Check for common manipulation patterns
    for claim in claims:
        text = claim.get("text", "").lower()
        checkworthiness = claim.get("checkworthiness", 0.5)
        
        # High checkworthiness but no supporting evidence
        if checkworthiness > 0.8 and not evidence:
            manipulation_detected = True
            manipulation_types.append("Unsupported high-stakes claims")
        
        # Check for common misinformation keywords
        misinformation_keywords = [
            "conspiracy", "hoax", "fake", "false flag", "cover up",
            "secret", "hidden", "they don't want you to know",
            "mainstream media lies", "government lies"
        ]
        
        for keyword in misinformation_keywords:
            if keyword in text:
                manipulation_detected = True
                manipulation_types.append("Conspiracy language")
                break
    
    return {
        "manipulation_detected": manipulation_detected,
        "manipulation_types": list(set(manipulation_types))
    }

def determine_verdict(claims: List[Dict[str, Any]], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Determine overall verdict based on claims and evidence."""
    if not claims:
        return {
            "overall_verdict": "Unverified",
            "confidence": 0.5
        }
    
    # Calculate average checkworthiness
    avg_checkworthiness = sum(c.get("checkworthiness", 0.5) for c in claims) / len(claims)
    
    # Count supporting vs contradicting evidence
    supporting_evidence = sum(1 for e in evidence if e.get("stance") == "supports")
    contradicting_evidence = sum(1 for e in evidence if e.get("stance") == "contradicts")
    
    # Determine verdict
    if not evidence:
        if avg_checkworthiness > 0.7:
            verdict = "Likely True"
            confidence = min(0.8, avg_checkworthiness)
        else:
            verdict = "Unverified"
            confidence = 0.5
    elif supporting_evidence > contradicting_evidence:
        verdict = "Likely True"
        confidence = min(0.9, avg_checkworthiness + 0.1)
    elif contradicting_evidence > supporting_evidence:
        verdict = "Likely False"
        confidence = min(0.9, 0.9 - avg_checkworthiness + 0.1)
    else:
        verdict = "Mixed Evidence"
        confidence = 0.6
    
    return {
        "overall_verdict": verdict,
        "confidence": confidence
    }

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "truthlens_available": TRUTHLENS_AVAILABLE,
        "vector_retriever": vector_retriever is not None,
        "search_client": search_client is not None
    })

@app.route('/api/v1/fact-check', methods=['POST'])
def fact_check():
    """Main fact-checking endpoint using TruthLens models."""
    try:
        data = request.get_json()
        input_type = data.get('input_type', 'text')
        content = data.get('content', '')
        
        if not content:
            return jsonify({"error": "No content provided"}), 400
        
        logger.info(f"Processing fact-check request: {input_type} - {content[:100]}...")
        
        # Extract claims using TruthLens
        claims = extract_claims_with_truthlens(content)
        
        # Search for evidence
        evidence = []
        if claims:
            # Use the first claim as search query
            search_query = claims[0].get("text", content[:100])
            evidence = search_evidence_with_truthlens(search_query, max_results=5)
        
        # Analyze manipulation
        manipulation_analysis = analyze_manipulation(claims, evidence)
        
        # Determine verdict
        verdict_analysis = determine_verdict(claims, evidence)
        
        result = {
            "claims": claims,
            "evidence": evidence,
            "manipulation_detected": manipulation_analysis["manipulation_detected"],
            "manipulation_types": manipulation_analysis["manipulation_types"],
            "overall_verdict": verdict_analysis["overall_verdict"],
            "confidence": verdict_analysis["confidence"],
            "processing_time": datetime.now().isoformat()
        }
        
        logger.info(f"Fact-check completed: {verdict_analysis['overall_verdict']} ({verdict_analysis['confidence']:.2f})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Fact-check error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/extract-claims', methods=['POST'])
def extract_claims():
    """Extract claims from text."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        claims = extract_claims_with_truthlens(text)
        return jsonify({"claims": claims})
        
    except Exception as e:
        logger.error(f"Claim extraction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/search-evidence', methods=['POST'])
def search_evidence():
    """Search for evidence."""
    try:
        data = request.get_json()
        query = data.get('query', '')
        max_results = data.get('max_results', 5)
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        evidence = search_evidence_with_truthlens(query, max_results)
        return jsonify({"evidence": evidence})
        
    except Exception as e:
        logger.error(f"Evidence search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/languages', methods=['GET'])
def get_languages():
    """Get supported languages."""
    return jsonify({
        "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
    })

@app.route('/api/v1/trusted-sources', methods=['GET'])
def get_trusted_sources():
    """Get trusted sources."""
    return jsonify({
        "government": [
            "whitehouse.gov", "gov.uk", "canada.ca", "australia.gov.au",
            "who.int", "cdc.gov", "fda.gov", "nih.gov"
        ],
        "news": [
            "reuters.com", "ap.org", "bbc.com", "npr.org",
            "pbs.org", "abc.net.au", "cbc.ca"
        ],
        "academic": [
            "nature.com", "science.org", "arxiv.org", "scholar.google.com",
            "pubmed.ncbi.nlm.nih.gov", "ieee.org"
        ],
        "fact_checkers": [
            "snopes.com", "factcheck.org", "politifact.com", "reuters.com/fact-check",
            "ap.org/fact-check", "bbc.com/news/fact-check"
        ]
    })

if __name__ == '__main__':
    print("🚀 Starting TruthLens Model Backend Server...")
    print("📁 Working directory:", current_dir)
    print("🔧 Parent directory:", parent_dir)
    
    # Initialize TruthLens components
    initialize_truthlens()
    
    print("🌐 Starting Flask server on http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)
