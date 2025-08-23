#!/usr/bin/env python3
"""
Simple Flask backend server for TruthLens web application
This provides a basic API interface for testing the web app
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Mock data for demonstration
MOCK_CLAIMS = {
    "covid_vaccine_autism": {
        "id": "claim_1",
        "text": "COVID-19 vaccines cause autism in children",
        "subject": "COVID-19 vaccines",
        "predicate": "cause",
        "object": "autism in children",
        "checkworthiness": 0.85,
        "context": {
            "negation": False,
            "modality": "assertive",
            "conditional_trigger": "",
            "sarcasm_score": 0.1,
            "attribution": ""
        }
    },
    "5g_health": {
        "id": "claim_2",
        "text": "5G technology causes health problems",
        "subject": "5G technology",
        "predicate": "causes",
        "object": "health problems",
        "checkworthiness": 0.78,
        "context": {
            "negation": False,
            "modality": "assertive",
            "conditional_trigger": "",
            "sarcasm_score": 0.1,
            "attribution": ""
        }
    },
    "climate_hoax": {
        "id": "claim_3",
        "text": "Climate change is a hoax",
        "subject": "Climate change",
        "predicate": "is",
        "object": "a hoax",
        "checkworthiness": 0.82,
        "context": {
            "negation": False,
            "modality": "assertive",
            "conditional_trigger": "",
            "sarcasm_score": 0.1,
            "attribution": ""
        }
    }
}

MOCK_EVIDENCE = {
    "covid_vaccine_autism": [
        {
            "id": "evidence_1",
            "title": "COVID-19 Vaccines and Autism: No Link Found",
            "content": "Multiple large-scale studies have found no evidence linking COVID-19 vaccines to autism. The CDC, WHO, and FDA all confirm that vaccines are safe and do not cause autism.",
            "url": "https://www.cdc.gov/coronavirus/2019-ncov/vaccines/safety.html",
            "source_type": "government",
            "relevance_score": 0.95,
            "stance": "contradicts",
            "confidence": 0.92
        },
        {
            "id": "evidence_2",
            "title": "Vaccine Safety and Autism: Scientific Evidence",
            "content": "Extensive research has consistently shown that vaccines, including COVID-19 vaccines, do not cause autism spectrum disorder. The original study linking vaccines to autism has been thoroughly debunked.",
            "url": "https://www.who.int/news-room/fact-sheets/detail/autism-spectrum-disorders",
            "source_type": "government",
            "relevance_score": 0.88,
            "stance": "contradicts",
            "confidence": 0.89
        }
    ],
    "5g_health": [
        {
            "id": "evidence_1",
            "title": "5G Technology and Health: Scientific Assessment",
            "content": "The World Health Organization and numerous scientific studies have found no evidence that 5G technology causes adverse health effects. 5G radiation levels are well below international safety guidelines.",
            "url": "https://www.who.int/news-room/q-a-detail/radiation-5g-mobile-networks-and-health",
            "source_type": "government",
            "relevance_score": 0.92,
            "stance": "contradicts",
            "confidence": 0.88
        }
    ],
    "climate_hoax": [
        {
            "id": "evidence_1",
            "title": "Climate Change Evidence: Scientific Consensus",
            "content": "97% of climate scientists agree that climate change is real and primarily caused by human activities. Multiple lines of evidence from temperature records, ice cores, and satellite data confirm this.",
            "url": "https://climate.nasa.gov/evidence/",
            "source_type": "government",
            "relevance_score": 0.95,
            "stance": "contradicts",
            "confidence": 0.94
        }
    ]
}

def detect_claim_type(content):
    """Simple keyword-based claim detection"""
    content_lower = content.lower()
    
    if 'covid' in content_lower and 'vaccine' in content_lower and 'autism' in content_lower:
        return "covid_vaccine_autism"
    elif '5g' in content_lower and 'cause' in content_lower:
        return "5g_health"
    elif 'climate change' in content_lower and 'hoax' in content_lower:
        return "climate_hoax"
    else:
        return "generic"

def extract_claims_from_text(text):
    """Extract claims from text using simple keyword matching"""
    sentences = text.split('.')
    claims = []
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
            
        # Simple claim detection based on keywords
        claim_keywords = ['cause', 'causes', 'caused', 'lead to', 'leads to', 'result in', 'results in', 'is', 'are', 'was', 'were']
        if any(keyword in sentence.lower() for keyword in claim_keywords):
            claims.append({
                "id": f"claim_{i}",
                "text": sentence,
                "subject": "Unknown",
                "predicate": "Unknown", 
                "object": "Unknown",
                "checkworthiness": 0.5,
                "context": {
                    "negation": False,
                    "modality": "assertive",
                    "conditional_trigger": "",
                    "sarcasm_score": 0.1,
                    "attribution": ""
                }
            })
    
    return claims

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": True,
        "uptime": time.time(),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/v1/fact-check', methods=['POST'])
def fact_check():
    """Main fact-checking endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'content' not in data:
            return jsonify({"error": "Missing content field"}), 400
            
        content = data['content']
        input_type = data.get('input_type', 'text')
        options = data.get('options', {})
        
        # Simulate processing time
        time.sleep(2)
        
        # Detect claim type based on content
        claim_type = detect_claim_type(content)
        
        if claim_type == "generic":
            # For generic content, extract claims and return basic analysis
            claims = extract_claims_from_text(content)
            result = {
                "claims": claims,
                "evidence": [],
                "manipulation_detected": False,
                "manipulation_types": [],
                "overall_verdict": "Unverified",
                "confidence": 0.5,
                "processing_time": 2.0,
                "metadata": {
                    "model_version": "1.0.0",
                    "timestamp": datetime.now().isoformat(),
                    "input_length": len(content)
                }
            }
        else:
            # Use predefined mock data for known claim types
            claims = [MOCK_CLAIMS[claim_type]]
            evidence = MOCK_EVIDENCE[claim_type]
            
            # Determine verdict based on evidence stance
            contradicting_evidence = [e for e in evidence if e['stance'] == 'contradicts']
            supporting_evidence = [e for e in evidence if e['stance'] == 'supports']
            
            if contradicting_evidence and not supporting_evidence:
                verdict = "False"
                confidence = 0.9
                manipulation_detected = True
                manipulation_types = ["False causation", "Misleading claims"]
            elif supporting_evidence and not contradicting_evidence:
                verdict = "True"
                confidence = 0.8
                manipulation_detected = False
                manipulation_types = []
            else:
                verdict = "Misleading"
                confidence = 0.7
                manipulation_detected = True
                manipulation_types = ["Mixed evidence"]
            
            result = {
                "claims": claims,
                "evidence": evidence,
                "manipulation_detected": manipulation_detected,
                "manipulation_types": manipulation_types,
                "overall_verdict": verdict,
                "confidence": confidence,
                "processing_time": 2.0,
                "metadata": {
                    "model_version": "1.0.0",
                    "timestamp": datetime.now().isoformat(),
                    "input_length": len(content)
                }
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/extract-claims', methods=['POST'])
def extract_claims():
    """Extract claims from text without full fact-checking"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing text field"}), 400
            
        text = data['text']
        claims = extract_claims_from_text(text)
        
        return jsonify({"claims": claims})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/search-evidence', methods=['POST'])
def search_evidence():
    """Search for evidence related to a query"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "Missing query field"}), 400
            
        query = data['query']
        max_results = data.get('max_results', 5)
        
        # Simple keyword-based evidence search
        all_evidence = []
        for evidence_list in MOCK_EVIDENCE.values():
            all_evidence.extend(evidence_list)
        
        # Filter evidence based on query keywords
        query_lower = query.lower()
        relevant_evidence = []
        
        for evidence in all_evidence:
            title_lower = evidence['title'].lower()
            content_lower = evidence['content'].lower()
            
            if any(keyword in title_lower or keyword in content_lower 
                   for keyword in query_lower.split()):
                relevant_evidence.append(evidence)
        
        # Sort by relevance score and limit results
        relevant_evidence.sort(key=lambda x: x['relevance_score'], reverse=True)
        relevant_evidence = relevant_evidence[:max_results]
        
        return jsonify({"evidence": relevant_evidence})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/languages', methods=['GET'])
def get_languages():
    """Get supported languages"""
    return jsonify({
        "languages": ["en", "hi", "ta", "te", "mr", "bn", "gu", "kn", "ml", "pa", "ur"]
    })

@app.route('/api/v1/trusted-sources', methods=['GET'])
def get_trusted_sources():
    """Get list of trusted sources"""
    return jsonify({
        "government": [
            "who.int",
            "cdc.gov", 
            "nih.gov",
            "fda.gov",
            "gov.in",
            "gov.uk",
            "europa.eu"
        ],
        "news": [
            "reuters.com",
            "apnews.com",
            "ap.org",
            "bbc.com",
            "pib.gov.in"
        ],
        "academic": [
            "wikipedia.org",
            "scholar.google.com",
            "pubmed.ncbi.nlm.nih.gov"
        ],
        "fact_checkers": [
            "snopes.com",
            "factcheck.org",
            "politifact.com"
        ]
    })

if __name__ == '__main__':
    print("Starting TruthLens Backend Server...")
    print("API will be available at: http://localhost:8000")
    print("Health check: http://localhost:8000/api/v1/health")
    print("Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=8000, debug=True)
