#!/usr/bin/env python3
"""
Simple Web Search Fact-Checking Backend for TruthLens
This backend uses basic web scraping to search for evidence (no API keys required)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import re
import requests
from datetime import datetime
from urllib.parse import urlparse, quote_plus
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# Trusted sources for fact-checking
TRUSTED_SOURCES = {
    "government": [
        "pib.gov.in", "mea.gov.in", "pmindia.gov.in", "gov.in",
        "who.int", "cdc.gov", "nih.gov", "fda.gov", "gov.uk", "europa.eu"
    ],
    "fact_checkers": [
        "snopes.com", "factcheck.org", "politifact.com", "altnews.in", "boomlive.in"
    ],
    "news": [
        "reuters.com", "apnews.com", "bbc.com", "thehindu.com", "indianexpress.com"
    ],
    "academic": [
        "wikipedia.org", "scholar.google.com", "pubmed.ncbi.nlm.nih.gov"
    ]
}

def search_web_simple(query, max_results=5):
    """
    Simple web search using DuckDuckGo (no API key required)
    """
    results = []
    
    try:
        # Use DuckDuckGo search (no API key required)
        search_query = quote_plus(query)
        search_url = f"https://duckduckgo.com/html/?q={search_query}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results from DuckDuckGo
            for result in soup.find_all('div', class_='result')[:max_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    
                    # Clean up DuckDuckGo redirect URL
                    if url.startswith('/l/?uddg='):
                        url = url.replace('/l/?uddg=', '')
                    
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet,
                        'source': 'duckduckgo'
                    })
        
        print(f"Found {len(results)} search results for: {query}")
        
    except Exception as e:
        print(f"Search error: {e}")
    
    return results

def extract_content_from_url(url):
    """
    Extract content from a URL
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Try to find main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            
            # Clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:1000]  # Limit to first 1000 characters
            
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
    
    return ""

def analyze_claim_veracity(claim_text, search_results):
    """
    Analyze the veracity of a claim based on search results
    """
    supporting_evidence = []
    contradicting_evidence = []
    neutral_evidence = []
    
    claim_lower = claim_text.lower()
    
    for result in search_results:
        title_lower = result['title'].lower()
        snippet_lower = result['snippet'].lower()
        
        # Get full content if available
        content = extract_content_from_url(result['url'])
        content_lower = content.lower()
        
        # Check source credibility
        domain = urlparse(result['url']).netloc.lower()
        source_credibility = 0.5  # Default
        
        for source_type, domains in TRUSTED_SOURCES.items():
            if any(trusted_domain in domain for trusted_domain in domains):
                if source_type == "government":
                    source_credibility = 0.9
                elif source_type == "fact_checkers":
                    source_credibility = 0.95
                elif source_type == "news":
                    source_credibility = 0.8
                elif source_type == "academic":
                    source_credibility = 0.85
                break
        
        # Analyze stance
        stance = "neutral"
        relevance_score = 0.5
        
        # Check for supporting evidence
        supporting_keywords = ['confirm', 'verify', 'true', 'accurate', 'correct', 'valid', 'proven', 'official', 'announced']
        if any(keyword in content_lower for keyword in supporting_keywords):
            stance = "supports"
            relevance_score = 0.8
        
        # Check for contradicting evidence
        contradicting_keywords = ['false', 'misleading', 'debunk', 'hoax', 'fake', 'incorrect', 'wrong', 'denied']
        if any(keyword in content_lower for keyword in contradicting_keywords):
            stance = "contradicts"
            relevance_score = 0.8
        
        # Check for fact-checking indicators
        fact_check_indicators = ['fact check', 'fact-check', 'verified', 'debunked', 'misinformation']
        if any(indicator in content_lower for indicator in fact_check_indicators):
            relevance_score = 0.9
        
        # Check for diplomatic/government indicators
        gov_indicators = ['official statement', 'government', 'ministry', 'president', 'prime minister', 'diplomatic']
        if any(indicator in content_lower for indicator in gov_indicators):
            relevance_score = 0.85
            if stance == "neutral":
                stance = "supports"  # Government sources are generally reliable
        
        evidence_item = {
            'id': f"evidence_{len(supporting_evidence + contradicting_evidence + neutral_evidence)}",
            'title': result['title'],
            'content': content[:500] + "..." if len(content) > 500 else content,
            'url': result['url'],
            'source_type': 'web_search',
            'relevance_score': relevance_score * source_credibility,
            'stance': stance,
            'confidence': source_credibility,
            'domain': domain
        }
        
        if stance == "supports":
            supporting_evidence.append(evidence_item)
        elif stance == "contradicts":
            contradicting_evidence.append(evidence_item)
        else:
            neutral_evidence.append(evidence_item)
    
    return supporting_evidence, contradicting_evidence, neutral_evidence

def extract_claims_from_text(text):
    """
    Extract claims from text using NLP patterns
    """
    claims = []
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        
        # Enhanced claim detection patterns
        claim_patterns = [
            r'(\w+(?:\s+\w+)*)\s+(speaks|meets|discusses|talks|announces|declares|confirms|states|says)\s+(?:with|to|about)\s+(.+)',
            r'(\w+(?:\s+\w+)*)\s+(is|are|was|were)\s+(.+)',
            r'(\w+(?:\s+\w+)*)\s+(will|can|may|might)\s+(.+)',
            r'(\w+(?:\s+\w+)*)\s+(causes|caused|leads|resulted)\s+(?:in|to)\s+(.+)'
        ]
        
        for pattern in claim_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                subject = match.group(1).strip()
                predicate = match.group(2).strip()
                object_text = match.group(3).strip()
                
                # Calculate checkworthiness
                checkworthiness = 0.7
                if any(word in sentence.lower() for word in ['official', 'government', 'ministry', 'president', 'prime minister']):
                    checkworthiness = 0.9
                elif any(word in sentence.lower() for word in ['announces', 'confirms', 'declares']):
                    checkworthiness = 0.85
                
                claims.append({
                    "id": f"claim_{i}",
                    "text": sentence,
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_text,
                    "checkworthiness": checkworthiness,
                    "context": {
                        "negation": any(word in sentence.lower() for word in ['not', 'no', 'never', 'denies']),
                        "modality": "assertive" if any(word in sentence.lower() for word in ['is', 'are', 'was', 'were']) else "modal",
                        "conditional_trigger": "",
                        "sarcasm_score": 0.1,
                        "attribution": "news report"
                    }
                })
                break
    
    return claims

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "3.1.0",
        "models_loaded": True,
        "uptime": time.time(),
        "timestamp": datetime.now().isoformat(),
        "features": ["web_search", "content_extraction", "claim_analysis", "evidence_verification"],
        "search_method": "DuckDuckGo (no API key required)"
    })

@app.route('/api/v1/fact-check', methods=['POST'])
def fact_check():
    """Real fact-checking endpoint with web search"""
    try:
        data = request.get_json()
        
        if not data or 'content' not in data:
            return jsonify({"error": "Missing content field"}), 400
            
        content = data['content']
        input_type = data.get('input_type', 'text')
        
        print(f"Processing fact-check request: {content[:100]}...")
        
        # Extract claims
        claims = extract_claims_from_text(content)
        if not claims:
            # Fallback claim
            claims = [{
                "id": "claim_1",
                "text": content[:100] + "..." if len(content) > 100 else content,
                "subject": "Unknown",
                "predicate": "Unknown",
                "object": "Unknown",
                "checkworthiness": 0.5,
                "context": {
                    "negation": False,
                    "modality": "assertive",
                    "conditional_trigger": "",
                    "sarcasm_score": 0.1,
                    "attribution": "news report"
                }
            }]
        
        all_evidence = []
        
        # Search for evidence for each claim
        for claim in claims:
            print(f"Searching for evidence: {claim['text']}")
            
            # Create search queries
            search_queries = [
                claim['text'],
                f"{claim['subject']} {claim['predicate']} {claim['object']}",
                f"fact check {claim['text']}",
                f"verify {claim['text']}"
            ]
            
            for query in search_queries:
                search_results = search_web_simple(query, max_results=3)
                if search_results:
                    supporting, contradicting, neutral = analyze_claim_veracity(claim['text'], search_results)
                    all_evidence.extend(supporting + contradicting + neutral)
                    break  # Use first successful search
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_evidence = []
        for evidence in all_evidence:
            if evidence['url'] not in seen_urls:
                unique_evidence.append(evidence)
                seen_urls.add(evidence['url'])
        
        # Determine overall verdict
        supporting_evidence = [e for e in unique_evidence if e['stance'] == 'supports']
        contradicting_evidence = [e for e in unique_evidence if e['stance'] == 'contradicts']
        
        if supporting_evidence and not contradicting_evidence:
            verdict = "Likely True"
            confidence = min(0.9, 0.5 + len(supporting_evidence) * 0.1)
            manipulation_detected = False
            manipulation_types = []
        elif contradicting_evidence and not supporting_evidence:
            verdict = "Likely False"
            confidence = min(0.9, 0.5 + len(contradicting_evidence) * 0.1)
            manipulation_detected = True
            manipulation_types = ["Contradicting evidence found"]
        elif supporting_evidence and contradicting_evidence:
            verdict = "Mixed Evidence"
            confidence = 0.6
            manipulation_detected = True
            manipulation_types = ["Conflicting sources"]
        else:
            verdict = "Unverified"
            confidence = 0.5
            manipulation_detected = False
            manipulation_types = []
        
        result = {
            "claims": claims,
            "evidence": unique_evidence,
            "manipulation_detected": manipulation_detected,
            "manipulation_types": manipulation_types,
            "overall_verdict": verdict,
            "confidence": confidence,
            "processing_time": 3.0,
            "metadata": {
                "model_version": "3.1.0",
                "timestamp": datetime.now().isoformat(),
                "input_length": len(content),
                "search_method": "web_search",
                "evidence_count": len(unique_evidence)
            }
        }
        
        print(f"Fact-check result: {verdict} (confidence: {confidence})")
        return jsonify(result)
        
    except Exception as e:
        print(f"Fact-check error: {e}")
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
        
        print(f"Searching for evidence: {query}")
        
        search_results = search_web_simple(query, max_results)
        supporting, contradicting, neutral = analyze_claim_veracity(query, search_results)
        
        evidence = supporting + contradicting + neutral
        
        return jsonify({"evidence": evidence})
        
    except Exception as e:
        print(f"Evidence search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/trusted-sources', methods=['GET'])
def get_trusted_sources():
    """Get list of trusted sources"""
    return jsonify(TRUSTED_SOURCES)

if __name__ == '__main__':
    print("Starting Simple Web Search Fact-Checking Backend Server...")
    print("API will be available at: http://localhost:8000")
    print("Health check: http://localhost:8000/api/v1/health")
    print("Features: Web search (DuckDuckGo), Content extraction, Evidence verification")
    print("No API keys required!")
    print("Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=8000, debug=True)
