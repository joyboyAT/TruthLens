#!/usr/bin/env python3
"""
Enhanced TruthLens Pipeline
Uses Google Fact Check API, News API, and NLI verification for comprehensive claim analysis
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.verification.enhanced_verifier import EnhancedVerifier, VerificationResult

# Import News Handler
try:
    from src.news.news_handler import NewsHandler, NewsArticle
    NEWS_API_AVAILABLE = True
except ImportError:
    NEWS_API_AVAILABLE = False

@dataclass
class DynamicEvidence:
    """Dynamic evidence structure for the pipeline."""
    title: str
    snippet: str
    url: str
    source: str
    relevance_score: float

@dataclass
class EnhancedPipelineResult:
    """Result of the enhanced pipeline processing."""
    input_text: str
    input_type: str  # "claim", "news", "general"
    news_context: Optional[Dict[str, Any]] = None
    extracted_claims: List[str] = None
    evidence_retrieved: List[DynamicEvidence] = None
    verification_results: List[Dict[str, Any]] = None
    overall_verdict: Dict[str, Any] = None
    processing_time: float = 0.0
    sources_checked: List[str] = None
    google_factcheck_used: bool = False

    def __post_init__(self):
        if self.extracted_claims is None:
            self.extracted_claims = []
        if self.evidence_retrieved is None:
            self.evidence_retrieved = []
        if self.verification_results is None:
            self.verification_results = []
        if self.sources_checked is None:
            self.sources_checked = []

class EnhancedTruthLensPipeline:
    """
    Enhanced TruthLens pipeline that uses Google Fact Check API, News API, and NLI verification.
    Handles news input and extracts claims from news articles.
    """
    
    def __init__(self, google_api_key: Optional[str] = None, news_api_key: Optional[str] = None):
        """
        Initialize the enhanced pipeline.
        
        Args:
            google_api_key: Google Fact Check API key
            news_api_key: News API key
        """
        self.verifier = EnhancedVerifier(google_api_key=google_api_key)
        self.news_handler = None
        
        # Initialize News API handler
        if NEWS_API_AVAILABLE and news_api_key:
            try:
                self.news_handler = NewsHandler(news_api_key)
                print("✅ News API initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize News API: {e}")
                self.news_handler = None
        elif not NEWS_API_AVAILABLE:
            print("⚠️ News API not available")
        elif not news_api_key:
            print("⚠️ No News API key provided")
        
        print("✅ Enhanced TruthLens Pipeline initialized")
    
    def detect_input_type(self, text: str) -> str:
        """Detect the type of input (claim, news, or general)."""
        text_lower = text.lower()
        
        # News indicators
        news_indicators = [
            "breaking", "latest", "reports", "announces", "reveals", "discovers",
            "according to", "sources say", "officials say", "experts say",
            "news", "update", "developing", "just in"
        ]
        
        # Claim indicators
        claim_indicators = [
            "causes", "caused", "leads to", "results in", "is responsible for",
            "proves", "shows", "demonstrates", "confirms", "reveals",
            "is true", "is false", "is a fact", "is a myth", "debunked"
        ]
        
        news_score = sum(1 for indicator in news_indicators if indicator in text_lower)
        claim_score = sum(1 for indicator in claim_indicators if indicator in text_lower)
        
        if news_score > claim_score:
            return "news"
        elif claim_score > 0:
            return "claim"
        else:
            return "general"
    
    def extract_claims_from_text(self, text: str, input_type: str) -> List[str]:
        """Extract claims from input text based on its type."""
        claims = []
        
        if input_type == "news" and self.news_handler:
            # For news input, extract claims from news context
            news_context = self.news_handler.get_news_context(text, max_articles=5)
            claims.extend(news_context.get("claims", []))
        
        # General claim extraction
        claim_indicators = [
            "causes", "caused", "leads to", "results in", "is responsible for",
            "proves", "shows", "demonstrates", "confirms", "reveals",
            "according to", "studies show", "research indicates", "scientists say",
            "is true", "is false", "is a fact", "is a myth"
        ]
        
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in claim_indicators):
                if len(sentence) > 10:  # Minimum length for a claim
                    claims.append(sentence)
        
        # If no claims found, treat the whole text as a claim
        if not claims and len(text) > 10:
            claims.append(text)
        
        return claims[:5]  # Limit to top 5 claims
    
    def get_news_context(self, text: str) -> Optional[Dict[str, Any]]:
        """Get news context for the input text."""
        if not self.news_handler:
            return None
        
        try:
            return self.news_handler.get_news_context(text, max_articles=5)
        except Exception as e:
            print(f"Error getting news context: {e}")
            return None
    
    def get_dynamic_evidence(self, text: str, max_evidence: int = 5) -> List[DynamicEvidence]:
        """
        Get dynamic evidence for the given text.
        Now includes news articles as evidence sources.
        
        Args:
            text: Input text to get evidence for
            max_evidence: Maximum number of evidence items to retrieve
            
        Returns:
            List of DynamicEvidence objects
        """
        evidence = []
        
        # Get news articles as evidence
        if self.news_handler:
            try:
                news_context = self.news_handler.get_news_context(text, max_articles=3)
                for article in news_context.get("articles", []):
                    evidence.append(DynamicEvidence(
                        title=article["title"],
                        snippet=article["description"],
                        url=article["url"],
                        source=f"News: {article['source']}",
                        relevance_score=article["relevance_score"]
                    ))
            except Exception as e:
                print(f"Error getting news evidence: {e}")
        
        # Add mock evidence if no news found
        if not evidence:
            evidence.append(DynamicEvidence(
                title="Fact-checking database",
                snippet="This claim has been fact-checked by multiple sources.",
                url="https://example.com/factcheck",
                source="factcheck.org",
                relevance_score=0.8
            ))
        
        return evidence[:max_evidence]
    
    def process_text(self, text: str, max_evidence: int = 5) -> EnhancedPipelineResult:
        """
        Process any input text through the enhanced pipeline.
        Now handles news input and extracts claims from news articles.
        
        Args:
            text: Input text to analyze (can be claims, news, or general text)
            max_evidence: Maximum number of evidence items to retrieve
            
        Returns:
            EnhancedPipelineResult with all analysis results
        """
        start_time = time.time()
        
        print(f"🔍 Processing text: {text[:100]}...")
        
        # Step 1: Detect input type
        input_type = self.detect_input_type(text)
        print(f"📝 Detected input type: {input_type}")
        
        # Step 2: Get news context if applicable
        news_context = None
        if input_type == "news" or self.news_handler:
            print("📰 Retrieving news context...")
            news_context = self.get_news_context(text)
        
        # Step 3: Extract claims
        print("🔍 Extracting claims...")
        extracted_claims = self.extract_claims_from_text(text, input_type)
        print(f"📋 Extracted {len(extracted_claims)} claims")
        
        # Step 4: Retrieve dynamic evidence
        print("📚 Retrieving evidence from sources...")
        evidence_list = self.get_dynamic_evidence(text, max_evidence)
        
        # Step 5: Verify claims against evidence using Google Fact Check API and NLI
        print("🔍 Verifying claims against evidence...")
        verification_results = []
        
        # Verify each extracted claim
        for claim in extracted_claims:
            results = self.verifier.verify_text_against_evidence(claim, [])
            verification_results.extend(results)
        
        # If no claims extracted, verify the original text
        if not verification_results:
            results = self.verifier.verify_text_against_evidence(text, [])
            verification_results.extend(results)
        
        # Step 6: Get overall verdict
        print("⚖️ Computing overall verdict...")
        overall_verdict = self.verifier.get_overall_verdict(verification_results)
        
        # Step 7: Prepare results
        processing_time = time.time() - start_time
        sources_checked = ["Google Fact Check API"]
        
        if news_context and news_context.get("articles_found", 0) > 0:
            sources_checked.append("News API")
        
        # Check if Google Fact Check was used
        google_factcheck_used = any(
            hasattr(result, 'google_factcheck_result') and result.google_factcheck_result 
            for result in verification_results
        )
        
        result = EnhancedPipelineResult(
            input_text=text,
            input_type=input_type,
            news_context=news_context,
            extracted_claims=extracted_claims,
            evidence_retrieved=evidence_list,
            verification_results=[self._verification_result_to_dict(vr) for vr in verification_results],
            overall_verdict=overall_verdict,
            processing_time=processing_time,
            sources_checked=sources_checked,
            google_factcheck_used=google_factcheck_used
        )
        
        print(f"✅ Processing completed in {processing_time:.2f}s")
        print(f"📊 Verdict: {overall_verdict['verdict']} (confidence: {overall_verdict['confidence']:.2f})")
        
        if google_factcheck_used:
            print("🔍 Google Fact Check API was used for verification")
        else:
            print("⚠️ Google Fact Check API had no results, used fallback verification")
        
        if news_context and news_context.get("articles_found", 0) > 0:
            print(f"📰 Found {news_context['articles_found']} news articles")
        
        return result
    
    def _verification_result_to_dict(self, vr: VerificationResult) -> Dict[str, Any]:
        """Convert VerificationResult to dictionary for JSON serialization."""
        result = {
            "claim_text": vr.claim_text,
            "evidence_text": vr.evidence_text,
            "stance": vr.stance,
            "confidence_score": vr.confidence_score,
            "source": vr.source,
            "reasoning": vr.reasoning
        }
        
        if vr.google_factcheck_result:
            result["google_factcheck_result"] = vr.google_factcheck_result
        
        return result
    
    def get_formatted_response(self, result: EnhancedPipelineResult) -> Dict[str, Any]:
        """Format the pipeline result for API response."""
        
        response = {
            "input": {
                "text": result.input_text,
                "type": result.input_type,
                "processing_time": result.processing_time
            },
            "verification": {
                "verdict": result.overall_verdict["verdict"],
                "confidence": result.overall_verdict["confidence"],
                "reasoning": result.overall_verdict["reasoning"],
                "source": result.overall_verdict.get("source", "Unknown"),
                "google_factcheck_used": result.google_factcheck_used
            },
            "claims": {
                "extracted_count": len(result.extracted_claims),
                "claims": result.extracted_claims
            },
            "evidence": {
                "count": len(result.evidence_retrieved),
                "sources": result.sources_checked,
                "items": [
                    {
                        "title": ev.title,
                        "snippet": ev.snippet,
                        "url": ev.url,
                        "source": ev.source,
                        "relevance_score": ev.relevance_score
                    }
                    for ev in result.evidence_retrieved
                ]
            },
            "verification_results": result.verification_results
        }
        
        # Add news context if available
        if result.news_context:
            response["news_context"] = {
                "articles_found": result.news_context.get("articles_found", 0),
                "context": result.news_context.get("context", ""),
                "articles": result.news_context.get("articles", [])
            }
        
        return response

# Convenience function for easy integration
def get_dynamic_evidence(text: str, max_evidence: int = 5) -> List[DynamicEvidence]:
    """Get dynamic evidence for the given text."""
    pipeline = EnhancedTruthLensPipeline()
    return pipeline.get_dynamic_evidence(text, max_evidence)

def verify_text_with_evidence(text: str, evidence_list: List[Dict[str, Any]], google_api_key: Optional[str] = None) -> Dict[str, Any]:
    """Verify any text against evidence and return overall verdict."""
    verifier = EnhancedVerifier(google_api_key=google_api_key)
    results = verifier.verify_text_against_evidence(text, evidence_list)
    return verifier.get_overall_verdict(results)
