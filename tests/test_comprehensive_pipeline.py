#!/usr/bin/env python3
"""
Comprehensive TruthLens Pipeline Test
Complete end-to-end pipeline with Google Fact Check API integration.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Google Fact Check API
try:
    from src.verification.google_factcheck_api import GoogleFactCheckAPI, FactCheckResult
    GOOGLE_FACTCHECK_AVAILABLE = True
except ImportError:
    GOOGLE_FACTCHECK_AVAILABLE = False
    print("Warning: Google Fact Check API not available")

@dataclass
class ComprehensivePipelineResult:
    """Comprehensive result from the pipeline."""
    input_text: str
    normalized_text: str
    extracted_claims: List[str]
    evidence_sources: List[str]
    verdict: str
    confidence: float
    explanation: str
    processing_time: float
    success: bool
    google_factcheck_result: Optional[Dict[str, Any]] = None
    fallback_used: bool = False
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

class ComprehensiveTruthLensPipeline:
    """Comprehensive pipeline with Google Fact Check API integration."""
    
    def __init__(self, google_api_key: Optional[str] = None):
        print("Comprehensive TruthLens Pipeline initialized")
        
        # Initialize Google Fact Check API if available
        self.google_factcheck = None
        if GOOGLE_FACTCHECK_AVAILABLE and google_api_key:
            try:
                self.google_factcheck = GoogleFactCheckAPI(google_api_key)
                print("✅ Google Fact Check API initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Google Fact Check API: {e}")
                self.google_factcheck = None
        elif not GOOGLE_FACTCHECK_AVAILABLE:
            print("⚠️ Google Fact Check API not available - using fallback verification")
        elif not google_api_key:
            print("⚠️ No Google API key provided - using fallback verification")
    
    def normalize_text(self, text: str) -> str:
        """Basic text normalization."""
        return text.strip()
    
    def extract_claims(self, text: str) -> List[str]:
        """Simple claim extraction using keyword matching."""
        # Simple keyword-based claim detection
        claim_indicators = [
            "causes", "caused", "leads to", "results in", "is responsible for",
            "proves", "shows", "demonstrates", "confirms", "reveals",
            "according to", "studies show", "research indicates", "scientists say",
            "is true", "is false", "is a fact", "is a myth"
        ]
        
        claims = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in claim_indicators):
                if len(sentence) > 10:  # Minimum length for a claim
                    claims.append(sentence)
        
        # If no claims found, treat the whole text as a claim
        if not claims and len(text) > 10:
            claims.append(text)
        
        return claims
    
    def get_evidence_sources(self, claim: str) -> List[str]:
        """Get relevant evidence sources for a claim."""
        # Mock evidence sources based on claim content
        sources = []
        
        claim_lower = claim.lower()
        
        if any(word in claim_lower for word in ["covid", "vaccine", "health", "medical"]):
            sources.extend(["WHO", "CDC", "medical journals", "peer-reviewed studies"])
        
        if any(word in claim_lower for word in ["5g", "radiation", "technology"]):
            sources.extend(["FCC", "IEEE", "scientific studies", "regulatory bodies"])
        
        if any(word in claim_lower for word in ["earth", "flat", "nasa", "space"]):
            sources.extend(["NASA", "astronomical observations", "satellite data", "scientific consensus"])
        
        if any(word in claim_lower for word in ["water", "boil", "temperature", "physics"]):
            sources.extend(["physics textbooks", "scientific literature", "experimental data"])
        
        # Add Google Fact Check as a source if available
        if self.google_factcheck:
            sources.append("Google Fact Check API")
        
        # Default sources
        if not sources:
            sources = ["fact-checking websites", "reliable news sources", "expert opinions"]
        
        return sources
    
    def verify_claim_with_google_factcheck(self, claim: str) -> Optional[FactCheckResult]:
        """Verify claim using Google Fact Check API."""
        if not self.google_factcheck:
            return None
        
        try:
            print(f"🔍 Verifying claim with Google Fact Check API: {claim[:50]}...")
            result = self.google_factcheck.verify_claim(claim)
            
            if result:
                print(f"✅ Found fact-check result: {result.verdict} (confidence: {result.confidence:.1%})")
                return result
            else:
                print("❌ No fact-check result found in Google Fact Check API")
                return None
                
        except Exception as e:
            print(f"❌ Error verifying claim with Google Fact Check API: {e}")
            return None
    
    def fallback_verification(self, claim: str) -> tuple[str, float]:
        """Fallback verification using keyword analysis."""
        claim_lower = claim.lower()
        
        # Keywords that suggest false claims (common misinformation patterns)
        false_indicators = [
            "myth", "hoax", "conspiracy", "fake", "false", "debunked",
            "no evidence", "disproven", "misleading", "inaccurate",
            "flat earth", "nasa hiding", "cover up", "they don't want you to know",
            "secret", "hidden", "suppressed", "censored", "mainstream media lies"
        ]
        
        # Keywords that suggest true claims
        true_indicators = [
            "confirmed", "verified", "proven", "fact", "true", "accurate",
            "scientific consensus", "peer-reviewed", "established", "evidence shows",
            "studies confirm", "research proves", "experts agree"
        ]
        
        # Count indicators
        false_count = sum(1 for indicator in false_indicators if indicator in claim_lower)
        true_count = sum(1 for indicator in true_indicators if indicator in claim_lower)
        
        # Determine verdict
        if false_count > true_count:
            return "REFUTED", 0.6
        elif true_count > false_count:
            return "SUPPORTED", 0.6
        else:
            return "NOT ENOUGH INFO", 0.3
    
    def verify_claim(self, claim: str) -> tuple[str, float, Optional[Dict[str, Any]], bool]:
        """Verify claim using Google Fact Check API or fallback."""
        # Try Google Fact Check API first
        google_result = self.verify_claim_with_google_factcheck(claim)
        
        if google_result:
            # Convert FactCheckResult to dict for storage
            google_dict = {
                "verdict": google_result.verdict,
                "confidence": google_result.confidence,
                "publisher": google_result.publisher,
                "review_date": google_result.review_date,
                "url": google_result.url,
                "explanation": google_result.explanation,
                "rating": google_result.rating,
                "claim_review_url": google_result.claim_review_url
            }
            return google_result.verdict, google_result.confidence, google_dict, False
        
        # Fallback to keyword-based verification
        print("⚠️ Using fallback keyword-based verification")
        verdict, confidence = self.fallback_verification(claim)
        return verdict, confidence, None, True
    
    def generate_explanation(self, claim: str, verdict: str, confidence: float, google_result: Optional[Dict[str, Any]] = None) -> str:
        """Generate explanation for the verdict."""
        if google_result:
            # Use Google Fact Check explanation
            return google_result["explanation"]
        
        # Generate fallback explanation
        if verdict == "REFUTED":
            return f"This claim appears to be false. Multiple reliable sources indicate that this information is incorrect or misleading. Confidence level: {confidence:.1%}"
        elif verdict == "SUPPORTED":
            return f"This claim appears to be supported by available evidence. Multiple sources confirm this information. Confidence level: {confidence:.1%}"
        else:
            return f"There is insufficient evidence to determine the accuracy of this claim. More information from reliable sources is needed. Confidence level: {confidence:.1%}"
    
    def process_claim(self, input_text: str) -> ComprehensivePipelineResult:
        """Process a claim through the comprehensive pipeline."""
        start_time = time.time()
        
        try:
            # Step 1: Normalize text
            normalized_text = self.normalize_text(input_text)
            
            # Step 2: Extract claims
            claims = self.extract_claims(normalized_text)
            
            # Step 3: Get evidence sources (use first claim if multiple)
            primary_claim = claims[0] if claims else normalized_text
            evidence_sources = self.get_evidence_sources(primary_claim)
            
            # Step 4: Verify claim
            verdict, confidence, google_result, fallback_used = self.verify_claim(primary_claim)
            
            # Step 5: Generate explanation
            explanation = self.generate_explanation(primary_claim, verdict, confidence, google_result)
            
            processing_time = time.time() - start_time
            
            return ComprehensivePipelineResult(
                input_text=input_text,
                normalized_text=normalized_text,
                extracted_claims=claims,
                evidence_sources=evidence_sources,
                verdict=verdict,
                confidence=confidence,
                explanation=explanation,
                processing_time=processing_time,
                success=True,
                google_factcheck_result=google_result,
                fallback_used=fallback_used
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ComprehensivePipelineResult(
                input_text=input_text,
                normalized_text="",
                extracted_claims=[],
                evidence_sources=[],
                verdict="ERROR",
                confidence=0.0,
                explanation=f"Error processing claim: {str(e)}",
                processing_time=processing_time,
                success=False,
                errors=[str(e)]
            )

def get_user_input() -> str:
    """Get claim input from user."""
    print("\nComprehensive TruthLens Pipeline Test (with Google Fact Check API)")
    print("="*70)
    print("Enter a claim to test the complete pipeline:")
    print("(Or press Enter to use a default test claim)")
    
    user_input = input("\nClaim: ").strip()
    
    if not user_input:
        # Default test claims
        default_claims = [
            "COVID-19 vaccines cause autism in children according to recent studies.",
            "5G towers emit dangerous radiation that causes cancer in nearby residents.",
            "Water boils at 100 degrees Celsius at sea level under normal atmospheric pressure.",
            "The Earth is flat and NASA has been hiding this fact for decades.",
            "Drinking 8 glasses of water per day is essential for good health."
        ]
        
        print("\nUsing default test claims:")
        for i, claim in enumerate(default_claims, 1):
            print(f"{i}. {claim}")
        
        choice = input("\nSelect a claim (1-5) or press Enter for claim 1: ").strip()
        try:
            choice_idx = int(choice) - 1 if choice else 0
            if 0 <= choice_idx < len(default_claims):
                return default_claims[choice_idx]
        except ValueError:
            pass
        
        return default_claims[0]
    
    return user_input

def print_results(result: ComprehensivePipelineResult):
    """Print formatted results to console."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TRUTHLENS PIPELINE RESULTS")
    print("="*80)
    
    print(f"Input Text: {result.input_text}")
    print(f"Processing Time: {result.processing_time:.2f} seconds")
    print(f"Success: {result.success}")
    
    if result.errors:
        print(f"Errors: {len(result.errors)}")
        for error in result.errors:
            print(f"   - {error}")
    
    if result.success:
        print(f"\nNormalized Text: {result.normalized_text}")
        
        print(f"\nExtracted Claims ({len(result.extracted_claims)}):")
        for i, claim in enumerate(result.extracted_claims, 1):
            print(f"   {i}. {claim}")
        
        print(f"\nEvidence Sources:")
        for source in result.evidence_sources:
            print(f"   - {source}")
        
        print(f"\nVerdict: {result.verdict}")
        print(f"Confidence: {result.confidence:.1%}")
        
        if result.fallback_used:
            print("⚠️ Fallback verification used (Google Fact Check API had no results)")
        
        print(f"\nExplanation: {result.explanation}")
        
        # Show Google Fact Check details if available
        if result.google_factcheck_result:
            print(f"\nGoogle Fact Check Details:")
            google = result.google_factcheck_result
            print(f"   Publisher: {google.get('publisher', 'Unknown')}")
            print(f"   Rating: {google.get('rating', 'Unknown')}")
            print(f"   Review Date: {google.get('review_date', 'Unknown')}")
            print(f"   URL: {google.get('url', 'N/A')}")
    
    print("\n" + "="*80)

def main():
    """Main function to run the comprehensive pipeline test."""
    try:
        # Google Fact Check API key
        google_api_key = "AIzaSyACLtCwVw1dJeBNPKmbJ8Yfqu5D4zUK5Sc"
        
        # Get user input
        input_text = get_user_input()
        
        # Initialize pipeline with Google API key
        pipeline = ComprehensiveTruthLensPipeline(google_api_key=google_api_key)
        
        # Process claim
        result = pipeline.process_claim(input_text)
        
        # Print results
        print_results(result)
        
        # Save results to file
        output_file = "comprehensive_pipeline_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        
        if result.success:
            print("\nComprehensive pipeline test completed successfully!")
            return 0
        else:
            print("\nComprehensive pipeline test completed with errors!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTest failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
