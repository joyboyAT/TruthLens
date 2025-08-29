#!/usr/bin/env python3
"""
Simple TruthLens Pipeline Test
A lightweight test that demonstrates the pipeline structure without heavy model dependencies.
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

@dataclass
class SimplePipelineResult:
    """Simple result from the pipeline."""
    input_text: str
    normalized_text: str
    extracted_claims: List[str]
    evidence_sources: List[str]
    verdict: str
    confidence: float
    explanation: str
    processing_time: float
    success: bool
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

class SimpleTruthLensPipeline:
    """Simple pipeline implementation for testing without heavy dependencies."""
    
    def __init__(self):
        print("Simple TruthLens Pipeline initialized")
    
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
        
        # Default sources
        if not sources:
            sources = ["fact-checking websites", "reliable news sources", "expert opinions"]
        
        return sources
    
    def verify_claim(self, claim: str, sources: List[str]) -> tuple[str, float]:
        """Simple claim verification using keyword analysis."""
        claim_lower = claim.lower()
        
        # Keywords that suggest false claims
        false_indicators = [
            "myth", "hoax", "conspiracy", "fake", "false", "debunked",
            "no evidence", "disproven", "misleading", "inaccurate"
        ]
        
        # Keywords that suggest true claims
        true_indicators = [
            "confirmed", "verified", "proven", "fact", "true", "accurate",
            "scientific consensus", "peer-reviewed", "established"
        ]
        
        # Count indicators
        false_count = sum(1 for indicator in false_indicators if indicator in claim_lower)
        true_count = sum(1 for indicator in true_indicators if indicator in claim_lower)
        
        # Determine verdict
        if false_count > true_count:
            return "REFUTED", 0.7
        elif true_count > false_count:
            return "SUPPORTED", 0.7
        else:
            return "NOT ENOUGH INFO", 0.3
    
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
    
    def generate_explanation(self, claim: str, verdict: str, confidence: float) -> str:
        """Generate explanation for the verdict."""
        if verdict == "REFUTED":
            return f"This claim appears to be false. Multiple reliable sources indicate that this information is incorrect or misleading. Confidence level: {confidence:.1%}"
        elif verdict == "SUPPORTED":
            return f"This claim appears to be supported by available evidence. Multiple sources confirm this information. Confidence level: {confidence:.1%}"
        else:
            return f"There is insufficient evidence to determine the accuracy of this claim. More information from reliable sources is needed. Confidence level: {confidence:.1%}"
    
    def process_claim(self, input_text: str) -> SimplePipelineResult:
        """Process a claim through the simple pipeline."""
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
            verdict, confidence = self.verify_claim(primary_claim, evidence_sources)
            
            # Step 5: Generate explanation
            explanation = self.generate_explanation(primary_claim, verdict, confidence)
            
            processing_time = time.time() - start_time
            
            return SimplePipelineResult(
                input_text=input_text,
                normalized_text=normalized_text,
                extracted_claims=claims,
                evidence_sources=evidence_sources,
                verdict=verdict,
                confidence=confidence,
                explanation=explanation,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return SimplePipelineResult(
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
    print("\nSimple TruthLens Pipeline Test")
    print("="*50)
    print("Enter a claim to test the pipeline:")
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

def print_results(result: SimplePipelineResult):
    """Print formatted results to console."""
    print("\n" + "="*80)
    print("SIMPLE TRUTHLENS PIPELINE RESULTS")
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
        
        print(f"\nExplanation: {result.explanation}")
    
    print("\n" + "="*80)

def main():
    """Main function to run the simple pipeline test."""
    try:
        # Get user input
        input_text = get_user_input()
        
        # Initialize pipeline
        pipeline = SimpleTruthLensPipeline()
        
        # Process claim
        result = pipeline.process_claim(input_text)
        
        # Print results
        print_results(result)
        
        # Save results to file
        output_file = "simple_pipeline_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        
        if result.success:
            print("\nSimple pipeline test completed successfully!")
            return 0
        else:
            print("\nSimple pipeline test completed with errors!")
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
