#!/usr/bin/env python3
"""
Complete TruthLens Workflow Test
Final comprehensive test demonstrating the complete TruthLens pipeline with user input.
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

# Import pipeline components
try:
    from src.enhanced_pipeline import EnhancedTruthLensPipeline, EnhancedPipelineResult
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Integration not available - {e}")
    INTEGRATION_AVAILABLE = False

@dataclass
class CompleteWorkflowResult:
    """Result from the complete workflow test."""
    input_text: str
    input_type: str
    news_context: Optional[Dict[str, Any]] = None
    extracted_claims: List[str] = None
    verification_results: List[Dict[str, Any]] = None
    overall_verdict: Dict[str, Any] = None
    processing_time: float = 0.0
    sources_checked: List[str] = None
    google_factcheck_used: bool = False
    success: bool = False
    errors: List[str] = None

    def __post_init__(self):
        if self.extracted_claims is None:
            self.extracted_claims = []
        if self.verification_results is None:
            self.verification_results = []
        if self.sources_checked is None:
            self.sources_checked = []
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

class CompleteWorkflowTester:
    """Complete workflow tester for TruthLens."""
    
    def __init__(self, google_api_key: str, news_api_key: str):
        """
        Initialize the complete workflow tester.
        
        Args:
            google_api_key: Google Fact Check API key
            news_api_key: News API key
        """
        self.google_api_key = google_api_key
        self.news_api_key = news_api_key
        self.pipeline = None
        
        if not INTEGRATION_AVAILABLE:
            raise ImportError("Required components not available")
        
        # Initialize Enhanced Pipeline
        try:
            self.pipeline = EnhancedTruthLensPipeline(
                google_api_key=google_api_key,
                news_api_key=news_api_key
            )
            print("✅ Complete TruthLens Pipeline initialized")
        except Exception as e:
            print(f"❌ Failed to initialize pipeline: {e}")
            raise
    
    def process_user_input(self, user_input: str) -> CompleteWorkflowResult:
        """Process user input through the complete pipeline."""
        start_time = time.time()
        
        try:
            print(f"\n🔍 Processing user input: {user_input}")
            print("="*60)
            
            # Process through pipeline
            result = self.pipeline.process_text(user_input)
            
            processing_time = time.time() - start_time
            
            # Create workflow result
            workflow_result = CompleteWorkflowResult(
                input_text=user_input,
                input_type=result.input_type,
                news_context=result.news_context,
                extracted_claims=result.extracted_claims,
                verification_results=result.verification_results,
                overall_verdict=result.overall_verdict,
                processing_time=processing_time,
                sources_checked=result.sources_checked,
                google_factcheck_used=result.google_factcheck_used,
                success=True
            )
            
            return workflow_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            return CompleteWorkflowResult(
                input_text=user_input,
                input_type="unknown",
                processing_time=processing_time,
                success=False,
                errors=[str(e)]
            )
    
    def print_workflow_result(self, result: CompleteWorkflowResult):
        """Print formatted workflow result."""
        print("\n" + "="*80)
        print("TRUTHLENS WORKFLOW RESULTS")
        print("="*80)
        
        print(f"Input Text: {result.input_text}")
        print(f"Input Type: {result.input_type}")
        print(f"Processing Time: {result.processing_time:.2f} seconds")
        print(f"Success: {result.success}")
        
        if result.errors:
            print(f"Errors: {len(result.errors)}")
            for error in result.errors:
                print(f"   - {error}")
        
        if result.success:
            # News Context
            if result.news_context:
                print(f"\n📰 News Context:")
                print(f"   Articles Found: {result.news_context.get('articles_found', 0)}")
                print(f"   Context: {result.news_context.get('context', '')[:200]}...")
            
            # Extracted Claims
            print(f"\n🔍 Extracted Claims ({len(result.extracted_claims)}):")
            for i, claim in enumerate(result.extracted_claims, 1):
                print(f"   {i}. {claim}")
            
            # Sources Checked
            print(f"\n📚 Sources Checked:")
            for source in result.sources_checked:
                print(f"   - {source}")
            
            # Overall Verdict
            print(f"\n⚖️ Overall Verdict:")
            print(f"   Verdict: {result.overall_verdict['verdict']}")
            print(f"   Confidence: {result.overall_verdict['confidence']:.1%}")
            print(f"   Reasoning: {result.overall_verdict['reasoning']}")
            print(f"   Source: {result.overall_verdict.get('source', 'Unknown')}")
            
            if result.google_factcheck_used:
                print("   ✅ Google Fact Check API was used")
            else:
                print("   ⚠️ Google Fact Check API had no results, used fallback")
            
            # Verification Results
            if result.verification_results:
                print(f"\n🔍 Verification Results ({len(result.verification_results)}):")
                for i, verification in enumerate(result.verification_results, 1):
                    print(f"   {i}. {verification['claim_text'][:50]}...")
                    print(f"      Stance: {verification['stance']}")
                    print(f"      Confidence: {verification['confidence_score']:.1%}")
                    print(f"      Source: {verification['source']}")
                    
                    if verification.get('google_factcheck_result'):
                        google = verification['google_factcheck_result']
                        print(f"      Google Fact Check: {google.get('publisher', 'Unknown')} - {google.get('rating', 'Unknown')}")
        
        print("\n" + "="*80)

def get_user_input() -> str:
    """Get input from user."""
    print("\n🚀 TruthLens Complete Workflow Test")
    print("="*60)
    print("Enter text to analyze (news, claims, or general text):")
    print("(Or press Enter to use example inputs)")
    
    user_input = input("\nInput: ").strip()
    
    if not user_input:
        # Example inputs
        examples = [
            "nanded floods",
            "COVID-19 vaccines cause autism",
            "5G causes coronavirus",
            "The Earth is flat",
            "Maharashtra heavy rains",
            "Climate change is a hoax"
        ]
        
        print("\nExample inputs:")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
        
        choice = input("\nSelect an example (1-6) or press Enter for example 1: ").strip()
        try:
            choice_idx = int(choice) - 1 if choice else 0
            if 0 <= choice_idx < len(examples):
                return examples[choice_idx]
        except ValueError:
            pass
        
        return examples[0]
    
    return user_input

def main():
    """Main function to run the complete workflow test."""
    try:
        # API keys
        google_api_key = "AIzaSyACLtCwVw1dJeBNPKmbJ8Yfqu5D4zUK5Sc"
        news_api_key = "9c7e59e19af34bb8adb97d0a8bec458d"
        
        if not INTEGRATION_AVAILABLE:
            print("❌ Integration components not available")
            return 1
        
        # Initialize tester
        print("🚀 Initializing Complete TruthLens Workflow Tester...")
        tester = CompleteWorkflowTester(google_api_key, news_api_key)
        
        # Get user input
        user_input = get_user_input()
        
        # Process input
        result = tester.process_user_input(user_input)
        
        # Print results
        tester.print_workflow_result(result)
        
        # Save results
        output_file = "complete_workflow_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        if result.success:
            print("\n✅ Complete workflow test completed successfully!")
            return 0
        else:
            print("\n❌ Complete workflow test failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
