#!/usr/bin/env python3
"""
Integrated TruthLens Workflow Test
Complete end-to-end test demonstrating Google Fact Check API integration with the pipeline.
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
    from src.verification.google_factcheck_api import GoogleFactCheckAPI, FactCheckResult
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Integration not available - {e}")
    INTEGRATION_AVAILABLE = False

@dataclass
class WorkflowTestResult:
    """Result from the integrated workflow test."""
    test_name: str
    input_text: str
    google_factcheck_result: Optional[Dict[str, Any]] = None
    pipeline_result: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    success: bool = False
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

class IntegratedWorkflowTester:
    """Comprehensive tester for the integrated TruthLens workflow."""
    
    def __init__(self, google_api_key: str):
        """
        Initialize the integrated workflow tester.
        
        Args:
            google_api_key: Google Fact Check API key
        """
        self.google_api_key = google_api_key
        self.google_factcheck = None
        self.pipeline = None
        
        if not INTEGRATION_AVAILABLE:
            raise ImportError("Required components not available")
        
        # Initialize Google Fact Check API
        try:
            self.google_factcheck = GoogleFactCheckAPI(google_api_key)
            print("✅ Google Fact Check API initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Google Fact Check API: {e}")
            raise
        
        # Initialize Enhanced Pipeline
        try:
            self.pipeline = EnhancedTruthLensPipeline(google_api_key=google_api_key)
            print("✅ Enhanced Pipeline initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Enhanced Pipeline: {e}")
            raise
    
    def test_google_factcheck_direct(self, claim: str) -> Optional[FactCheckResult]:
        """Test Google Fact Check API directly."""
        try:
            print(f"🔍 Testing Google Fact Check API directly: {claim[:50]}...")
            result = self.google_factcheck.verify_claim(claim)
            
            if result:
                print(f"✅ Found result: {result.verdict} (confidence: {result.confidence:.1%})")
                return result
            else:
                print("❌ No result found")
                return None
                
        except Exception as e:
            print(f"❌ Error testing Google Fact Check API: {e}")
            return None
    
    def test_pipeline_integration(self, text: str) -> Optional[EnhancedPipelineResult]:
        """Test the complete pipeline integration."""
        try:
            print(f"🔍 Testing pipeline integration: {text[:50]}...")
            result = self.pipeline.process_text(text)
            
            if result:
                print(f"✅ Pipeline result: {result.overall_verdict['verdict']} (confidence: {result.overall_verdict['confidence']:.2f})")
                return result
            else:
                print("❌ No pipeline result")
                return None
                
        except Exception as e:
            print(f"❌ Error testing pipeline: {e}")
            return None
    
    def run_comprehensive_test(self, test_claims: List[str]) -> List[WorkflowTestResult]:
        """Run comprehensive tests on multiple claims."""
        results = []
        
        print(f"\n🚀 Running comprehensive tests on {len(test_claims)} claims...")
        print("="*80)
        
        for i, claim in enumerate(test_claims, 1):
            print(f"\n--- Test {i}/{len(test_claims)} ---")
            print(f"Claim: {claim}")
            
            start_time = time.time()
            
            try:
                # Test 1: Google Fact Check API directly
                google_result = self.test_google_factcheck_direct(claim)
                
                # Test 2: Pipeline integration
                pipeline_result = self.test_pipeline_integration(claim)
                
                processing_time = time.time() - start_time
                
                # Create test result
                test_result = WorkflowTestResult(
                    test_name=f"Test {i}",
                    input_text=claim,
                    processing_time=processing_time,
                    success=True
                )
                
                # Add Google Fact Check result
                if google_result:
                    test_result.google_factcheck_result = {
                        "verdict": google_result.verdict,
                        "confidence": google_result.confidence,
                        "publisher": google_result.publisher,
                        "rating": google_result.rating,
                        "url": google_result.url,
                        "explanation": google_result.explanation
                    }
                
                # Add pipeline result
                if pipeline_result:
                    test_result.pipeline_result = {
                        "verdict": pipeline_result.overall_verdict["verdict"],
                        "confidence": pipeline_result.overall_verdict["confidence"],
                        "reasoning": pipeline_result.overall_verdict["reasoning"],
                        "source": pipeline_result.overall_verdict.get("source", "Unknown"),
                        "google_factcheck_used": pipeline_result.google_factcheck_used
                    }
                
                results.append(test_result)
                
                # Print summary
                print(f"⏱️ Processing time: {processing_time:.2f}s")
                if google_result:
                    print(f"🔍 Google Fact Check: {google_result.verdict} ({google_result.confidence:.1%})")
                if pipeline_result:
                    print(f"🔄 Pipeline: {pipeline_result.overall_verdict['verdict']} ({pipeline_result.overall_verdict['confidence']:.1%})")
                
            except Exception as e:
                processing_time = time.time() - start_time
                test_result = WorkflowTestResult(
                    test_name=f"Test {i}",
                    input_text=claim,
                    processing_time=processing_time,
                    success=False,
                    errors=[str(e)]
                )
                results.append(test_result)
                print(f"❌ Test failed: {e}")
            
            # Rate limiting
            if i < len(test_claims):
                time.sleep(1)
        
        return results
    
    def print_test_summary(self, results: List[WorkflowTestResult]):
        """Print comprehensive test summary."""
        print("\n" + "="*80)
        print("INTEGRATED WORKFLOW TEST SUMMARY")
        print("="*80)
        
        successful_tests = sum(1 for r in results if r.success)
        print(f"Successful Tests: {successful_tests}/{len(results)}")
        
        if successful_tests > 0:
            # Google Fact Check results
            google_results = [r for r in results if r.google_factcheck_result]
            print(f"\nGoogle Fact Check API Results: {len(google_results)}/{len(results)}")
            
            if google_results:
                google_verdicts = {}
                for result in google_results:
                    verdict = result.google_factcheck_result["verdict"]
                    google_verdicts[verdict] = google_verdicts.get(verdict, 0) + 1
                
                print("Google Fact Check Verdict Distribution:")
                for verdict, count in google_verdicts.items():
                    print(f"  {verdict}: {count}")
            
            # Pipeline results
            pipeline_results = [r for r in results if r.pipeline_result]
            print(f"\nPipeline Integration Results: {len(pipeline_results)}/{len(results)}")
            
            if pipeline_results:
                pipeline_verdicts = {}
                google_used_count = 0
                for result in pipeline_results:
                    verdict = result.pipeline_result["verdict"]
                    pipeline_verdicts[verdict] = pipeline_verdicts.get(verdict, 0) + 1
                    
                    if result.pipeline_result.get("google_factcheck_used", False):
                        google_used_count += 1
                
                print("Pipeline Verdict Distribution:")
                for verdict, count in pipeline_verdicts.items():
                    print(f"  {verdict}: {count}")
                
                print(f"\nGoogle Fact Check API used in pipeline: {google_used_count}/{len(pipeline_results)}")
            
            # Performance metrics
            avg_time = sum(r.processing_time for r in results) / len(results)
            print(f"\nAverage Processing Time: {avg_time:.2f}s per claim")
        
        # Error summary
        failed_tests = [r for r in results if not r.success]
        if failed_tests:
            print(f"\nFailed Tests: {len(failed_tests)}")
            for test in failed_tests:
                print(f"  - {test.test_name}: {test.errors}")

def get_test_claims() -> List[str]:
    """Get a comprehensive list of test claims."""
    return [
        # Claims likely to have Google Fact Check results
        "COVID-19 vaccines cause autism",
        "5G causes coronavirus",
        "Bill Gates wants to microchip people",
        "Hydroxychloroquine cures COVID-19",
        "The Earth is flat",
        "Climate change is a hoax",
        "The moon landing was fake",
        "Chemtrails are real",
        
        # Claims that might not have Google Fact Check results
        "Water boils at 100 degrees Celsius at sea level",
        "The sky is blue due to Rayleigh scattering",
        "Modi visited Japan in 2024",
        "Python is a programming language",
        "The Great Wall of China is visible from space",
        "Drinking 8 glasses of water per day is essential"
    ]

def main():
    """Main function to run the integrated workflow test."""
    try:
        # Google Fact Check API key
        google_api_key = "AIzaSyACLtCwVw1dJeBNPKmbJ8Yfqu5D4zUK5Sc"
        
        if not INTEGRATION_AVAILABLE:
            print("❌ Integration components not available")
            return 1
        
        # Initialize tester
        print("🚀 Initializing Integrated Workflow Tester...")
        tester = IntegratedWorkflowTester(google_api_key)
        
        # Get test claims
        test_claims = get_test_claims()
        
        # Run comprehensive tests
        results = tester.run_comprehensive_test(test_claims)
        
        # Print summary
        tester.print_test_summary(results)
        
        # Save detailed results
        output_file = "integrated_workflow_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        successful_tests = sum(1 for r in results if r.success)
        if successful_tests == len(results):
            print("\n✅ All tests completed successfully!")
            return 0
        else:
            print(f"\n⚠️ {len(results) - successful_tests} tests failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
