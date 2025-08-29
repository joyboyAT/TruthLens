#!/usr/bin/env python3
"""
News Integration Test for TruthLens
Tests the integration of News API with Google Fact Check API for news input processing.
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
    from src.news.news_handler import NewsHandler, NewsArticle
    from src.verification.google_factcheck_api import GoogleFactCheckAPI, FactCheckResult
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Integration not available - {e}")
    INTEGRATION_AVAILABLE = False

@dataclass
class NewsTestResult:
    """Result from the news integration test."""
    test_name: str
    input_text: str
    input_type: str
    news_context: Optional[Dict[str, Any]] = None
    extracted_claims: List[str] = None
    google_factcheck_results: List[Dict[str, Any]] = None
    pipeline_result: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    success: bool = False
    errors: List[str] = None

    def __post_init__(self):
        if self.extracted_claims is None:
            self.extracted_claims = []
        if self.google_factcheck_results is None:
            self.google_factcheck_results = []
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

class NewsIntegrationTester:
    """Comprehensive tester for the news integration workflow."""
    
    def __init__(self, google_api_key: str, news_api_key: str):
        """
        Initialize the news integration tester.
        
        Args:
            google_api_key: Google Fact Check API key
            news_api_key: News API key
        """
        self.google_api_key = google_api_key
        self.news_api_key = news_api_key
        self.google_factcheck = None
        self.news_handler = None
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
        
        # Initialize News API handler
        try:
            self.news_handler = NewsHandler(news_api_key)
            print("✅ News API initialized")
        except Exception as e:
            print(f"❌ Failed to initialize News API: {e}")
            raise
        
        # Initialize Enhanced Pipeline
        try:
            self.pipeline = EnhancedTruthLensPipeline(
                google_api_key=google_api_key,
                news_api_key=news_api_key
            )
            print("✅ Enhanced Pipeline initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Enhanced Pipeline: {e}")
            raise
    
    def test_news_search(self, query: str) -> List[NewsArticle]:
        """Test news search functionality."""
        try:
            print(f"📰 Testing news search for: {query}")
            articles = self.news_handler.search_news(query, max_results=5)
            print(f"✅ Found {len(articles)} news articles")
            return articles
        except Exception as e:
            print(f"❌ Error testing news search: {e}")
            return []
    
    def test_claim_extraction(self, news_articles: List[NewsArticle]) -> List[str]:
        """Test claim extraction from news articles."""
        try:
            print("🔍 Testing claim extraction from news articles...")
            claims = self.news_handler.extract_claims_from_news(news_articles)
            print(f"✅ Extracted {len(claims)} claims")
            return claims
        except Exception as e:
            print(f"❌ Error testing claim extraction: {e}")
            return []
    
    def test_google_factcheck_verification(self, claims: List[str]) -> List[Dict[str, Any]]:
        """Test Google Fact Check verification for extracted claims."""
        results = []
        
        for i, claim in enumerate(claims, 1):
            try:
                print(f"🔍 Verifying claim {i}/{len(claims)}: {claim[:50]}...")
                result = self.google_factcheck.verify_claim(claim)
                
                if result:
                    print(f"✅ Found fact-check result: {result.verdict}")
                    results.append({
                        "claim": claim,
                        "verdict": result.verdict,
                        "confidence": result.confidence,
                        "publisher": result.publisher,
                        "rating": result.rating,
                        "url": result.url
                    })
                else:
                    print("❌ No fact-check result found")
                    results.append({
                        "claim": claim,
                        "verdict": "NOT ENOUGH INFO",
                        "confidence": 0.0,
                        "publisher": "None",
                        "rating": "None",
                        "url": "None"
                    })
                
            except Exception as e:
                print(f"❌ Error verifying claim: {e}")
                results.append({
                    "claim": claim,
                    "error": str(e)
                })
            
            # Rate limiting
            if i < len(claims):
                time.sleep(0.5)
        
        return results
    
    def test_pipeline_integration(self, text: str) -> Optional[EnhancedPipelineResult]:
        """Test the complete pipeline integration with news input."""
        try:
            print(f"🔄 Testing pipeline integration: {text[:50]}...")
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
    
    def run_comprehensive_test(self, test_inputs: List[str]) -> List[NewsTestResult]:
        """Run comprehensive tests on multiple news inputs."""
        results = []
        
        print(f"\n🚀 Running comprehensive news integration tests on {len(test_inputs)} inputs...")
        print("="*80)
        
        for i, input_text in enumerate(test_inputs, 1):
            print(f"\n--- Test {i}/{len(test_inputs)} ---")
            print(f"Input: {input_text}")
            
            start_time = time.time()
            
            try:
                # Test 1: News search
                news_articles = self.test_news_search(input_text)
                
                # Test 2: Claim extraction
                extracted_claims = self.test_claim_extraction(news_articles)
                
                # Test 3: Google Fact Check verification
                google_results = self.test_google_factcheck_verification(extracted_claims)
                
                # Test 4: Pipeline integration
                pipeline_result = self.test_pipeline_integration(input_text)
                
                processing_time = time.time() - start_time
                
                # Create test result
                test_result = NewsTestResult(
                    test_name=f"Test {i}",
                    input_text=input_text,
                    input_type=pipeline_result.input_type if pipeline_result else "unknown",
                    extracted_claims=extracted_claims,
                    google_factcheck_results=google_results,
                    processing_time=processing_time,
                    success=True
                )
                
                # Add news context
                if pipeline_result and pipeline_result.news_context:
                    test_result.news_context = {
                        "articles_found": pipeline_result.news_context.get("articles_found", 0),
                        "context": pipeline_result.news_context.get("context", ""),
                        "articles": pipeline_result.news_context.get("articles", [])
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
                print(f"📰 News articles found: {len(news_articles)}")
                print(f"🔍 Claims extracted: {len(extracted_claims)}")
                print(f"✅ Google Fact Check results: {len([r for r in google_results if r.get('verdict') != 'NOT ENOUGH INFO'])}")
                if pipeline_result:
                    print(f"🔄 Pipeline verdict: {pipeline_result.overall_verdict['verdict']}")
                
            except Exception as e:
                processing_time = time.time() - start_time
                test_result = NewsTestResult(
                    test_name=f"Test {i}",
                    input_text=input_text,
                    input_type="unknown",
                    processing_time=processing_time,
                    success=False,
                    errors=[str(e)]
                )
                results.append(test_result)
                print(f"❌ Test failed: {e}")
            
            # Rate limiting
            if i < len(test_inputs):
                time.sleep(1)
        
        return results
    
    def print_test_summary(self, results: List[NewsTestResult]):
        """Print comprehensive test summary."""
        print("\n" + "="*80)
        print("NEWS INTEGRATION TEST SUMMARY")
        print("="*80)
        
        successful_tests = sum(1 for r in results if r.success)
        print(f"Successful Tests: {successful_tests}/{len(results)}")
        
        if successful_tests > 0:
            # Input type distribution
            input_types = {}
            for result in results:
                input_type = result.input_type
                input_types[input_type] = input_types.get(input_type, 0) + 1
            
            print(f"\nInput Type Distribution:")
            for input_type, count in input_types.items():
                print(f"  {input_type}: {count}")
            
            # News context summary
            news_tests = [r for r in results if r.news_context and r.news_context.get("articles_found", 0) > 0]
            print(f"\nNews Context Found: {len(news_tests)}/{len(results)}")
            
            if news_tests:
                total_articles = sum(r.news_context.get("articles_found", 0) for r in news_tests)
                avg_articles = total_articles / len(news_tests)
                print(f"Average articles per test: {avg_articles:.1f}")
            
            # Claim extraction summary
            total_claims = sum(len(r.extracted_claims) for r in results)
            avg_claims = total_claims / len(results)
            print(f"\nClaim Extraction:")
            print(f"  Total claims extracted: {total_claims}")
            print(f"  Average claims per test: {avg_claims:.1f}")
            
            # Google Fact Check summary
            google_results = []
            for result in results:
                google_results.extend(result.google_factcheck_results)
            
            if google_results:
                factcheck_found = sum(1 for r in google_results if r.get('verdict') != 'NOT ENOUGH INFO')
                print(f"\nGoogle Fact Check Results:")
                print(f"  Claims with fact-check results: {factcheck_found}/{len(google_results)}")
                
                verdict_counts = {}
                for result in google_results:
                    verdict = result.get('verdict', 'UNKNOWN')
                    verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
                
                print("  Verdict Distribution:")
                for verdict, count in verdict_counts.items():
                    print(f"    {verdict}: {count}")
            
            # Pipeline results
            pipeline_results = [r for r in results if r.pipeline_result]
            if pipeline_results:
                pipeline_verdicts = {}
                google_used_count = 0
                for result in pipeline_results:
                    verdict = result.pipeline_result["verdict"]
                    pipeline_verdicts[verdict] = pipeline_verdicts.get(verdict, 0) + 1
                    
                    if result.pipeline_result.get("google_factcheck_used", False):
                        google_used_count += 1
                
                print(f"\nPipeline Results:")
                print(f"  Google Fact Check API used: {google_used_count}/{len(pipeline_results)}")
                print("  Verdict Distribution:")
                for verdict, count in pipeline_verdicts.items():
                    print(f"    {verdict}: {count}")
            
            # Performance metrics
            avg_time = sum(r.processing_time for r in results) / len(results)
            print(f"\nPerformance:")
            print(f"  Average Processing Time: {avg_time:.2f}s per test")
        
        # Error summary
        failed_tests = [r for r in results if not r.success]
        if failed_tests:
            print(f"\nFailed Tests: {len(failed_tests)}")
            for test in failed_tests:
                print(f"  - {test.test_name}: {test.errors}")

def get_test_inputs() -> List[str]:
    """Get a comprehensive list of test inputs including news queries."""
    return [
        # News queries (based on the provided News API results)
        "nanded floods",
        "Maharashtra heavy rains",
        "Mumbai rains disruption",
        "Nanded cloudburst",
        "Maharashtra army rescue",
        
        # General news topics
        "COVID-19 vaccines",
        "climate change",
        "artificial intelligence",
        "space exploration",
        "renewable energy",
        
        # Specific claims
        "COVID-19 vaccines cause autism",
        "5G causes coronavirus",
        "The Earth is flat",
        "Climate change is a hoax"
    ]

def main():
    """Main function to run the news integration test."""
    try:
        # API keys
        google_api_key = "AIzaSyACLtCwVw1dJeBNPKmbJ8Yfqu5D4zUK5Sc"
        news_api_key = "9c7e59e19af34bb8adb97d0a8bec458d"
        
        if not INTEGRATION_AVAILABLE:
            print("❌ Integration components not available")
            return 1
        
        # Initialize tester
        print("🚀 Initializing News Integration Tester...")
        tester = NewsIntegrationTester(google_api_key, news_api_key)
        
        # Get test inputs
        test_inputs = get_test_inputs()
        
        # Run comprehensive tests
        results = tester.run_comprehensive_test(test_inputs)
        
        # Print summary
        tester.print_test_summary(results)
        
        # Save detailed results
        output_file = "news_integration_results.json"
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
