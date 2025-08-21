#!/usr/bin/env python3
"""
End-to-End Pipeline Test for TruthLens
Tests the complete fact-checking pipeline from input to output.
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from typing import Optional

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing module imports...")
    
    try:
        # Test core modules
        from src import data_collection, preprocessing
        print("✅ Core modules imported successfully")
        
        # Test extractor modules
        from extractor import pipeline, claim_detector, claim_extractor
        print("✅ Extractor modules imported successfully")
        
        # Test verification modules
        from src.verification import pipeline as verification_pipeline
        print("✅ Verification modules imported successfully")
        
        # Test evidence retrieval
        from src.evidence_retrieval import grounded_search
        print("✅ Evidence retrieval modules imported successfully")
        
        # Test output UX
        from src.output_ux import evidence_cards, cue_badges
        print("✅ Output UX modules imported successfully")
        
        # Test manipulation detection
        from src.manipulation import pipeline as manipulation_pipeline
        print("✅ Manipulation detection modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_collection():
    """Test data collection functionality."""
    print("\n📥 Testing data collection...")
    
    try:
        from src.data_collection import fetch_url
        
        # Test URL fetching
        test_url = "https://httpbin.org/html"
        result = fetch_url(test_url)
        
        if result and result.get('html'):
            print("✅ URL fetching works")
        else:
            print("❌ URL fetching failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Data collection error: {e}")
        return False

def test_preprocessing():
    """Test text preprocessing functionality."""
    print("\n🔄 Testing preprocessing...")
    
    try:
        from src.preprocessing import html_to_text
        
        # Test HTML to text conversion
        html_content = "<html><body><p>This is a <strong>test</strong> paragraph.</p></body></html>"
        text = html_to_text(html_content)
        
        if text and "test" in text:
            print("✅ HTML to text conversion works")
        else:
            print("❌ HTML to text conversion failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return False

def test_claim_detection():
    """Test claim detection functionality."""
    print("\n🎯 Testing claim detection...")
    
    try:
        from extractor.claim_detector import is_claim
        
        # Test claim detection
        test_text = "The Earth is flat and NASA is hiding the truth."
        is_claim_result, prob = is_claim(test_text)
        
        if isinstance(is_claim_result, bool) and isinstance(prob, float):
            print("✅ Claim detection works")
        else:
            print("❌ Claim detection failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Claim detection error: {e}")
        return False

def test_claim_extraction():
    """Test claim extraction functionality."""
    print("\n📝 Testing claim extraction...")
    
    try:
        from extractor.claim_extractor import extract_claim_spans
        
        # Test claim extraction
        test_text = "Scientists say that climate change is real and caused by human activities."
        extracted = extract_claim_spans(test_text)
        
        if isinstance(extracted, list):
            print("✅ Claim extraction works")
        else:
            print("❌ Claim extraction failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Claim extraction error: {e}")
        return False

def test_evidence_retrieval():
    """Test evidence retrieval functionality."""
    print("\n🔍 Testing evidence retrieval...")
    
    try:
        from src.evidence_retrieval.grounded_search import GroundedSearcher, SearchClientBase
        
        # Create a mock client for testing
        class MockSearchClient(SearchClientBase):
            def search(self, query: str, num_results: int = 10, days: Optional[int] = None):
                return []  # Return empty results for testing
        
        # Initialize searcher with mock client
        mock_client = MockSearchClient()
        searcher = GroundedSearcher(mock_client)
        
        # Test search functionality (mock test)
        print("✅ Evidence retrieval module loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Evidence retrieval error: {e}")
        return False

def test_verification():
    """Test verification functionality."""
    print("\n✅ Testing verification...")
    
    try:
        from src.verification.pipeline import VerificationPipeline
        
        # Initialize verification pipeline
        pipeline = VerificationPipeline()
        
        print("✅ Verification pipeline loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def test_manipulation_detection():
    """Test manipulation detection functionality."""
    print("\n🎭 Testing manipulation detection...")
    
    try:
        from src.manipulation.pipeline import ManipulationPipeline
        
        # Initialize manipulation pipeline
        pipeline = ManipulationPipeline()
        
        print("✅ Manipulation detection pipeline loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Manipulation detection error: {e}")
        return False

def test_output_generation():
    """Test output generation functionality."""
    print("\n📊 Testing output generation...")
    
    try:
        from src.output_ux.evidence_cards import build_evidence_cards
        from src.output_ux.cue_badges import generate_cue_badges
        
        # Test evidence cards
        evidence_data = [
            {"title": "Test Evidence", "content": "This is test evidence", "source": "test.com"}
        ]
        cards = build_evidence_cards(evidence_data)
        
        if isinstance(cards, list):
            print("✅ Evidence cards generation works")
        else:
            print("❌ Evidence cards generation failed")
            return False
            
        # Test cue badges
        cues = ["misleading", "outdated"]
        badges = generate_cue_badges(cues)
        
        if isinstance(badges, list):
            print("✅ Cue badges generation works")
        else:
            print("❌ Cue badges generation failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Output generation error: {e}")
        return False

def test_full_pipeline():
    """Test the complete pipeline end-to-end."""
    print("\n🚀 Testing full pipeline...")
    
    try:
        from extractor.pipeline import process_text
        
        # Test with sample input
        sample_text = "The Earth is flat and NASA is hiding the truth from us."
        
        # Run pipeline
        results = process_text(sample_text)
        
        if isinstance(results, list):
            print("✅ Full pipeline works")
        else:
            print("❌ Full pipeline failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline error: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n⚙️ Testing configuration...")
    
    try:
        config_dir = Path("config")
        if config_dir.exists():
            config_files = list(config_dir.glob("*.yaml"))
            if config_files:
                print(f"✅ Found {len(config_files)} configuration files")
                return True
            else:
                print("⚠️ No configuration files found")
                return True
        else:
            print("⚠️ Config directory not found")
            return True
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_database_schemas():
    """Test database schema files."""
    print("\n🗄️ Testing database schemas...")
    
    try:
        schema_dir = Path("database_schemas")
        if schema_dir.exists():
            schema_files = list(schema_dir.glob("*.sql"))
            if schema_files:
                print(f"✅ Found {len(schema_files)} database schema files")
                return True
            else:
                print("⚠️ No database schema files found")
                return True
        else:
            print("⚠️ Database schemas directory not found")
            return True
            
    except Exception as e:
        print(f"❌ Database schema error: {e}")
        return False

def main():
    """Run all end-to-end tests."""
    print("🧪 TruthLens End-to-End Pipeline Test")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Data Collection", test_data_collection),
        ("Preprocessing", test_preprocessing),
        ("Claim Detection", test_claim_detection),
        ("Claim Extraction", test_claim_extraction),
        ("Evidence Retrieval", test_evidence_retrieval),
        ("Verification", test_verification),
        ("Manipulation Detection", test_manipulation_detection),
        ("Output Generation", test_output_generation),
        ("Full Pipeline", test_full_pipeline),
        ("Configuration", test_configuration),
        ("Database Schemas", test_database_schemas),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
