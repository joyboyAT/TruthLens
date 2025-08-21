#!/usr/bin/env python3
"""
Test script for Vector-based Evidence Retrieval
Tests the FAISS-based semantic search functionality.
"""

import sys
import os
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_vector_retriever_initialization():
    """Test vector retriever initialization."""
    print("🔧 Testing vector retriever initialization...")
    
    try:
        from src.evidence_retrieval.vector_search import VectorEvidenceRetriever
        
        # Create a temporary directory for test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Initialize retriever
            retriever = VectorEvidenceRetriever(
                model_name="all-MiniLM-L6-v2",
                index_path=temp_path / "test_index.bin",
                embeddings_path=temp_path / "test_embeddings.pkl",
                use_gpu=False
            )
            
            # Check initialization
            assert retriever.model is not None, "Model should be loaded"
            assert retriever.index is not None, "FAISS index should be created"
            assert len(retriever.evidence_list) == 0, "Evidence list should be empty initially"
            
            print("✅ Vector retriever initialized successfully")
            return True
            
    except Exception as e:
        print(f"❌ Vector retriever initialization failed: {e}")
        return False

def test_evidence_addition():
    """Test adding evidence to the vector index."""
    print("\n📝 Testing evidence addition...")
    
    try:
        from src.evidence_retrieval.vector_search import VectorEvidenceRetriever
        from src.schemas.evidence import Evidence, SourceType
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Initialize retriever
            retriever = VectorEvidenceRetriever(
                index_path=temp_path / "test_index.bin",
                embeddings_path=temp_path / "test_embeddings.pkl",
                use_gpu=False
            )
            
            # Create sample evidence
            evidence_list = [
                Evidence(
                    id="test_1",
                    claim_id="claim_1",
                    source_type=SourceType.NEWS_ARTICLE,
                    url="https://example.com/article1",
                    domain="example.com",
                    title="Climate Change Study Shows Global Warming",
                    snippet="A new study published in Nature shows that global temperatures have increased by 1.1°C since pre-industrial times.",
                    full_text="A comprehensive study published in Nature journal reveals that global temperatures have increased by 1.1°C since pre-industrial times, with significant impacts on weather patterns and ecosystems worldwide."
                ),
                Evidence(
                    id="test_2",
                    claim_id="claim_1",
                    source_type=SourceType.RESEARCH_PAPER,
                    url="https://example.com/paper2",
                    domain="example.com",
                    title="Vaccine Safety Research Findings",
                    snippet="Large-scale study finds no link between vaccines and autism in children.",
                    full_text="A comprehensive large-scale study involving over 100,000 children found no statistical link between vaccination and autism spectrum disorders."
                ),
                Evidence(
                    id="test_3",
                    claim_id="claim_2",
                    source_type=SourceType.GOVERNMENT_DOCUMENT,
                    url="https://gov.example.com/report",
                    domain="gov.example.com",
                    title="Economic Growth Report 2023",
                    snippet="GDP growth reached 3.2% in the third quarter of 2023.",
                    full_text="The government's quarterly economic report shows GDP growth of 3.2% in Q3 2023, exceeding expectations and indicating strong economic recovery."
                )
            ]
            
            # Add evidence to index
            retriever.add_evidence(evidence_list)
            
            # Check that evidence was added
            assert len(retriever.evidence_list) == 3, f"Expected 3 evidence items, got {len(retriever.evidence_list)}"
            assert retriever.index.ntotal == 3, f"Expected 3 items in index, got {retriever.index.ntotal}"
            
            print("✅ Evidence added successfully to vector index")
            return True
            
    except Exception as e:
        print(f"❌ Evidence addition failed: {e}")
        return False

def test_vector_search():
    """Test vector-based semantic search."""
    print("\n🔍 Testing vector search...")
    
    try:
        from src.evidence_retrieval.vector_search import VectorEvidenceRetriever
        from src.schemas.evidence import Evidence, SourceType
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Initialize retriever
            retriever = VectorEvidenceRetriever(
                index_path=temp_path / "test_index.bin",
                embeddings_path=temp_path / "test_embeddings.pkl",
                use_gpu=False
            )
            
            # Add sample evidence
            evidence_list = [
                Evidence(
                    id="climate_1",
                    claim_id="claim_1",
                    source_type=SourceType.NEWS_ARTICLE,
                    url="https://example.com/climate1",
                    domain="example.com",
                    title="Global Warming Evidence Mounts",
                    snippet="Scientists report unprecedented temperature increases worldwide.",
                    full_text="Scientists from leading research institutions report unprecedented temperature increases worldwide, with data showing clear evidence of global warming trends."
                ),
                Evidence(
                    id="vaccine_1",
                    claim_id="claim_2",
                    source_type=SourceType.RESEARCH_PAPER,
                    url="https://example.com/vaccine1",
                    domain="example.com",
                    title="Vaccine Safety Comprehensive Study",
                    snippet="No correlation found between vaccines and developmental disorders.",
                    full_text="A comprehensive study involving over 500,000 participants found no correlation between vaccination and developmental disorders including autism."
                ),
                Evidence(
                    id="economy_1",
                    claim_id="claim_3",
                    source_type=SourceType.GOVERNMENT_DOCUMENT,
                    url="https://gov.example.com/economy1",
                    domain="gov.example.com",
                    title="Economic Indicators Report",
                    snippet="Strong economic growth reported across all sectors.",
                    full_text="The latest economic indicators report shows strong growth across all sectors, with GDP increasing by 3.5% year-over-year."
                )
            ]
            
            retriever.add_evidence(evidence_list)
            
            # Test search queries
            test_queries = [
                ("global warming temperature increase", "climate_1"),
                ("vaccine autism correlation", "vaccine_1"),
                ("economic growth GDP", "economy_1")
            ]
            
            for query, expected_id in test_queries:
                results = retriever.search(query, top_k=3, similarity_threshold=0.1)
                
                if results:
                    top_result = results[0]
                    assert top_result.evidence.id == expected_id, f"Expected {expected_id}, got {top_result.evidence.id}"
                    print(f"✅ Search for '{query}' returned correct result: {top_result.evidence.id}")
                else:
                    print(f"⚠️ No results for query: {query}")
            
            return True
            
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return False

def test_hybrid_retriever():
    """Test hybrid retriever combining vector and grounded search."""
    print("\n🔄 Testing hybrid retriever...")
    
    try:
        from src.evidence_retrieval.vector_search import (
            VectorEvidenceRetriever, 
            HybridEvidenceRetriever,
            create_vector_retriever,
            create_hybrid_retriever
        )
        from src.schemas.evidence import Evidence, SourceType
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create vector retriever
            vector_retriever = create_vector_retriever(use_gpu=False)
            
            # Add some evidence
            evidence_list = [
                Evidence(
                    id="test_hybrid_1",
                    claim_id="claim_1",
                    source_type=SourceType.NEWS_ARTICLE,
                    url="https://example.com/hybrid1",
                    domain="example.com",
                    title="AI Breakthrough in Medical Diagnosis",
                    snippet="New AI system achieves 95% accuracy in disease detection.",
                    full_text="A new artificial intelligence system has achieved 95% accuracy in medical diagnosis, potentially revolutionizing healthcare delivery."
                )
            ]
            
            vector_retriever.add_evidence(evidence_list)
            
            # Create hybrid retriever (without grounded searcher for testing)
            hybrid_retriever = create_hybrid_retriever(
                vector_retriever=vector_retriever,
                grounded_searcher=None
            )
            
            # Test hybrid search
            results = hybrid_retriever.search(
                query="AI medical diagnosis accuracy",
                top_k=5,
                use_grounded_search=False
            )
            
            assert len(results) > 0, "Hybrid search should return results"
            print(f"✅ Hybrid retriever returned {len(results)} results")
            
            return True
            
    except Exception as e:
        print(f"❌ Hybrid retriever test failed: {e}")
        return False

def test_index_management():
    """Test index management operations."""
    print("\n🗂️ Testing index management...")
    
    try:
        from src.evidence_retrieval.vector_search import VectorEvidenceRetriever
        from src.schemas.evidence import Evidence, SourceType
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Initialize retriever
            retriever = VectorEvidenceRetriever(
                index_path=temp_path / "test_index.bin",
                embeddings_path=temp_path / "test_embeddings.pkl",
                use_gpu=False
            )
            
            # Add evidence
            evidence_list = [
                Evidence(
                    id="keep_1",
                    claim_id="claim_1",
                    source_type=SourceType.NEWS_ARTICLE,
                    url="https://example.com/keep1",
                    domain="example.com",
                    title="Important Research Finding",
                    snippet="This is important research that should be kept.",
                    full_text="This is important research that should be kept in the index."
                ),
                Evidence(
                    id="remove_1",
                    claim_id="claim_2",
                    source_type=SourceType.NEWS_ARTICLE,
                    url="https://example.com/remove1",
                    domain="example.com",
                    title="Outdated Information",
                    snippet="This is outdated information that should be removed.",
                    full_text="This is outdated information that should be removed from the index."
                )
            ]
            
            retriever.add_evidence(evidence_list)
            assert len(retriever.evidence_list) == 2, "Should have 2 evidence items"
            
            # Test index stats
            stats = retriever.get_index_stats()
            assert stats["total_evidence"] == 2, "Stats should show 2 evidence items"
            print("✅ Index stats retrieved successfully")
            
            # Test removing evidence
            retriever.remove_evidence(["remove_1"])
            assert len(retriever.evidence_list) == 1, "Should have 1 evidence item after removal"
            assert retriever.evidence_list[0].id == "keep_1", "Should keep the correct evidence"
            print("✅ Evidence removal successful")
            
            # Test clearing index
            retriever.clear_index()
            assert len(retriever.evidence_list) == 0, "Index should be empty after clearing"
            print("✅ Index clearing successful")
            
            return True
            
    except Exception as e:
        print(f"❌ Index management test failed: {e}")
        return False

def test_performance():
    """Test performance with larger datasets."""
    print("\n⚡ Testing performance...")
    
    try:
        from src.evidence_retrieval.vector_search import VectorEvidenceRetriever
        from src.schemas.evidence import Evidence, SourceType
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Initialize retriever
            retriever = VectorEvidenceRetriever(
                index_path=temp_path / "test_index.bin",
                embeddings_path=temp_path / "test_embeddings.pkl",
                use_gpu=False
            )
            
            # Create larger dataset
            evidence_list = []
            for i in range(50):  # 50 evidence items
                evidence = Evidence(
                    id=f"perf_test_{i}",
                    claim_id=f"claim_{i}",
                    source_type=SourceType.NEWS_ARTICLE,
                    url=f"https://example.com/article{i}",
                    domain="example.com",
                    title=f"Test Article {i}",
                    snippet=f"This is test article {i} with some content about various topics.",
                    full_text=f"This is test article {i} with comprehensive content about various topics including science, technology, and current events."
                )
                evidence_list.append(evidence)
            
            # Measure addition time
            start_time = time.time()
            retriever.add_evidence(evidence_list)
            addition_time = time.time() - start_time
            
            print(f"✅ Added {len(evidence_list)} evidence items in {addition_time:.2f} seconds")
            
            # Measure search time
            start_time = time.time()
            results = retriever.search("test query", top_k=10)
            search_time = time.time() - start_time
            
            print(f"✅ Searched index in {search_time:.4f} seconds")
            
            # Performance assertions
            assert addition_time < 30, f"Addition took too long: {addition_time:.2f}s"
            assert search_time < 1, f"Search took too long: {search_time:.4f}s"
            
            return True
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all vector evidence retrieval tests."""
    print("🧪 Vector Evidence Retrieval Test Suite")
    print("=" * 50)
    
    tests = [
        ("Vector Retriever Initialization", test_vector_retriever_initialization),
        ("Evidence Addition", test_evidence_addition),
        ("Vector Search", test_vector_search),
        ("Hybrid Retriever", test_hybrid_retriever),
        ("Index Management", test_index_management),
        ("Performance", test_performance),
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
        print("🎉 All vector evidence retrieval tests passed!")
        print("🚀 FAISS-based evidence retrieval is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
