#!/usr/bin/env python3
"""
Enhanced Evidence Retrieval System for TruthLens
Implements multiple retrieval methods: BM25, Elasticsearch, and Dense Retrieval
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import re

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

# Optional dependencies
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None

try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    Elasticsearch = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = logging.getLogger(__name__)


@dataclass
class EvidenceDocument:
    """Evidence document structure"""
    id: str
    title: str
    content: str
    url: Optional[str] = None
    source: Optional[str] = None
    domain: Optional[str] = None
    published_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result from evidence retrieval"""
    document: EvidenceDocument
    score: float
    method: str
    rank: int


class BM25Retriever:
    """BM25-based keyword retrieval"""
    
    def __init__(self, documents: List[EvidenceDocument] = None):
        self.documents = documents or []
        self.bm25 = None
        self.tokenized_docs = []
        self._build_index()
    
    def _build_index(self):
        """Build BM25 index from documents"""
        if not self.documents:
            return
        
        # Tokenize documents
        self.tokenized_docs = []
        for doc in self.documents:
            tokens = self._tokenize(f"{doc.title} {doc.content}")
            self.tokenized_docs.append(tokens)
        
        # Build BM25 index
        if BM25_AVAILABLE and self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)
            logger.info(f"Built BM25 index with {len(self.documents)} documents")
        else:
            logger.warning("BM25 not available, using fallback")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove special characters and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [word for word in text.split() if len(word) > 2]
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Search using BM25"""
        if not self.bm25 or not self.documents:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Create results
        results = []
        for i, score in enumerate(scores):
            if score > 0:
                results.append(RetrievalResult(
                    document=self.documents[i],
                    score=float(score),
                    method="BM25",
                    rank=0  # Will be set later
                ))
        
        # Sort by score and assign ranks
        results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1
        
        return results[:top_k]


class ElasticsearchRetriever:
    """Elasticsearch-based retrieval"""
    
    def __init__(self, host: str = "localhost", port: int = 9200, index_name: str = "truthlens_evidence"):
        self.host = host
        self.port = port
        self.index_name = index_name
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Elasticsearch"""
        if not ELASTICSEARCH_AVAILABLE:
            logger.warning("Elasticsearch not available")
            return
        
        try:
            self.client = Elasticsearch([{'host': self.host, 'port': self.port}])
            if self.client.ping():
                logger.info(f"Connected to Elasticsearch at {self.host}:{self.port}")
            else:
                logger.warning("Could not connect to Elasticsearch")
                self.client = None
        except Exception as e:
            logger.error(f"Elasticsearch connection failed: {e}")
            self.client = None
    
    def create_index(self, mapping: Dict = None):
        """Create Elasticsearch index"""
        if not self.client:
            return False
        
        if mapping is None:
            mapping = {
                "mappings": {
                    "properties": {
                        "title": {"type": "text", "analyzer": "standard"},
                        "content": {"type": "text", "analyzer": "standard"},
                        "source": {"type": "keyword"},
                        "domain": {"type": "keyword"},
                        "published_at": {"type": "date"},
                        "url": {"type": "keyword"}
                    }
                }
            }
        
        try:
            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(index=self.index_name, body=mapping)
                logger.info(f"Created Elasticsearch index: {self.index_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def index_documents(self, documents: List[EvidenceDocument]):
        """Index documents in Elasticsearch"""
        if not self.client:
            return False
        
        try:
            for doc in documents:
                body = {
                    "title": doc.title,
                    "content": doc.content,
                    "source": doc.source,
                    "domain": doc.domain,
                    "published_at": doc.published_at,
                    "url": doc.url,
                    "metadata": doc.metadata
                }
                self.client.index(index=self.index_name, id=doc.id, body=body)
            
            self.client.indices.refresh(index=self.index_name)
            logger.info(f"Indexed {len(documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Search using Elasticsearch"""
        if not self.client:
            return []
        
        try:
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "content"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "size": top_k
            }
            
            response = self.client.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                doc = EvidenceDocument(
                    id=hit['_id'],
                    title=source.get('title', ''),
                    content=source.get('content', ''),
                    url=source.get('url'),
                    source=source.get('source'),
                    domain=source.get('domain'),
                    published_at=source.get('published_at'),
                    metadata=source.get('metadata', {})
                )
                
                results.append(RetrievalResult(
                    document=doc,
                    score=float(hit['_score']),
                    method="Elasticsearch",
                    rank=len(results) + 1
                ))
            
            return results
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []


class DenseRetriever:
    """Dense retrieval using sentence transformers"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/msmarco-MiniLM-L-6-v2",
        documents: List[EvidenceDocument] = None,
        use_gpu: bool = True
    ):
        self.model_name = model_name
        self.documents = documents or []
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = None
        self.embeddings = None
        self.index = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the dense retriever"""
        try:
            self.model = SentenceTransformer(self.model_name)
            if self.use_gpu:
                self.model = self.model.to('cuda')
            logger.info(f"Loaded dense retriever model: {self.model_name}")
            
            if self.documents:
                self._build_index()
        except Exception as e:
            logger.error(f"Failed to initialize dense retriever: {e}")
    
    def _build_index(self):
        """Build FAISS index for documents"""
        if not FAISS_AVAILABLE or not self.documents:
            return
        
        try:
            # Create document texts
            texts = [f"{doc.title} {doc.content}" for doc in self.documents]
            
            # Compute embeddings
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Build FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings.astype('float32'))
            
            logger.info(f"Built FAISS index with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Search using dense retrieval"""
        if not self.index or not self.documents:
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.documents))
            )
            
            # Create results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    results.append(RetrievalResult(
                        document=self.documents[idx],
                        score=float(score),
                        method="Dense",
                        rank=i + 1
                    ))
            
            return results
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []


class CrossEncoderReranker:
    """Cross-encoder for reranking results"""
    
    def __init__(self, model_name: str = "cross-encoder/msmarco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize cross-encoder"""
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded cross-encoder: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
    
    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 10) -> List[RetrievalResult]:
        """Rerank results using cross-encoder"""
        if not self.model or not results:
            return results
        
        try:
            # Prepare pairs for cross-encoder
            pairs = []
            for result in results:
                text = f"{result.document.title} {result.document.content}"
                pairs.append([query, text])
            
            # Get scores
            scores = self.model.predict(pairs)
            
            # Update results with cross-encoder scores
            for result, score in zip(results, scores):
                result.score = float(score)
                result.method = "CrossEncoder"
            
            # Sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(results[:top_k]):
                result.rank = i + 1
            
            return results[:top_k]
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return results


class EnhancedEvidenceRetriever:
    """
    Enhanced evidence retrieval system combining multiple methods:
    - BM25 for keyword-based retrieval
    - Elasticsearch for full-text search
    - Dense retrieval for semantic search
    - Cross-encoder for reranking
    """
    
    def __init__(
        self,
        bm25_documents: List[EvidenceDocument] = None,
        elasticsearch_config: Dict = None,
        dense_model: str = "sentence-transformers/msmarco-MiniLM-L-6-v2",
        cross_encoder_model: str = "cross-encoder/msmarco-MiniLM-L-6-v2",
        use_gpu: bool = True
    ):
        # Initialize retrievers
        self.bm25_retriever = BM25Retriever(bm25_documents) if BM25_AVAILABLE else None
        
        if elasticsearch_config:
            self.elasticsearch_retriever = ElasticsearchRetriever(**elasticsearch_config)
        else:
            self.elasticsearch_retriever = None
        
        self.dense_retriever = DenseRetriever(
            model_name=dense_model,
            documents=bm25_documents,
            use_gpu=use_gpu
        )
        
        self.cross_encoder = CrossEncoderReranker(cross_encoder_model)
        
        logger.info("Enhanced evidence retriever initialized")
    
    def search(
        self,
        query: str,
        methods: List[str] = None,
        top_k: int = 10,
        use_reranking: bool = True
    ) -> List[RetrievalResult]:
        """
        Search for evidence using multiple methods
        
        Args:
            query: Search query
            methods: List of methods to use ['bm25', 'elasticsearch', 'dense']
            top_k: Number of results to return
            use_reranking: Whether to use cross-encoder reranking
        
        Returns:
            List of retrieval results
        """
        if methods is None:
            methods = ['bm25', 'dense']
        
        all_results = []
        
        # BM25 search
        if 'bm25' in methods and self.bm25_retriever:
            bm25_results = self.bm25_retriever.search(query, top_k * 2)
            all_results.extend(bm25_results)
            logger.info(f"BM25 found {len(bm25_results)} results")
        
        # Elasticsearch search
        if 'elasticsearch' in methods and self.elasticsearch_retriever:
            es_results = self.elasticsearch_retriever.search(query, top_k * 2)
            all_results.extend(es_results)
            logger.info(f"Elasticsearch found {len(es_results)} results")
        
        # Dense search
        if 'dense' in methods and self.dense_retriever:
            dense_results = self.dense_retriever.search(query, top_k * 2)
            all_results.extend(dense_results)
            logger.info(f"Dense retrieval found {len(dense_results)} results")
        
        # Remove duplicates based on document ID
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.document.id not in seen_ids:
                seen_ids.add(result.document.id)
                unique_results.append(result)
        
        # Rerank if requested
        if use_reranking and unique_results:
            unique_results = self.cross_encoder.rerank(query, unique_results, top_k)
        
        # Sort by score and return top_k
        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results[:top_k]
    
    def add_documents(self, documents: List[EvidenceDocument]):
        """Add documents to all retrievers"""
        if self.bm25_retriever:
            self.bm25_retriever.documents.extend(documents)
            self.bm25_retriever._build_index()
        
        if self.dense_retriever:
            self.dense_retriever.documents.extend(documents)
            self.dense_retriever._build_index()
        
        if self.elasticsearch_retriever:
            self.elasticsearch_retriever.index_documents(documents)
        
        logger.info(f"Added {len(documents)} documents to retrievers")


def create_sample_documents() -> List[EvidenceDocument]:
    """Create sample documents for testing"""
    return [
        EvidenceDocument(
            id="1",
            title="COVID-19 Vaccines and Autism",
            content="Multiple large-scale studies have found no link between COVID-19 vaccines and autism. The original study suggesting this link was retracted due to methodological flaws.",
            source="Scientific Journal",
            domain="science.org",
            published_at="2023-01-15"
        ),
        EvidenceDocument(
            id="2", 
            title="5G Technology Safety",
            content="5G technology uses non-ionizing radiation that is safe for human exposure. The World Health Organization has confirmed that 5G radiation levels are well below safety limits.",
            source="WHO",
            domain="who.int",
            published_at="2023-02-20"
        ),
        EvidenceDocument(
            id="3",
            title="Moon Landing Evidence",
            content="The Apollo 11 moon landing in 1969 is supported by extensive evidence including photographs, rock samples, and independent tracking by multiple countries.",
            source="NASA",
            domain="nasa.gov",
            published_at="1969-07-20"
        )
    ]


def main():
    """Test the enhanced evidence retriever"""
    print("🧪 Testing Enhanced Evidence Retriever")
    print("=" * 50)
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"📚 Created {len(documents)} sample documents")
    
    # Initialize retriever
    retriever = EnhancedEvidenceRetriever(
        bm25_documents=documents,
        dense_model="sentence-transformers/all-MiniLM-L6-v2"  # Smaller model for testing
    )
    
    # Test queries
    test_queries = [
        "COVID-19 vaccines cause autism",
        "5G towers are dangerous",
        "The moon landing was faked"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = retriever.search(query, methods=['bm25', 'dense'], top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.document.title} (score: {result.score:.3f}, method: {result.method})")
    
    print("\n✅ Enhanced evidence retriever test completed!")


if __name__ == "__main__":
    main()
