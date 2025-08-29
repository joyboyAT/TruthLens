"""
Vector-based Evidence Retrieval for TruthLens
Uses FAISS for fast local semantic search with sentence transformers.
Enhanced with better semantic search and deduplication.
"""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Import schemas
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from schemas.evidence import Evidence, SourceType

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector search with similarity score."""
    evidence: Evidence
    similarity_score: float
    rank: int
    cluster_id: Optional[int] = None


class VectorEvidenceRetriever:
    """
    FAISS-based vector search for evidence retrieval.
    
    Enhanced Features:
    - Local semantic search using sentence transformers (all-MiniLM-L6-v2)
    - FAISS index for fast similarity search
    - Support for multiple embedding models
    - Automatic index persistence and loading
    - Content deduplication and clustering
    - Improved semantic ranking
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",  # Better model for semantic search
        index_path: Optional[str] = None,
        embeddings_path: Optional[str] = None,
        dimension: int = 384,  # all-MiniLM-L6-v2 dimension
        use_gpu: bool = False,
        deduplication_threshold: float = 0.95,
        clustering_eps: float = 0.3
    ):
        """
        Initialize the vector retriever.
        
        Args:
            model_name: Sentence transformer model name (default: all-MiniLM-L6-v2)
            index_path: Path to save/load FAISS index
            embeddings_path: Path to save/load embeddings cache
            dimension: Embedding dimension
            use_gpu: Whether to use GPU for embeddings
            deduplication_threshold: Similarity threshold for deduplication
            clustering_eps: Epsilon for DBSCAN clustering
        """
        self.model_name = model_name
        self.dimension = dimension
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.deduplication_threshold = deduplication_threshold
        self.clustering_eps = clustering_eps
        
        # Set default paths
        if index_path is None:
            index_path = Path("data/processed/faiss_index.bin")
        if embeddings_path is None:
            embeddings_path = Path("data/processed/embeddings_cache.pkl")
        
        self.index_path = Path(index_path)
        self.embeddings_path = Path(embeddings_path)
        
        # Initialize components
        self.model = None
        self.index = None
        self.evidence_list = []
        self.embeddings = None
        
        # Load or initialize
        self._initialize_model()
        self._load_or_create_index()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            if self.use_gpu:
                self.model = self.model.to('cuda')
                logger.info("Model loaded on GPU")
            else:
                logger.info("Model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to a simpler model
            try:
                logger.info("Falling back to all-MiniLM-L6-v2")
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.dimension = 384
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise
    
    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        try:
            if self.index_path.exists() and self.embeddings_path.exists():
                logger.info("Loading existing FAISS index and embeddings")
                self.index = faiss.read_index(str(self.index_path))
                
                with open(self.embeddings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.evidence_list = data.get('evidence_list', [])
                    self.embeddings = data.get('embeddings', None)
                
                logger.info(f"Loaded {len(self.evidence_list)} evidence items")
            else:
                logger.info("Creating new FAISS index")
                self._create_new_index()
                
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            logger.info("Creating new FAISS index")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.evidence_list = []
        self.embeddings = None
    
    def _get_content_hash(self, content: str) -> str:
        """Generate a hash for content deduplication."""
        return hashlib.md5(content.lower().encode()).hexdigest()
    
    def _deduplicate_evidence(self, evidence_list: List[Evidence]) -> List[Evidence]:
        """Remove duplicate evidence based on content similarity."""
        if not evidence_list:
            return evidence_list
        
        # Generate embeddings for all evidence
        texts = [f"{e.title} {e.content}" for e in evidence_list]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Find duplicates
        seen_indices = set()
        unique_evidence = []
        
        for i in range(len(evidence_list)):
            if i in seen_indices:
                continue
            
            unique_evidence.append(evidence_list[i])
            seen_indices.add(i)
            
            # Find similar items
            for j in range(i + 1, len(evidence_list)):
                if j not in seen_indices and similarities[i][j] > self.deduplication_threshold:
                    seen_indices.add(j)
                    logger.debug(f"Removing duplicate: {evidence_list[j].title}")
        
        logger.info(f"Deduplicated {len(evidence_list)} evidence to {len(unique_evidence)} unique items")
        return unique_evidence
    
    def _cluster_evidence(self, evidence_list: List[Evidence], embeddings: np.ndarray) -> List[Tuple[int, List[int]]]:
        """Cluster evidence using DBSCAN."""
        if len(evidence_list) < 2:
            return []
        
        # Perform clustering
        clustering = DBSCAN(eps=self.clustering_eps, min_samples=2, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Group evidence by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label >= 0:  # Skip noise points
                clusters[label].append(i)
        
        # Convert to list of tuples
        cluster_list = [(cluster_id, indices) for cluster_id, indices in clusters.items()]
        
        logger.info(f"Created {len(cluster_list)} clusters from {len(evidence_list)} evidence items")
        return cluster_list
    
    def add_evidence(self, evidence_list: List[Evidence]) -> bool:
        """
        Add evidence to the index with deduplication and clustering.
        
        Args:
            evidence_list: List of evidence to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not evidence_list:
                return True
            
            logger.info(f"Adding {len(evidence_list)} evidence items")
            
            # Deduplicate evidence
            unique_evidence = self._deduplicate_evidence(evidence_list)
            
            if not unique_evidence:
                return True
            
            # Generate embeddings
            texts = [f"{e.title} {e.content}" for e in unique_evidence]
            new_embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Add to existing evidence
            if self.evidence_list:
                # Combine with existing embeddings
                if self.embeddings is not None:
                    combined_embeddings = np.vstack([self.embeddings, new_embeddings])
                else:
                    combined_embeddings = new_embeddings
                
                # Update index
                self.index.reset()
                self.index.add(combined_embeddings.astype('float32'))
                
                # Update evidence list and embeddings
                self.evidence_list.extend(unique_evidence)
                self.embeddings = combined_embeddings
            else:
                # First time adding evidence
                self.evidence_list = unique_evidence
                self.embeddings = new_embeddings
                self.index.add(new_embeddings.astype('float32'))
            
            # Save index and embeddings
            self._save_index()
            
            logger.info(f"Successfully added {len(unique_evidence)} evidence items")
            return True
            
        except Exception as e:
            logger.error(f"Error adding evidence: {e}")
            return False
    
    def search_evidence(self, query: str, top_k: int = 10, 
                       apply_clustering: bool = True) -> List[VectorSearchResult]:
        """
        Search for evidence using semantic similarity with clustering.
        
        Args:
            query: Search query
            top_k: Number of results to return
            apply_clustering: Whether to apply clustering to results
            
        Returns:
            List of VectorSearchResult with clustering information
        """
        try:
            if not self.evidence_list or self.embeddings is None:
                logger.warning("No evidence available for search")
                return []
            
            # Encode query
            query_embedding = self.model.encode([query])
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k * 2, len(self.evidence_list)))
            
            # Create results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.evidence_list):
                    evidence = self.evidence_list[idx]
                    result = VectorSearchResult(
                        evidence=evidence,
                        similarity_score=float(score),
                        rank=i + 1
                    )
                    results.append(result)
            
            # Apply clustering if requested
            if apply_clustering and len(results) > 1:
                results = self._apply_result_clustering(results)
            
            # Return top_k results
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching evidence: {e}")
            return []
    
    def _apply_result_clustering(self, results: List[VectorSearchResult]) -> List[VectorSearchResult]:
        """Apply clustering to search results to group similar evidence."""
        if len(results) < 2:
            return results
        
        # Extract embeddings for results
        result_embeddings = []
        for result in results:
            idx = self.evidence_list.index(result.evidence)
            result_embeddings.append(self.embeddings[idx])
        
        result_embeddings = np.array(result_embeddings)
        
        # Perform clustering
        clusters = self._cluster_evidence([r.evidence for r in results], result_embeddings)
        
        # Assign cluster IDs to results
        cluster_map = {}
        for cluster_id, indices in clusters:
            for idx in indices:
                cluster_map[idx] = cluster_id
        
        # Update results with cluster information
        for i, result in enumerate(results):
            result.cluster_id = cluster_map.get(i)
        
        # Sort by cluster (similar items together) and then by similarity
        results.sort(key=lambda x: (x.cluster_id if x.cluster_id is not None else -1, -x.similarity_score))
        
        return results
    
    def _save_index(self):
        """Save the FAISS index and embeddings."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save embeddings and evidence list
            data = {
                'evidence_list': self.evidence_list,
                'embeddings': self.embeddings
            }
            
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info("Index and embeddings saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        return {
            'total_evidence': len(self.evidence_list),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.dimension,
            'model_name': self.model_name,
            'index_path': str(self.index_path),
            'embeddings_path': str(self.embeddings_path)
        }
    
    def clear_index(self):
        """Clear the index and all data."""
        self._create_new_index()
        self._save_index()
        logger.info("Index cleared successfully")


class HybridEvidenceRetriever:
    """
    Hybrid retriever that combines vector search with grounded search.
    """
    
    def __init__(
        self,
        vector_retriever: VectorEvidenceRetriever,
        grounded_searcher: Optional[Any] = None,
        vector_weight: float = 0.7,
        grounded_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_retriever: Vector-based retriever
            grounded_searcher: Grounded search retriever
            vector_weight: Weight for vector search results
            grounded_weight: Weight for grounded search results
        """
        self.vector_retriever = vector_retriever
        self.grounded_searcher = grounded_searcher
        self.vector_weight = vector_weight
        self.grounded_weight = grounded_weight
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_grounded_search: bool = True
    ) -> List[Evidence]:
        """
        Perform hybrid search combining vector and grounded search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_grounded_search: Whether to use grounded search
            
        Returns:
            List of Evidence objects
        """
        results = []
        
        # Vector search
        vector_results = self.vector_retriever.search_evidence(query, top_k=top_k)
        results.extend([r.evidence for r in vector_results])
        
        # Grounded search (if available and enabled)
        if use_grounded_search and self.grounded_searcher:
            try:
                grounded_results = self.grounded_searcher.search(
                    query, top_k=top_k//2, days=90
                )
                results.extend(grounded_results)
            except Exception as e:
                logger.warning(f"Grounded search failed: {e}")
        
        # Remove duplicates and limit results
        seen_ids = set()
        unique_results = []
        for evidence in results:
            if evidence.id not in seen_ids:
                seen_ids.add(evidence.id)
                unique_results.append(evidence)
                if len(unique_results) >= top_k:
                    break
        
        return unique_results


def create_vector_retriever(
    model_name: str = "all-MiniLM-L6-v2",
    use_gpu: bool = False
) -> VectorEvidenceRetriever:
    """
    Factory function to create a vector retriever.
    
    Args:
        model_name: Sentence transformer model name
        use_gpu: Whether to use GPU
        
    Returns:
        VectorEvidenceRetriever instance
    """
    return VectorEvidenceRetriever(
        model_name=model_name,
        use_gpu=use_gpu
    )


def create_hybrid_retriever(
    vector_retriever: VectorEvidenceRetriever,
    grounded_searcher: Optional[Any] = None
) -> HybridEvidenceRetriever:
    """
    Factory function to create a hybrid retriever.
    
    Args:
        vector_retriever: Vector-based retriever
        grounded_searcher: Grounded search retriever
        
    Returns:
        HybridEvidenceRetriever instance
    """
    return HybridEvidenceRetriever(
        vector_retriever=vector_retriever,
        grounded_searcher=grounded_searcher
    )
