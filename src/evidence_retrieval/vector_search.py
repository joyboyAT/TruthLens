"""
Vector-based Evidence Retrieval for TruthLens
Uses FAISS for fast local semantic search with sentence transformers.
"""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch

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


class VectorEvidenceRetriever:
    """
    FAISS-based vector search for evidence retrieval.
    
    Features:
    - Local semantic search using sentence transformers
    - FAISS index for fast similarity search
    - Support for multiple embedding models
    - Automatic index persistence and loading
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
        embeddings_path: Optional[str] = None,
        dimension: int = 384,
        use_gpu: bool = False
    ):
        """
        Initialize the vector retriever.
        
        Args:
            model_name: Sentence transformer model name
            index_path: Path to save/load FAISS index
            embeddings_path: Path to save/load embeddings cache
            dimension: Embedding dimension
            use_gpu: Whether to use GPU for embeddings
        """
        self.model_name = model_name
        self.dimension = dimension
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
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
            # Fallback to a smaller model
            self.model_name = "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name)
            if self.use_gpu:
                self.model = self.model.to('cuda')
    
    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        if self.index_path.exists() and self.embeddings_path.exists():
            try:
                logger.info("Loading existing FAISS index and embeddings")
                self.index = faiss.read_index(str(self.index_path))
                
                with open(self.embeddings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.evidence_list = data['evidence_list']
                    self.embeddings = data['embeddings']
                
                logger.info(f"Loaded index with {len(self.evidence_list)} evidence items")
                return
                
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
        
        # Create new index
        logger.info("Creating new FAISS index")
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.evidence_list = []
        self.embeddings = np.array([]).reshape(0, self.dimension)
    
    def _save_index(self):
        """Save the FAISS index and embeddings."""
        try:
            # Ensure directory exists
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save embeddings and evidence
            data = {
                'evidence_list': self.evidence_list,
                'embeddings': self.embeddings
            }
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved index with {len(self.evidence_list)} evidence items")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def add_evidence(self, evidence_list: List[Evidence]) -> None:
        """
        Add evidence to the vector index.
        
        Args:
            evidence_list: List of Evidence objects to add
        """
        if not evidence_list:
            return
        
        logger.info(f"Adding {len(evidence_list)} evidence items to vector index")
        
        # Prepare texts for embedding
        texts = []
        for evidence in evidence_list:
            # Combine title and snippet for better semantic search
            text = f"{evidence.title} {evidence.snippet}"
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        if self.index.ntotal == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.index.add(embeddings.astype('float32'))
        self.evidence_list.extend(evidence_list)
        
        # Save updated index
        self._save_index()
        
        logger.info(f"Index now contains {len(self.evidence_list)} evidence items")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[VectorSearchResult]:
        """
        Search for evidence using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of VectorSearchResult objects
        """
        if self.index.ntotal == 0:
            logger.warning("No evidence in index")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search in FAISS index
        scores, indices = self.index.search(
            query_embedding.astype('float32'),
            min(top_k * 2, self.index.ntotal)  # Get more results to filter by threshold
        )
        
        # Convert to results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            if score >= similarity_threshold:
                result = VectorSearchResult(
                    evidence=self.evidence_list[idx],
                    similarity_score=float(score),
                    rank=i + 1
                )
                results.append(result)
                
                if len(results) >= top_k:
                    break
        
        logger.info(f"Found {len(results)} evidence items for query: {query}")
        return results
    
    def search_by_claim(
        self,
        claim_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[VectorSearchResult]:
        """
        Search for evidence relevant to a specific claim.
        
        Args:
            claim_text: The claim to find evidence for
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of VectorSearchResult objects
        """
        return self.search(claim_text, top_k, similarity_threshold)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        return {
            "total_evidence": len(self.evidence_list),
            "index_size": self.index.ntotal,
            "embedding_dimension": self.dimension,
            "model_name": self.model_name,
            "use_gpu": self.use_gpu,
            "index_path": str(self.index_path),
            "embeddings_path": str(self.embeddings_path)
        }
    
    def clear_index(self) -> None:
        """Clear the entire index."""
        logger.info("Clearing vector index")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.evidence_list = []
        self.embeddings = np.array([]).reshape(0, self.dimension)
        self._save_index()
    
    def remove_evidence(self, evidence_ids: List[str]) -> None:
        """
        Remove specific evidence from the index.
        Note: This requires rebuilding the index.
        
        Args:
            evidence_ids: List of evidence IDs to remove
        """
        logger.info(f"Removing {len(evidence_ids)} evidence items from index")
        
        # Filter out evidence to remove
        filtered_evidence = [
            ev for ev in self.evidence_list 
            if ev.id not in evidence_ids
        ]
        
        if len(filtered_evidence) == len(self.evidence_list):
            logger.warning("No evidence items found to remove")
            return
        
        # Rebuild index with filtered evidence
        self.clear_index()
        self.add_evidence(filtered_evidence)
        
        logger.info(f"Index rebuilt with {len(filtered_evidence)} evidence items")


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
        vector_results = self.vector_retriever.search(query, top_k=top_k)
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
