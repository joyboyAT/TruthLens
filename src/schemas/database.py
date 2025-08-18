"""
Database schemas for TruthLens evidence storage.

This module defines the database table schemas for storing evidence data,
including raw evidence, chunks, and claim-evidence relationships.
"""

from typing import List, Dict, Any
from datetime import datetime


# BigQuery Schema Definitions
BIGQUERY_SCHEMAS = {
    "evidence_raw": {
        "fields": [
            {"name": "id", "type": "STRING", "mode": "REQUIRED"},
            {"name": "claim_id", "type": "STRING", "mode": "REQUIRED"},
            {"name": "source_type", "type": "STRING", "mode": "REQUIRED"},
            {"name": "url", "type": "STRING", "mode": "REQUIRED"},
            {"name": "domain", "type": "STRING", "mode": "REQUIRED"},
            {"name": "title", "type": "STRING", "mode": "REQUIRED"},
            {"name": "published_at", "type": "TIMESTAMP", "mode": "NULLABLE"},
            {"name": "retrieved_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
            {"name": "language", "type": "STRING", "mode": "REQUIRED"},
            {"name": "snippet", "type": "STRING", "mode": "NULLABLE"},
            {"name": "full_text", "type": "STRING", "mode": "NULLABLE"},
            {"name": "full_text_hash", "type": "STRING", "mode": "REQUIRED"},
            {"name": "chunk_ids", "type": "STRING", "mode": "REPEATED"},
            {"name": "support_label", "type": "STRING", "mode": "REQUIRED"},
            {"name": "relevance_score", "type": "FLOAT64", "mode": "NULLABLE"},
            {"name": "freshness_score", "type": "FLOAT64", "mode": "NULLABLE"},
            {"name": "source_score", "type": "FLOAT64", "mode": "NULLABLE"},
            {"name": "final_score", "type": "FLOAT64", "mode": "NULLABLE"},
            {"name": "metadata", "type": "JSON", "mode": "NULLABLE"},
            {"name": "created_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
            {"name": "updated_at", "type": "TIMESTAMP", "mode": "REQUIRED"}
        ],
        "description": "Raw evidence data with full text content"
    },
    
    "evidence_chunks": {
        "fields": [
            {"name": "chunk_id", "type": "STRING", "mode": "REQUIRED"},
            {"name": "evidence_id", "type": "STRING", "mode": "REQUIRED"},
            {"name": "chunk_index", "type": "INT64", "mode": "REQUIRED"},
            {"name": "text", "type": "STRING", "mode": "REQUIRED"},
            {"name": "text_hash", "type": "STRING", "mode": "REQUIRED"},
            {"name": "embedding", "type": "FLOAT64", "mode": "REPEATED"},
            {"name": "language", "type": "STRING", "mode": "REQUIRED"},
            {"name": "metadata", "type": "JSON", "mode": "NULLABLE"},
            {"name": "created_at", "type": "TIMESTAMP", "mode": "REQUIRED"}
        ],
        "description": "Evidence text chunks with embeddings for vector search"
    },
    
    "claims_evidence": {
        "fields": [
            {"name": "claim_id", "type": "STRING", "mode": "REQUIRED"},
            {"name": "evidence_id", "type": "STRING", "mode": "REQUIRED"},
            {"name": "support_label", "type": "STRING", "mode": "REQUIRED"},
            {"name": "relevance_score", "type": "FLOAT64", "mode": "NULLABLE"},
            {"name": "freshness_score", "type": "FLOAT64", "mode": "NULLABLE"},
            {"name": "source_score", "type": "FLOAT64", "mode": "NULLABLE"},
            {"name": "final_score", "type": "FLOAT64", "mode": "NULLABLE"},
            {"name": "confidence", "type": "FLOAT64", "mode": "NULLABLE"},
            {"name": "annotator_id", "type": "STRING", "mode": "NULLABLE"},
            {"name": "annotated_at", "type": "TIMESTAMP", "mode": "NULLABLE"},
            {"name": "created_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
            {"name": "updated_at", "type": "TIMESTAMP", "mode": "REQUIRED"}
        ],
        "description": "Many-to-many relationship between claims and evidence with labels and scores"
    }
}


# PostgreSQL Schema Definitions
POSTGRES_SCHEMAS = {
    "evidence_raw": """
        CREATE TABLE evidence_raw (
            id VARCHAR(255) PRIMARY KEY,
            claim_id VARCHAR(255) NOT NULL,
            source_type VARCHAR(50) NOT NULL,
            url TEXT NOT NULL,
            domain VARCHAR(255) NOT NULL,
            title TEXT NOT NULL,
            published_at TIMESTAMP,
            retrieved_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            language VARCHAR(10) NOT NULL DEFAULT 'en',
            snippet TEXT,
            full_text TEXT,
            full_text_hash VARCHAR(64) NOT NULL,
            chunk_ids TEXT[], -- Array of chunk IDs
            support_label VARCHAR(20) NOT NULL DEFAULT 'neutral',
            relevance_score DECIMAL(5,4),
            freshness_score DECIMAL(5,4),
            source_score DECIMAL(5,4),
            final_score DECIMAL(5,4),
            metadata JSONB,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Indexes for performance
        CREATE INDEX idx_evidence_raw_claim_id ON evidence_raw(claim_id);
        CREATE INDEX idx_evidence_raw_source_type ON evidence_raw(source_type);
        CREATE INDEX idx_evidence_raw_domain ON evidence_raw(domain);
        CREATE INDEX idx_evidence_raw_published_at ON evidence_raw(published_at);
        CREATE INDEX idx_evidence_raw_full_text_hash ON evidence_raw(full_text_hash);
        CREATE INDEX idx_evidence_raw_support_label ON evidence_raw(support_label);
        CREATE INDEX idx_evidence_raw_final_score ON evidence_raw(final_score);
        CREATE INDEX idx_evidence_raw_metadata ON evidence_raw USING GIN(metadata);
    """,
    
    "evidence_chunks": """
        CREATE TABLE evidence_chunks (
            chunk_id VARCHAR(255) PRIMARY KEY,
            evidence_id VARCHAR(255) NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            text_hash VARCHAR(64) NOT NULL,
            embedding VECTOR(1536), -- Assuming 1536-dimensional embeddings
            language VARCHAR(10) NOT NULL DEFAULT 'en',
            metadata JSONB,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (evidence_id) REFERENCES evidence_raw(id) ON DELETE CASCADE
        );
        
        -- Indexes for performance
        CREATE INDEX idx_evidence_chunks_evidence_id ON evidence_chunks(evidence_id);
        CREATE INDEX idx_evidence_chunks_chunk_index ON evidence_chunks(evidence_id, chunk_index);
        CREATE INDEX idx_evidence_chunks_text_hash ON evidence_chunks(text_hash);
        CREATE INDEX idx_evidence_chunks_language ON evidence_chunks(language);
        CREATE INDEX idx_evidence_chunks_embedding ON evidence_chunks USING ivfflat (embedding vector_cosine_ops);
    """,
    
    "claims_evidence": """
        CREATE TABLE claims_evidence (
            claim_id VARCHAR(255) NOT NULL,
            evidence_id VARCHAR(255) NOT NULL,
            support_label VARCHAR(20) NOT NULL DEFAULT 'neutral',
            relevance_score DECIMAL(5,4),
            freshness_score DECIMAL(5,4),
            source_score DECIMAL(5,4),
            final_score DECIMAL(5,4),
            confidence DECIMAL(5,4),
            annotator_id VARCHAR(255),
            annotated_at TIMESTAMP,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (claim_id, evidence_id),
            FOREIGN KEY (evidence_id) REFERENCES evidence_raw(id) ON DELETE CASCADE
        );
        
        -- Indexes for performance
        CREATE INDEX idx_claims_evidence_claim_id ON claims_evidence(claim_id);
        CREATE INDEX idx_claims_evidence_evidence_id ON claims_evidence(evidence_id);
        CREATE INDEX idx_claims_evidence_support_label ON claims_evidence(support_label);
        CREATE INDEX idx_claims_evidence_final_score ON claims_evidence(final_score);
        CREATE INDEX idx_claims_evidence_annotator_id ON claims_evidence(annotator_id);
    """
}


# Vector Database Schema (for PostgreSQL with pgvector extension)
VECTOR_DB_SCHEMAS = {
    "evidence_chunks": """
        -- Enable pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;
        
        -- Create evidence_chunks table with vector support
        CREATE TABLE evidence_chunks (
            chunk_id VARCHAR(255) PRIMARY KEY,
            evidence_id VARCHAR(255) NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            text_hash VARCHAR(64) NOT NULL,
            embedding vector(1536), -- 1536-dimensional embeddings
            language VARCHAR(10) NOT NULL DEFAULT 'en',
            metadata JSONB,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (evidence_id) REFERENCES evidence_raw(id) ON DELETE CASCADE
        );
        
        -- Vector-specific indexes
        CREATE INDEX idx_evidence_chunks_embedding_cosine ON evidence_chunks 
            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        CREATE INDEX idx_evidence_chunks_embedding_l2 ON evidence_chunks 
            USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
        CREATE INDEX idx_evidence_chunks_embedding_ip ON evidence_chunks 
            USING ivfflat (embedding vector_ip_ops) WITH (lists = 100);
    """
}


def get_bigquery_schema(table_name: str) -> Dict[str, Any]:
    """Get BigQuery schema for a specific table."""
    if table_name not in BIGQUERY_SCHEMAS:
        raise ValueError(f"Unknown table: {table_name}")
    return BIGQUERY_SCHEMAS[table_name]


def get_postgres_schema(table_name: str) -> str:
    """Get PostgreSQL schema for a specific table."""
    if table_name not in POSTGRES_SCHEMAS:
        raise ValueError(f"Unknown table: {table_name}")
    return POSTGRES_SCHEMAS[table_name]


def get_vector_db_schema(table_name: str) -> str:
    """Get vector database schema for a specific table."""
    if table_name not in VECTOR_DB_SCHEMAS:
        raise ValueError(f"Unknown table: {table_name}")
    return VECTOR_DB_SCHEMAS[table_name]


def create_all_bigquery_schemas() -> Dict[str, Dict[str, Any]]:
    """Get all BigQuery schemas."""
    return BIGQUERY_SCHEMAS.copy()


def create_all_postgres_schemas() -> List[str]:
    """Get all PostgreSQL schema creation statements."""
    return list(POSTGRES_SCHEMAS.values())


def create_all_vector_db_schemas() -> List[str]:
    """Get all vector database schema creation statements."""
    return list(VECTOR_DB_SCHEMAS.values())
