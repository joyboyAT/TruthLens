-- TruthLens PostgreSQL Schema Setup
-- Generated automatically

-- Table 1

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
    

-- Table 2

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
    

-- Table 3

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
    

