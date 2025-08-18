# TruthLens Evidence Schema

This directory contains the core data schemas for the TruthLens fact-checking system, specifically focused on evidence management and storage.

## Overview

The Evidence schema provides a comprehensive data structure for storing and managing evidence used in claim verification. It includes:

- **Evidence dataclass**: Core data structure for evidence items
- **Database schemas**: BigQuery and PostgreSQL table definitions
- **Vector database support**: For semantic search and similarity matching

## Files

### `evidence.py`
Contains the main `Evidence` dataclass and related components:

- `Evidence`: Main dataclass for evidence data
- `EvidenceScores`: Scoring system for evidence quality
- `SourceType`: Enumeration of evidence source types
- `SupportLabel`: Enumeration of evidence support levels

### `database.py`
Contains database schema definitions for:

- **BigQuery**: Field definitions and table schemas
- **PostgreSQL**: SQL CREATE TABLE statements with indexes
- **Vector Database**: pgvector extension schemas for embeddings

## Evidence Schema Fields

### Core Identifiers
- `id`: Unique evidence identifier
- `claim_id`: Associated claim identifier

### Source Metadata
- `source_type`: Type of evidence source (news, research, social media, etc.)
- `url`: Source URL
- `domain`: Source domain
- `title`: Evidence title

### Temporal Information
- `published_at`: When the evidence was originally published
- `retrieved_at`: When the evidence was retrieved by TruthLens

### Content Information
- `language`: Content language (default: "en")
- `snippet`: Short excerpt from the evidence
- `full_text`: Complete text content
- `full_text_hash`: xxhash64 hash of the full text

### Chunking and Embeddings
- `chunk_ids`: List of text chunk identifiers for vector search

### Classification and Scoring
- `support_label`: How the evidence relates to the claim
- `scores`: Quality and relevance scores

### Additional Metadata
- `metadata`: Flexible JSON field for additional information

## Source Types

- `news_article`: News articles from established media
- `research_paper`: Academic or scientific papers
- `government_document`: Official government documents
- `social_media`: Posts from social media platforms
- `blog_post`: Blog articles and posts
- `video`: Video content
- `podcast`: Audio content
- `other`: Miscellaneous sources

## Support Labels

- `supports`: Evidence supports the claim
- `refutes`: Evidence contradicts the claim
- `neutral`: Evidence is neutral to the claim
- `mixed`: Evidence has mixed support/refutation
- `unclear`: Evidence relationship is unclear

## Scoring System

The `EvidenceScores` class provides a weighted scoring system:

- **Relevance** (40%): How relevant is the evidence to the claim
- **Freshness** (20%): How recent is the evidence
- **Source** (40%): Source credibility and reliability
- **Final**: Weighted combination of the above scores

## Database Tables

### `evidence_raw`
Stores the main evidence data including full text content.

### `evidence_chunks`
Stores text chunks with embeddings for vector similarity search.

### `claims_evidence`
Many-to-many relationship table linking claims to evidence with labels and scores.

## Usage Examples

### Creating Evidence

```python
from schemas.evidence import Evidence, SourceType, SupportLabel, EvidenceScores

evidence = Evidence(
    id="ev_001",
    claim_id="claim_001",
    source_type=SourceType.NEWS_ARTICLE,
    url="https://example.com/article",
    domain="example.com",
    title="Example Article",
    full_text="Article content here...",
    support_label=SupportLabel.SUPPORTS,
    scores=EvidenceScores(relevance=0.8, freshness=0.7, source=0.9)
)
```

### Serialization

```python
# Convert to JSON
json_str = evidence.to_json()

# Convert from JSON
reconstructed = Evidence.from_json(json_str)
```

### Score Updates

```python
# Update individual scores
evidence.update_scores(relevance=0.9, freshness=0.8)

# Final score is automatically recalculated
print(evidence.scores.final)
```

### Chunk Management

```python
# Add text chunks
evidence.add_chunk("chunk_001")
evidence.add_chunk("chunk_002")

# Remove chunks
evidence.remove_chunk("chunk_001")
```

## Testing

Run the test suite to validate the schema:

```bash
cd tests
python -m pytest test_schema.py -v
```

## Dependencies

- `xxhash`: For text hashing
- `dataclasses`: For the Evidence dataclass
- `datetime`: For temporal fields
- `json`: For serialization
- `typing`: For type hints

## Database Setup

### PostgreSQL with pgvector

```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Run the schema creation statements from database.py
```

### BigQuery

Use the schema definitions in `database.py` to create BigQuery tables with the appropriate field types and modes.

## Future Enhancements

- Support for additional source types
- Enhanced scoring algorithms
- Integration with external fact-checking APIs
- Advanced metadata schemas for different source types
- Support for multimedia evidence (images, videos, audio)
