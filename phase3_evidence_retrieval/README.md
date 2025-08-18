# Phase 3: Evidence Retrieval System

## Overview

The Evidence Retrieval System is the core component of TruthLens that automatically discovers, retrieves, and ranks evidence to support or refute claims. This system implements a multi-source, multi-modal approach to evidence gathering with intelligent ranking and deduplication.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Processing    │    │   Storage &     │
│                 │    │   Pipeline      │    │   Retrieval     │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Fact-check    │    │ • Text Cleaning │    │ • Vector DB     │
│ • Wikipedia     │    │ • Chunking      │    │ • BigQuery      │
│ • News APIs     │    │ • Embeddings    │    │ • Cache Layer   │
│ • Government    │    │ • NLI Labeling  │    │ • Search API    │
│ • Social Media  │    │ • Scoring       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Features

- **Multi-Source Evidence**: Integrates fact-checking sites, Wikipedia, news APIs, and government sources
- **Intelligent Chunking**: Semantic text splitting for optimal retrieval
- **Vector Search**: Fast similarity search using multilingual embeddings
- **NLI Classification**: Natural Language Inference for claim-evidence relationships
- **Smart Scoring**: Relevance, freshness, and source credibility scoring
- **Deduplication**: Advanced clustering to remove duplicate evidence
- **Caching**: TTL-based caching with intelligent refresh policies
- **API Interface**: RESTful API for evidence retrieval

## Directory Structure

```
phase3_evidence_retrieval/
│── README.md                     # This documentation
│── requirements.txt               # All dependencies (HF models, libs)
│── config/
│   ├── settings.yaml              # API keys, vector DB configs, TTL policy
│   ├── domains_whitelist.json     # Trusted domains (gov, news, fact-check)
│   └── model_paths.json           # HuggingFace model names
│
│── data/
│   ├── raw/                       # Raw HTML/news/fact-check pages
│   ├── processed/                 # Cleaned text, chunked text
│   ├── cache/                     # Cached evidence JSON
│   └── eval/claims.jsonl          # Evaluation dataset
│
│── schemas/
│   └── evidence.py                # Evidence dataclass / Pydantic model
│
│── connectors/                    # Data source connectors
│   ├── factcheck.py               # PIB, PolitiFact, BOOM scrapers
│   ├── wikipedia.py               # Wikipedia API wrapper
│   └── grounded_search.py         # News/Gov/Health search queries
│
│── retrieval/                     # Evidence retrieval & re-ranking
│   ├── vector_search.py           # ANN search in Vector DB / BigQuery
│   ├── rerank.py                  # bge-reranker integration
│   └── grounded_search.py         # (alt placement if kept under retrieval)
│
│── nlp/                           # Text handling modules
│   ├── chunking.py                # Semantic chunk splitter
│   ├── embeddings.py              # multilingual-e5 embeddings
│   └── nli_labeler.py             # NLI classifier for support/contradict
│
│── scoring/
│   └── evidence_score.py          # Relevance + freshness + source trust score
│
│── cleaning/
│   └── dedup.py                   # Duplicate removal & clustering
│
│── cache/
│   ├── policy.py                  # TTL and refresh policies
│   └── refresh.py                 # Refresh jobs (cron-style)
│
│── pipeline/
│   └── retrieve_pipeline.py       # Orchestrates the full retrieval flow
│
│── api/
│   └── retrieve.py                # FastAPI endpoints (/retrieve-evidence)
│
│── tests/                         # Unit + integration tests
│   ├── test_schema.py
│   ├── test_factcheck.py
│   ├── test_wikipedia.py
│   ├── test_grounded_search.py
│   ├── test_embeddings.py
│   ├── test_rerank.py
│   ├── test_nli.py
│   ├── test_scoring.py
│   ├── test_dedup.py
│   ├── test_cache.py
│   ├── test_api.py
│   └── test_pipeline.py
│
└── logs/
    └── retrieval.log              # Logging for debugging & monitoring
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp config/settings.yaml.example config/settings.yaml
# Edit config/settings.yaml with your API keys
```

### 2. Configuration

```yaml
# config/settings.yaml
api_keys:
  openai: "your-openai-key"
  newsapi: "your-newsapi-key"
  wikipedia: "your-wikipedia-key"

vector_db:
  host: "localhost"
  port: 5432
  database: "truthlens"
  user: "postgres"
  password: "password"

cache:
  ttl_hours: 24
  refresh_threshold: 0.8
```

### 3. Run the System

```bash
# Start the API server
python -m api.retrieve

# Run the retrieval pipeline
python -m pipeline.retrieve_pipeline

# Run tests
python -m pytest tests/ -v
```

## Core Components

### 1. Data Connectors

- **Fact-check Connectors**: Scrape PIB, PolitiFact, BOOM, and other fact-checking sites
- **Wikipedia Connector**: API wrapper for Wikipedia content retrieval
- **Grounded Search**: News, government, and health domain search

### 2. NLP Pipeline

- **Text Chunking**: Semantic splitting using sentence transformers
- **Embeddings**: Multilingual E5 embeddings for cross-language support
- **NLI Classification**: BERT-based natural language inference for claim-evidence relationships

### 3. Retrieval Engine

- **Vector Search**: Fast ANN search using pgvector or BigQuery
- **Re-ranking**: BGE-reranker for improved relevance scoring
- **Hybrid Search**: Combines vector similarity with traditional keyword search

### 4. Scoring System

- **Relevance Score**: Semantic similarity to the claim
- **Freshness Score**: Recency of the evidence
- **Source Trust Score**: Domain credibility and fact-checking reputation

### 5. Caching Layer

- **TTL Policy**: Time-based cache expiration
- **Refresh Logic**: Intelligent cache refresh based on content staleness
- **Storage**: Redis or in-memory caching with persistence

## API Endpoints

### Retrieve Evidence

```http
POST /retrieve-evidence
Content-Type: application/json

{
  "claim": "Climate change is caused by human activities",
  "max_results": 10,
  "min_relevance": 0.7,
  "sources": ["factcheck", "wikipedia", "news"],
  "language": "en"
}
```

### Response

```json
{
  "claim": "Climate change is caused by human activities",
  "evidence": [
    {
      "id": "ev_001",
      "title": "IPCC Report Confirms Human Influence on Climate",
      "url": "https://example.com/ipcc-report",
      "domain": "ipcc.ch",
      "snippet": "The latest IPCC report provides strong evidence...",
      "support_label": "supports",
      "scores": {
        "relevance": 0.95,
        "freshness": 0.8,
        "source": 0.98,
        "final": 0.93
      },
      "metadata": {
        "source_type": "research_paper",
        "published_at": "2024-01-15T10:00:00Z"
      }
    }
  ],
  "total_found": 45,
  "processing_time_ms": 1250
}
```

## Configuration Files

### Settings (config/settings.yaml)

```yaml
# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4

# Database Configuration
database:
  vector_db:
    host: "localhost"
    port: 5432
    database: "truthlens"
    user: "postgres"
    password: "password"
  
  bigquery:
    project_id: "your-project"
    dataset: "truthlens"

# Model Configuration
models:
  embeddings: "intfloat/multilingual-e5-large"
  reranker: "BAAI/bge-reranker-v2-m3"
  nli: "microsoft/DialoGPT-medium"

# Cache Configuration
cache:
  ttl_hours: 24
  refresh_threshold: 0.8
  max_size: 10000
```

### Domain Whitelist (config/domains_whitelist.json)

```json
{
  "fact_checking": [
    "pib.gov.in",
    "politifact.com",
    "boomlive.in",
    "factcheck.org"
  ],
  "news": [
    "reuters.com",
    "ap.org",
    "bbc.com",
    "cnn.com"
  ],
  "government": [
    "gov.in",
    "whitehouse.gov",
    "parliament.uk",
    "europa.eu"
  ],
  "academic": [
    "nature.com",
    "science.org",
    "arxiv.org",
    "scholar.google.com"
  ]
}
```

### Model Paths (config/model_paths.json)

```json
{
  "embeddings": {
    "name": "intfloat/multilingual-e5-large",
    "max_length": 512,
    "device": "cuda"
  },
  "reranker": {
    "name": "BAAI/bge-reranker-v2-m3",
    "max_length": 512,
    "device": "cuda"
  },
  "nli": {
    "name": "microsoft/DialoGPT-medium",
    "max_length": 256,
    "device": "cuda"
  },
  "chunking": {
    "name": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 512,
    "overlap": 50
  }
}
```

## Testing

### Run All Tests

```bash
python -m pytest tests/ -v --cov=. --cov-report=html
```

### Run Specific Test Categories

```bash
# Test data connectors
python -m pytest tests/test_factcheck.py tests/test_wikipedia.py -v

# Test NLP components
python -m pytest tests/test_embeddings.py tests/test_nli.py -v

# Test retrieval system
python -m pytest tests/test_vector_search.py tests/test_rerank.py -v

# Test API endpoints
python -m pytest tests/test_api.py -v
```

## Monitoring & Logging

### Log Files

- **retrieval.log**: Main application logs
- **api.log**: API endpoint access logs
- **pipeline.log**: Pipeline execution logs
- **error.log**: Error and exception logs

### Metrics

- **Retrieval Latency**: Average response time for evidence retrieval
- **Cache Hit Rate**: Percentage of requests served from cache
- **Source Coverage**: Distribution of evidence across different sources
- **Quality Scores**: Average relevance, freshness, and trust scores

## Performance Optimization

### 1. Vector Search

- Use HNSW or IVFFlat indexes for fast similarity search
- Implement approximate nearest neighbor search
- Batch embedding generation for multiple texts

### 2. Caching Strategy

- Cache frequently requested claims
- Implement LRU eviction policy
- Use Redis for distributed caching

### 3. Parallel Processing

- Async/await for I/O operations
- Multiprocessing for CPU-intensive tasks
- Connection pooling for database operations

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch sizes or use gradient checkpointing
2. **API Rate Limits**: Implement exponential backoff and rate limiting
3. **Database Connection**: Check connection pool settings and timeouts
4. **Model Loading**: Ensure sufficient GPU memory for large models

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python -m api.retrieve

# Check system resources
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Contributing

1. Follow the existing code structure and naming conventions
2. Add comprehensive tests for new functionality
3. Update documentation for any API changes
4. Ensure all tests pass before submitting PRs

## License

This project is part of TruthLens and follows the same licensing terms.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the test files for usage examples
- Open an issue with detailed error information

## Grounded Search (News/Gov/Health + Recency)

Configure one of the following API keys in your environment:
- `SERPER_API_KEY` (preferred) or `GOOGLE_SERPER_API_KEY`
- `BING_API_KEY` (or `AZURE_BING_SEARCH_KEY`)
- `SERPAPI_API_KEY`

Run tests (will skip if no key is set):
```bash
python -m pytest tests/test_grounded_search.py -v
```

Programmatic usage:
```python
from retrieval.grounded_search import make_default_client, GroundedSearcher

client = make_default_client()
if client is None:
    raise RuntimeError("No search API key configured")
searcher = GroundedSearcher(client)
evidence = searcher.search("WHO guidance on COVID-19 boosters", top_k=5, days=90)
for ev in evidence:
    print(ev.url, ev.published_at, ev.snippet[:120])
```
