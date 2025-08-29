# TruthLens FastAPI App

A FastAPI wrapper for the enhanced TruthLens fact-checking pipeline that provides REST API endpoints for claim verification.

## 🚀 Features




- **POST /verify**: Basic claim verification using News API and Guardian API
- **POST /verify-advanced**: Full TruthLens pipeline integration
- **GET /health**: Health check endpoint
- **GET /sources**: List available news sources
- **GET /**: API information and status

## 📋 API Models

### Input
```json
{
  "claim": "COVID-19 vaccines cause autism"
}
```

### Output
```json
{
  "claim": "COVID-19 vaccines cause autism",
  "sources_checked": ["News API", "Guardian API"],
  "verification_badge": "✅ Verified by multiple sources",
  "evidence_strength": "Strong",
  "details": [
    {
      "title": "Article title",
      "url": "https://example.com/article",
      "source": "Source name",
      "source_name": "NewsAPI",
      "similarity_score": 0.85,
      "published_at": "2024-01-01",
      "relevance_score": 0.9
    }
  ],
  "processing_time": 2.5,
  "timestamp": "2024-01-01T12:00:00",
  "total_articles": 15,
  "source_breakdown": {
    "NewsAPI": 10,
    "Guardian": 5
  }
}
```

## 🔧 Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify TruthLens components**:
   Make sure all TruthLens source files are in the `src/` directory.

## 🚀 Running the App

### Start the FastAPI server:
```bash
uvicorn app:app --reload
```

The app will be available at: http://localhost:8000

### Interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Testing

### Test with curl:
```bash
curl -X POST "http://localhost:8000/verify" \
     -H "Content-Type: application/json" \
     -d '{"claim": "COVID-19 vaccines cause autism"}'
```

### Test with Python script:
```bash
python test_fastapi.py
```

## 📊 API Endpoints

### 1. POST /verify
Basic claim verification using multiple news sources.

**Logic**:
1. Query News API first
2. If insufficient results (< 3 articles), fallback to Guardian API
3. Calculate similarity scores between claim and article titles
4. Determine evidence strength and verification badge
5. Return structured verification results

**Evidence Strength Logic**:
- 0 articles → "No evidence"
- 1 article → "Weak"
- 2 articles → "Moderate"
- 3+ articles → "Strong" (if avg similarity ≥ 0.7)

### 2. POST /verify-advanced
Full TruthLens pipeline integration for comprehensive analysis.

**Features**:
- Enhanced stance classification
- Semantic search and ranking
- Google Fact Check API integration
- Cross-reference scoring
- Detailed evidence analysis

### 3. GET /health
Health check endpoint showing component status.

### 4. GET /sources
List available news sources and their status.

### 5. GET /
API information and available endpoints.

## 🔑 API Keys

The app uses these API keys (configured in `app.py`):

- **News API**: `9c7e59e19af34bb8adb97d0a8bec458d`
- **Guardian API**: `0b8f5a8d-d3a0-49e1-8472-f943dae59338`
- **Google Fact Check**: `AIzaSyACLtCwVw1dJeBNPKmbJ8Yfqu5D4zUK5Sc`

## 🏗️ Architecture

```
FastAPI App (app.py)
├── News API Handler
├── Guardian API Handler (fallback)
├── Semantic Cross-Reference Scorer
├── Enhanced TruthLens Pipeline
└── Response Formatter
```

## 📈 Performance

- **Basic verification**: ~2-5 seconds
- **Advanced verification**: ~5-10 seconds
- **News API rate limit**: 100 requests/day (free tier)
- **Guardian API**: Unlimited (with API key)

## 🚨 Error Handling

- **500**: Internal server error during verification
- **503**: Advanced verification not available
- **422**: Invalid input validation
- **429**: News API rate limit (handled gracefully with Guardian fallback)

## 🔍 Similarity Calculation

The app calculates similarity between claims and article titles using:

1. **Semantic Cross-Reference Scorer** (if available)
2. **Fallback to Jaccard similarity** (word overlap)
3. **Semantic boost** for better accuracy

## 📝 Example Usage

### Python requests:
```python
import requests

response = requests.post(
    "http://localhost:8000/verify",
    json={"claim": "COVID-19 vaccines cause autism"}
)

if response.status_code == 200:
    result = response.json()
    print(f"Evidence strength: {result['evidence_strength']}")
    print(f"Verification badge: {result['verification_badge']}")
    print(f"Total articles: {result['total_articles']}")
```

### JavaScript fetch:
```javascript
const response = await fetch('http://localhost:8000/verify', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        claim: "COVID-19 vaccines cause autism"
    })
});

const result = await response.json();
console.log(`Evidence strength: ${result.evidence_strength}`);
```

## 🎯 Next Steps

1. **Deploy to production** with proper environment variables
2. **Add authentication** for API access
3. **Implement rate limiting** for API endpoints
4. **Add caching** for repeated claims
5. **Monitor performance** and optimize similarity calculation

## 📚 Dependencies

- **FastAPI**: Modern web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **TruthLens**: Enhanced fact-checking pipeline
- **Sentence-BERT**: Semantic similarity
- **PyTorch**: Machine learning framework

## 🆘 Troubleshooting

### Common Issues:

1. **Import errors**: Ensure all TruthLens components are in `src/` directory
2. **API key errors**: Verify API keys are valid and have sufficient quota
3. **Memory issues**: Reduce `max_articles` parameter for large claims
4. **Rate limiting**: News API has daily limits, Guardian API provides fallback

### Debug Mode:
Set logging level to DEBUG in `app.py` for detailed error information.
