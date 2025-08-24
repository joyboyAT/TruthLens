# TruthLens FastAPI Backend

A complete FastAPI implementation that wraps the TruthLens pipeline and exposes a `/predict` endpoint for fact-checking and misinformation detection.

## 🚀 Features

- **Complete Pipeline Integration**: All 5 phases of TruthLens pipeline
- **FastAPI Framework**: Modern, fast, and auto-documenting API
- **Pydantic Models**: Type-safe request/response validation
- **CORS Support**: Ready for frontend integration
- **Auto-generated Docs**: Interactive API documentation
- **Mock Fallback**: Works even without full TruthLens components

## 📁 Files

- `truthlens_fastapi.py` - Main FastAPI application
- `requirements_fastapi.txt` - Python dependencies
- `start_fastapi.py` - Startup script with dependency checks
- `test_fastapi.py` - Test script for API endpoints
- `frontend/index.html` - Simple frontend interface

## 🛠️ Setup

### 1. Install Dependencies

```bash
cd webapp
pip install -r requirements_fastapi.txt
```

### 2. Environment Variables (Optional)

```bash
# For web search functionality
export SERPER_API_KEY="your_serper_api_key"
# OR
export BING_API_KEY="your_bing_api_key"

# Server port (default: 8000)
export PORT=8000
```

## 🚀 Quick Start

### Option 1: Using Startup Script (Recommended)

```bash
cd webapp
python start_fastapi.py
```

### Option 2: Direct Execution

```bash
cd webapp
python truthlens_fastapi.py
```

### Option 3: Using Uvicorn Directly

```bash
cd webapp
uvicorn truthlens_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

## 📡 API Endpoints

### Root
```http
GET /
```
Returns API information and available endpoints.

### Health Check
```http
GET /health
```
Returns health status and component availability.

### Pipeline Status
```http
GET /status
```
Returns detailed pipeline component status.

### Predict (Main Endpoint)
```http
POST /predict
Content-Type: application/json

{
  "text": "COVID-19 vaccines cause autism in children.",
  "input_type": "text",
  "max_claims": 5,
  "max_evidence_per_claim": 3
}
```

## 🧪 Testing

### 1. Test with Python Script

```bash
cd webapp
python test_fastapi.py
```

### 2. Test with Curl

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Pipeline status
curl -X GET "http://localhost:8000/status"

# Predict endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "COVID-19 vaccines cause autism in children.",
    "input_type": "text",
    "max_claims": 3,
    "max_evidence_per_claim": 2
  }'
```

### 3. Test with Postman

1. Import the collection or create a new request
2. Set URL to `http://localhost:8000/predict`
3. Set method to `POST`
4. Set headers: `Content-Type: application/json`
5. Set body (raw JSON):
```json
{
  "text": "COVID-19 vaccines cause autism in children.",
  "input_type": "text",
  "max_claims": 3,
  "max_evidence_per_claim": 2
}
```

## 🌐 Frontend Integration

### 1. Open Frontend

```bash
# Open the HTML file in your browser
open webapp/frontend/index.html
# OR
start webapp/frontend/index.html  # Windows
```

### 2. Frontend Features

- **Modern UI**: Clean, responsive design
- **Real-time Analysis**: Live processing feedback
- **Comprehensive Results**: All pipeline phases displayed
- **Interactive Elements**: Hover effects and animations
- **Error Handling**: Graceful error display

## 📊 API Response Format

### Success Response
```json
{
  "input": {
    "original": "COVID-19 vaccines cause autism in children.",
    "processed": "COVID-19 vaccines cause autism in children.",
    "type": "text"
  },
  "pipeline_results": {
    "phase1": {
      "status": "completed",
      "processing_time": 0.001,
      "details": {
        "normalized_text": "COVID-19 vaccines cause autism in children."
      }
    },
    "phase2": {
      "status": "completed",
      "processing_time": 0.234,
      "details": {
        "total_claims": 1,
        "top_claims": 1,
        "claims": [
          {
            "id": "claim_1",
            "text": "COVID-19 vaccines cause autism in children",
            "subject": "COVID-19 vaccines",
            "predicate": "cause",
            "object": "autism in children",
            "checkworthiness": 0.987,
            "context": {...}
          }
        ]
      }
    },
    "phase3": {
      "status": "completed",
      "processing_time": 0.456,
      "details": {
        "total_evidence": 2,
        "evidence_sources": ["Mock Source 1", "Mock Source 2"],
        "evidence": [...]
      }
    },
    "phase4": {
      "status": "completed",
      "processing_time": 0.123,
      "details": {
        "verification_results": [
          {
            "claim_text": "COVID-19 vaccines cause autism in children",
            "confidence_score": 0.75,
            "stance": "refutes",
            "verdict": "Likely False",
            "highlights": [...]
          }
        ]
      }
    },
    "phase5": {
      "status": "completed",
      "processing_time": 0.067,
      "details": {
        "explanations": [...],
        "cue_badges": [...],
        "prebunk_cards": [...],
        "evidence_cards": [...]
      }
    }
  },
  "summary": {
    "total_processing_time": 0.881,
    "phases_completed": 5,
    "claims_processed": 1,
    "evidence_retrieved": 2,
    "verdicts": ["Likely False"],
    "overall_confidence": 0.75
  },
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

### Error Response
```json
{
  "detail": "Error message here"
}
```

## 🔧 Configuration

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Input text to analyze |
| `input_type` | string | "text" | Type of input (text, url, image) |
| `max_claims` | integer | 5 | Maximum claims to extract (1-20) |
| `max_evidence_per_claim` | integer | 3 | Maximum evidence per claim (1-10) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `SERPER_API_KEY` | - | Serper API key for web search |
| `BING_API_KEY` | - | Bing API key for web search |

## 📈 Performance

### Typical Response Times

- **Mock Mode**: 0.1-0.5s
- **Full Pipeline**: 1-5s (depending on text length and complexity)
- **Individual Phases**: 0.1-2s each

### Memory Usage

- **Startup**: ~500MB-1GB (model loading)
- **Per Request**: ~50-100MB additional

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'fastapi'
   ```
   **Solution**: Install dependencies with `pip install -r requirements_fastapi.txt`

2. **Port Already in Use**
   ```
   OSError: [Errno 98] Address already in use
   ```
   **Solution**: Change port with `export PORT=8001` or kill existing process

3. **CORS Errors (Frontend)**
   ```
   CORS policy: No 'Access-Control-Allow-Origin' header
   ```
   **Solution**: CORS is already configured for all origins in development

4. **Model Loading Failures**
   ```
   Warning: Could not import TruthLens modules
   ```
   **Solution**: API will fall back to mock mode, still functional

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python truthlens_fastapi.py
```

## 🚀 Deployment

### Development
```bash
uvicorn truthlens_fastapi:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
# Using Gunicorn
pip install gunicorn
gunicorn truthlens_fastapi:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Using Docker
docker build -t truthlens-api .
docker run -p 8000:8000 truthlens-api
```

### Docker Example
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_fastapi.txt .
RUN pip install -r requirements_fastapi.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "truthlens_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📚 API Documentation

Once the server is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🔄 Integration Examples

### JavaScript/Frontend
```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'COVID-19 vaccines cause autism in children.',
    input_type: 'text',
    max_claims: 3,
    max_evidence_per_claim: 2
  })
});

const result = await response.json();
console.log('Claims processed:', result.summary.claims_processed);
```

### Python
```python
import requests

response = requests.post('http://localhost:8000/predict', json={
    'text': 'COVID-19 vaccines cause autism in children.',
    'input_type': 'text',
    'max_claims': 3,
    'max_evidence_per_claim': 2
})

result = response.json()
print(f"Claims processed: {result['summary']['claims_processed']}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "COVID-19 vaccines cause autism in children.",
    "input_type": "text",
    "max_claims": 3,
    "max_evidence_per_claim": 2
  }' | python -m json.tool
```

## 🎉 Success!

Your TruthLens FastAPI backend is now ready! The complete pipeline is wrapped in a Python function and exposed via a `/predict` endpoint. You can:

1. ✅ Test locally with curl or Postman
2. ✅ Connect frontend fetch → backend API
3. ✅ Deploy backend + frontend separately
4. ✅ Scale and monitor in production

The API provides comprehensive fact-checking results with all 5 pipeline phases, making it perfect for integration into any application! 🚀
