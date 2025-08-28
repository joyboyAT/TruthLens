# TruthLens Backend with Fine-tuned RoBERTa-base Model

This backend integrates the fine-tuned RoBERTa-base model for improved claim detection accuracy in the TruthLens webapp.

## 🚀 Features

- **Fine-tuned Claim Detection**: Uses the 91.94% F1-score RoBERTa-base model
- **Fallback Support**: Gracefully falls back to base model if fine-tuned model unavailable
- **Evidence Retrieval**: Vector search and web search integration
- **RESTful API**: Clean endpoints for text analysis
- **Real-time Processing**: Fast claim extraction and evidence retrieval

## 📁 Files

- `truthlens-backend-with-fine-tuned.py` - Main backend server
- `start-fine-tuned-backend.py` - Startup script with dependency checks
- `test-fine-tuned-backend.py` - Test script to verify integration

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install flask flask-cors torch transformers sentence-transformers numpy scikit-learn
```

### 2. Environment Variables (Optional)

```bash
# For web search functionality
export SERPER_API_KEY="your_serper_api_key"
# OR
export BING_API_KEY="your_bing_api_key"

# Server port (default: 5000)
export PORT=5000
```

### 3. Verify Model Files

Ensure your fine-tuned model is in:
```
models/claim-detection-roberta-base/
```

## 🚀 Quick Start

### Option 1: Using Startup Script (Recommended)

```bash
cd webapp
python start-fine-tuned-backend.py
```

### Option 2: Direct Execution

```bash
cd webapp
python truthlens-backend-with-fine-tuned.py
```

## 📡 API Endpoints

### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "truthlens_available": true,
  "fine_tuned_available": true,
  "vector_retriever": true,
  "search_client": false
}
```

### Model Status
```http
GET /api/model-status
```

**Response:**
```json
{
  "truthlens_available": true,
  "fine_tuned_available": true,
  "fine_tuned_model": {
    "name": "RoBERTa-base (Fine-tuned)",
    "dataset": "Nithiwat/claim-detection",
    "accuracy": "91.94% F1 Score",
    "status": "loaded"
  },
  "vector_retriever": {
    "status": "loaded",
    "model": "all-MiniLM-L6-v2"
  },
  "capabilities": {
    "claim_detection": true,
    "evidence_retrieval": true,
    "fine_tuned_detection": true
  }
}
```

### Extract Claims
```http
POST /api/claims
Content-Type: application/json

{
  "text": "COVID-19 vaccines cause autism in children."
}
```

**Response:**
```json
{
  "input_text": "COVID-19 vaccines cause autism in children.",
  "claims": [
    {
      "id": "uuid",
      "text": "COVID-19 vaccines cause autism in children",
      "subject": "COVID-19 vaccines",
      "predicate": "cause",
      "object": "autism in children",
      "checkworthiness": 0.987,
      "context": {...},
      "detection_method": "fine_tuned"
    }
  ],
  "total_claims": 1,
  "processing_time": 0.234,
  "model_info": {
    "truthlens_available": true,
    "fine_tuned_available": true,
    "detection_method": "fine_tuned"
  }
}
```

### Search Evidence
```http
POST /api/evidence
Content-Type: application/json

{
  "query": "COVID-19 vaccines autism",
  "max_results": 5
}
```

**Response:**
```json
{
  "query": "COVID-19 vaccines autism",
  "evidence": [
    {
      "id": "vector_1",
      "title": "Fact-check: COVID-19 vaccines and autism",
      "content": "Multiple studies have found no link...",
      "url": "https://example.com/fact-check",
      "source_type": "fact_check",
      "relevance_score": 0.95,
      "stance": "refutes",
      "confidence": 0.9
    }
  ],
  "total_results": 1,
  "processing_time": 0.123
}
```

### Full Analysis
```http
POST /api/analyze
Content-Type: application/json

{
  "text": "COVID-19 vaccines cause autism in children. 5G towers are dangerous."
}
```

**Response:**
```json
{
  "input_text": "COVID-19 vaccines cause autism in children. 5G towers are dangerous.",
  "claims": [
    {
      "id": "uuid1",
      "text": "COVID-19 vaccines cause autism in children",
      "checkworthiness": 0.987,
      "evidence": [...]
    },
    {
      "id": "uuid2", 
      "text": "5G towers are dangerous",
      "checkworthiness": 0.923,
      "evidence": [...]
    }
  ],
  "total_claims": 2,
  "total_evidence": 6,
  "processing_time": 0.456,
  "model_info": {
    "truthlens_available": true,
    "fine_tuned_available": true,
    "detection_method": "fine_tuned"
  }
}
```

## 🧪 Testing

Run the test script to verify everything works:

```bash
cd webapp
python test-fine-tuned-backend.py
```

This will test:
- ✅ Health check
- ✅ Model status
- ✅ Claim extraction
- ✅ Evidence search
- ✅ Full analysis

## 🔧 Configuration

### Model Settings

The backend automatically detects and uses:
1. **Fine-tuned model** (if available) - Highest accuracy
2. **Base model** (fallback) - Standard TruthLens model

### Performance Tuning

- **Batch Processing**: Process multiple sentences efficiently
- **Caching**: Vector embeddings cached for faster retrieval
- **Async Support**: Non-blocking evidence retrieval

## 🐛 Troubleshooting

### Model Not Loading
```
⚠️ Fine-tuned model not found at: models/claim-detection-roberta-base
```
**Solution**: Ensure the fine-tuned model files are in the correct location.

### Import Errors
```
⚠️ Warning: Could not import fine-tuned model
```
**Solution**: Check that `integrate_fine_tuned_roberta_base.py` exists in the project root.

### Memory Issues
```
CUDA out of memory
```
**Solution**: The backend automatically uses CPU if GPU memory is insufficient.

### API Key Issues
```
⚠️ No search API keys found
```
**Solution**: Set `SERPER_API_KEY` or `BING_API_KEY` environment variables for web search.

## 📊 Performance

### Typical Response Times
- **Claim Detection**: 0.1-0.3s per sentence
- **Evidence Retrieval**: 0.5-2.0s per query
- **Full Analysis**: 1-5s for typical text

### Accuracy Improvements
- **Fine-tuned Model**: 91.94% F1-score vs ~85% base model
- **Better False Positive Reduction**: More precise claim detection
- **Improved Confidence Scoring**: More reliable confidence values

## 🔄 Integration with Frontend

The backend is designed to work seamlessly with the TruthLens frontend:

1. **Same API Endpoints**: Compatible with existing frontend
2. **Enhanced Responses**: Additional model information
3. **Better Accuracy**: Improved claim detection results
4. **Fallback Support**: Works even if fine-tuned model unavailable

## 📈 Monitoring

Monitor the backend with:
- **Health Check**: `/api/health`
- **Model Status**: `/api/model-status`
- **Logs**: Check console output for detailed logging

## 🚀 Production Deployment

For production use:

1. **Environment Variables**: Set all required API keys
2. **Model Files**: Ensure fine-tuned model is accessible
3. **Dependencies**: Install all required packages
4. **Monitoring**: Set up health checks and logging
5. **Scaling**: Consider using WSGI server (gunicorn) for multiple workers

## 📝 Example Usage

```python
import requests

# Initialize connection
base_url = "http://localhost:5000"

# Check health
response = requests.get(f"{base_url}/api/health")
print(response.json())

# Analyze text
text = "COVID-19 vaccines cause autism in children."
response = requests.post(
    f"{base_url}/api/analyze",
    json={"text": text}
)
result = response.json()

print(f"Claims found: {result['total_claims']}")
print(f"Evidence retrieved: {result['total_evidence']}")
print(f"Processing time: {result['processing_time']:.3f}s")
```

This backend provides a production-ready integration of your fine-tuned model with the TruthLens webapp! 🎉
