# TruthLens API Deployment Guide

## Overview
TruthLens is a comprehensive fact-checking API that uses advanced NLP and machine learning to verify claims against multiple news sources and fact-checking databases.

## Features
- ✅ Enhanced stance classification with improved thresholds (0.6)
- ✅ Rule-based signals for contradictions and support
- ✅ Scientific consensus handling
- ✅ Multiple news sources (News API, Guardian API, Currents API)
- ✅ Google Fact Check API integration
- ✅ Semantic search and ranking
- ✅ Comprehensive verdict aggregation

## Quick Start

### 1. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app_enhanced.py
```

### 2. Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t truthlens-api .
docker run -p 8000:8000 truthlens-api
```

### 3. Environment Variables
Copy `env.example` to `.env` and configure your API keys:
```bash
cp env.example .env
# Edit .env with your API keys
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Verify Claim
```bash
POST /verify
Content-Type: application/json

{
  "claim": "The Earth is flat",
  "context": "A claim made in social media"
}
```

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

## Deployment Options

### 1. Docker (Recommended)
```bash
# Production deployment
docker-compose -f docker-compose.yml up -d

# With custom environment
docker-compose -f docker-compose.yml --env-file .env up -d
```

### 2. Cloud Platforms

#### Heroku
```bash
# Create Procfile
echo "web: uvicorn app_enhanced:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
heroku create your-app-name
heroku config:set NEWS_API_KEY=your_key
heroku config:set GUARDIAN_API_KEY=your_key
heroku config:set GOOGLE_API_KEY=your_key
git push heroku main
```

#### AWS ECS
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
docker build -t truthlens-api .
docker tag truthlens-api:latest your-account.dkr.ecr.us-east-1.amazonaws.com/truthlens-api:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/truthlens-api:latest
```

#### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/your-project/truthlens-api
gcloud run deploy truthlens-api --image gcr.io/your-project/truthlens-api --platform managed
```

### 3. Kubernetes
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/
```

## Configuration

### API Keys Required
- **News API**: Get from [newsapi.org](https://newsapi.org)
- **Guardian API**: Get from [open-platform.theguardian.com](https://open-platform.theguardian.com)
- **Google Fact Check API**: Get from [Google Cloud Console](https://console.cloud.google.com)

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `NEWS_API_KEY` | News API key | Required |
| `GUARDIAN_API_KEY` | Guardian API key | Required |
| `GOOGLE_API_KEY` | Google Fact Check API key | Required |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `DEBUG` | Debug mode | `false` |

## Performance Optimization

### Resource Requirements
- **CPU**: 2+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Optional but recommended for faster inference
- **Storage**: 2GB for models and cache

### Caching
The API includes built-in caching for:
- News API responses
- Semantic search results
- Fact-check results

### Rate Limiting
Configure rate limits in your reverse proxy or load balancer:
- 60 requests per minute per IP
- 1000 requests per hour per IP

## Monitoring

### Health Checks
```bash
# Check API health
curl http://your-api-url/health

# Check component status
curl http://your-api-url/components-status
```

### Logging
The API logs to stdout/stderr. Configure log aggregation for production:
```bash
# Docker logging
docker logs truthlens-api

# Kubernetes logging
kubectl logs deployment/truthlens-api
```

## Security

### API Key Management
- Store API keys in environment variables
- Use secrets management in production
- Rotate keys regularly

### CORS Configuration
Configure CORS for your domain:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure sufficient RAM (4GB+)
   - Check CUDA availability for GPU acceleration

2. **API Key Errors**
   - Verify all API keys are set correctly
   - Check API key quotas and limits

3. **Memory Issues**
   - Increase container memory limits
   - Enable model caching

4. **Slow Response Times**
   - Enable GPU acceleration
   - Optimize batch sizes
   - Use caching

### Debug Mode
Enable debug mode for detailed logging:
```bash
export DEBUG=true
python app_enhanced.py
```

## Support

For issues and questions:
1. Check the logs for error messages
2. Verify API key configuration
3. Test with the health check endpoint
4. Review the API documentation at `/docs`

## License
This project is licensed under the MIT License.
