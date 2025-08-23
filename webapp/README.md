# TruthLens Web Application

A modern, responsive web interface for the TruthLens fact-checking and misinformation detection system.

## 🚀 Quick Start

### Option 1: Enhanced Version (Recommended)
```bash
cd webapp
python start-enhanced.py
```

This will automatically:
- Install all dependencies
- Start the enhanced backend server (Flask) with better news analysis
- Start the frontend server (Next.js)
- Open the application in your browser

### Option 2: Basic Version
```bash
cd webapp
python start-app.py
```

This will start the basic version with simple mock data.

### Option 3: Manual Startup

#### 1. Install Dependencies
```bash
cd webapp

# Install frontend dependencies
npm install

# Install backend dependencies
pip install -r backend-requirements.txt
```

#### 2. Start Enhanced Backend Server
```bash
python enhanced-backend.py
```
Enhanced backend will be available at: http://localhost:8000

#### 3. Start Frontend Server (in a new terminal)
```bash
npm run dev
```
Frontend will be available at: http://localhost:3000

## Features

- **Modern UI/UX**: Built with Next.js 15, React, and Tailwind CSS
- **Multi-Modal Input**: Support for text, URL, and file uploads
- **Real-time Fact-Checking**: Interactive fact-checking interface
- **Evidence Display**: Clear presentation of supporting evidence
- **Manipulation Detection**: Identification of misinformation tactics
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Mock Backend**: Includes a working Flask backend for testing

## Technology Stack

### Frontend
- **Framework**: Next.js 15, React 18, TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **HTTP Client**: Axios
- **UI Components**: Headless UI, Heroicons

### Backend
- **Framework**: Flask
- **CORS**: Flask-CORS
- **HTTP Client**: Requests

## Project Structure

```
webapp/
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   └── fact-check/
│   │   │       └── route.ts          # Next.js API endpoint
│   │   ├── globals.css               # Global styles
│   │   ├── layout.tsx                # Root layout
│   │   └── page.tsx                  # Home page
│   ├── components/
│   │   ├── FactChecker.tsx           # Main fact-checking component
│   │   ├── Header.tsx                # Navigation header
│   │   └── Footer.tsx                # Footer component
│   └── lib/
│       └── truthlens-api.ts          # API integration library
├── backend-server.py                 # Flask backend server
├── backend-requirements.txt          # Backend dependencies
├── start-app.py                      # Automated startup script
├── package.json                      # Frontend dependencies
├── tailwind.config.js               # Tailwind CSS configuration
├── tsconfig.json                    # TypeScript configuration
└── README.md                        # This file
```

## Usage

### Fact-Checking Interface

1. **Select Input Type:**
   - Text Input: Paste or type text to fact-check
   - URL Analysis: Enter a URL to analyze
   - File Upload: Upload images, videos, or audio files

2. **Submit for Analysis:**
   - Click "Fact Check" to start the analysis
   - The system will process your input and return results

3. **Review Results:**
   - **Overall Verdict**: True, False, Misleading, or Unverified
   - **Detected Claims**: Individual claims extracted from the content
   - **Supporting Evidence**: Sources that support or contradict claims
   - **Manipulation Detection**: Identified misinformation tactics

### Testing Examples

The enhanced backend includes better analysis for real news articles. Try these examples:

#### Political/Diplomatic News
```
PM Modi speaks with French President Macron, discusses ending conflicts in Ukraine, West Asia
```

#### COVID-19 Vaccine Misinformation
```
COVID-19 vaccines cause autism in children
```

#### 5G Conspiracy Theory
```
5G technology causes health problems
```

#### Climate Change Denial
```
Climate change is a hoax
```

#### Government Announcements
```
Prime Minister announces new economic policy
```

### API Endpoints

#### Backend API (Flask - http://localhost:8000)

- `GET /api/v1/health` - Health check
- `POST /api/v1/fact-check` - Main fact-checking endpoint
- `POST /api/v1/extract-claims` - Extract claims from text
- `POST /api/v1/search-evidence` - Search for evidence
- `GET /api/v1/languages` - Get supported languages
- `GET /api/v1/trusted-sources` - Get trusted sources list

#### Frontend API (Next.js - http://localhost:3000)

- `POST /api/fact-check` - Frontend API endpoint (proxies to backend)

## Development

### Available Scripts

#### Frontend (Next.js)
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

#### Backend (Flask)
- `python enhanced-backend.py` - Start enhanced backend server (recommended)
- `python simple-backend.py` - Start basic backend server
- `python backend-server.py` - Start original backend server

### Customization

#### Styling
The app uses Tailwind CSS for styling. Modify `tailwind.config.js` to customize the design system.

#### Components
All components are located in `src/components/` and can be easily modified or extended.

#### Backend Integration
To integrate with the actual TruthLens backend:

1. **Update API Configuration**: Modify `src/lib/truthlens-api.ts`
2. **Replace Mock Data**: Update `backend-server.py` to call actual TruthLens modules
3. **Environment Variables**: Set `NEXT_PUBLIC_TRUTHLENS_API_URL` for production

### Environment Variables

Create a `.env.local` file in the webapp directory:

```env
# Backend API URL (for production)
NEXT_PUBLIC_TRUTHLENS_API_URL=http://localhost:8000

# API Keys (if needed)
TRUTHLENS_API_KEY=your_api_key_here
```

## API Integration

### Example API Request
```javascript
const response = await fetch('/api/fact-check', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    input_type: 'text',
    content: 'Your text to fact-check'
  })
});
```

### Example API Response
```json
{
  "claims": [
    {
      "id": "claim_1",
      "text": "Extracted claim text",
      "subject": "Subject",
      "predicate": "Predicate", 
      "object": "Object",
      "checkworthiness": 0.85,
      "context": {
        "negation": false,
        "modality": "assertive",
        "conditional_trigger": "",
        "sarcasm_score": 0.1,
        "attribution": ""
      }
    }
  ],
  "evidence": [
    {
      "id": "evidence_1",
      "title": "Evidence title",
      "content": "Evidence content",
      "url": "https://example.com",
      "source_type": "government",
      "relevance_score": 0.95,
      "stance": "contradicts",
      "confidence": 0.92
    }
  ],
  "manipulation_detected": true,
  "manipulation_types": ["False causation", "Misleading claims"],
  "overall_verdict": "False",
  "confidence": 0.91,
  "processing_time": 2.0,
  "metadata": {
    "model_version": "1.0.0",
    "timestamp": "2024-01-01T00:00:00Z",
    "input_length": 100
  }
}
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   - Frontend (3000): Kill process using port 3000
   - Backend (8000): Kill process using port 8000

2. **Dependencies Not Found**
   - Run `npm install` for frontend
   - Run `pip install -r backend-requirements.txt` for backend

3. **CORS Errors**
   - Ensure backend server is running on http://localhost:8000
   - Check that Flask-CORS is properly configured

4. **Module Import Errors**
   - Ensure you're running from the webapp directory
   - Check Python path includes parent directory for TruthLens modules

### Debug Mode

#### Frontend
```bash
npm run dev
# Check browser console for errors
```

#### Backend
```bash
python backend-server.py
# Check terminal output for errors
```

## Deployment

### Frontend Deployment

#### Vercel (Recommended)
1. Push your code to GitHub
2. Connect your repository to Vercel
3. Deploy automatically

#### Other Platforms
- Netlify
- AWS Amplify
- Railway
- DigitalOcean App Platform

### Backend Deployment

#### Heroku
1. Create `Procfile`:
   ```
   web: python backend-server.py
   ```
2. Deploy to Heroku

#### Other Platforms
- AWS Elastic Beanstalk
- Google Cloud Run
- DigitalOcean App Platform
- Railway

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the TruthLens system and follows the same license terms.

## Support

For support and questions:
- Check the main TruthLens documentation
- Open an issue in the repository
- Contact the development team

## Screenshots

The web application features:
- Clean, modern interface
- Responsive design
- Interactive fact-checking
- Evidence visualization
- Manipulation detection alerts
