# TruthLens

A comprehensive fact-checking and misinformation detection system that processes multilingual and multimedia content.

## Project Structure

```
TruthLens/
├── src/                          # Core source code
│   ├── ingestion/               # Input detection and processing
│   ├── ocr/                     # Optical Character Recognition
│   ├── asr/                     # Automatic Speech Recognition
│   ├── translation/             # Multilingual translation
│   ├── verification/            # Evidence verification
│   ├── evidence_retrieval/      # Evidence search and retrieval
│   ├── output_ux/              # User interface components
│   ├── data_collection.py      # Web scraping and data fetching
│   └── preprocessing.py        # Text cleaning and normalization
├── extractor/                   # Claim analysis pipeline
├── tests/                      # Test suites
├── data/                       # Raw and processed data
│   ├── raw/                   # Raw input data
│   └── processed/             # Processed data
├── notebooks/                  # Jupyter notebooks
├── config/                     # Configuration files
├── database_schemas/           # Database schema definitions
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Core Modules

### 1. Ingestion (`src/ingestion/`)
- Input type detection (URL, text, image, video, audio)
- Content validation and preprocessing

### 2. OCR (`src/ocr/`)
- Text extraction from images and videos
- Multilingual OCR support

### 3. ASR (`src/asr/`)
- Speech recognition from audio and video
- Multilingual speech transcription

### 4. Translation (`src/translation/`)
- Multilingual text translation
- Language detection and normalization

### 5. Verification (`src/verification/`)
- Evidence verification and stance classification
- Confidence calibration

### 6. Evidence Retrieval (`src/evidence_retrieval/`)
- Evidence search and retrieval
- Multiple search engine support

### 7. Output UX (`src/output_ux/`)
- User interface components
- Result formatting and presentation

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Run tests:**
   ```bash
   python -m pytest tests/
   ```

## Usage

```python
from src import data_collection, preprocessing
from src.ingestion import detect_input_type
from src.ocr import extract_text_from_image
from src.asr import transcribe_audio
from src.translation import translate_text

# Example usage
input_type = detect_input_type("https://example.com/article")
text = extract_text_from_image("image.jpg")
transcript = transcribe_audio("audio.wav")
translated = translate_text("नमस्ते", target_lang="en")
```

## Features

- **Multilingual Support**: Hindi, Tamil, Telugu, Marathi, and more
- **Multimedia Processing**: Images, videos, audio, text
- **Evidence-Based Verification**: Automated fact-checking pipeline
- **User-Friendly Output**: Clean, informative result presentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
