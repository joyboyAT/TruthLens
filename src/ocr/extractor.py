"""
OCR text extraction using EasyOCR for multilingual support.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import EasyOCR, fallback to Tesseract if not available
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available, falling back to Tesseract")

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available")

# Initialize EasyOCR reader (lazy loading)
_reader = None


def _get_easyocr_reader():
    """Get or create EasyOCR reader with multilingual support."""
    global _reader
    if _reader is None and EASYOCR_AVAILABLE:
        # Support for Indic languages and English
        languages = ['en', 'hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'ur']
        _reader = easyocr.Reader(languages, gpu=False)
    return _reader


def extract_text_from_image(image_path: Union[str, Path]) -> str:
    """
    Extract text from image file.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Extracted text as string
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Try EasyOCR first
    if EASYOCR_AVAILABLE:
        try:
            return _extract_with_easyocr(image_path)
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
    
    # Fallback to Tesseract
    if TESSERACT_AVAILABLE:
        try:
            return _extract_with_tesseract(image_path)
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
    
    # Final fallback
    raise RuntimeError("No OCR engine available")


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """
    Extract text from image bytes.
    
    Args:
        image_bytes: Image data as bytes
    
    Returns:
        Extracted text as string
    """
    # Try EasyOCR first
    if EASYOCR_AVAILABLE:
        try:
            return _extract_with_easyocr_bytes(image_bytes)
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
    
    # Fallback to Tesseract
    if TESSERACT_AVAILABLE:
        try:
            return _extract_with_tesseract_bytes(image_bytes)
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
    
    # Final fallback
    raise RuntimeError("No OCR engine available")


def _extract_with_easyocr(image_path: Path) -> str:
    """Extract text using EasyOCR."""
    reader = _get_easyocr_reader()
    if reader is None:
        raise RuntimeError("EasyOCR reader not available")
    
    # Read image and extract text
    results = reader.readtext(str(image_path))
    
    # Extract text from results
    texts = []
    for (bbox, text, confidence) in results:
        if confidence > 0.5:  # Filter low confidence results
            texts.append(text.strip())
    
    return ' '.join(texts)


def _extract_with_easyocr_bytes(image_bytes: bytes) -> str:
    """Extract text using EasyOCR from bytes."""
    reader = _get_easyocr_reader()
    if reader is None:
        raise RuntimeError("EasyOCR reader not available")
    
    # Extract text from bytes
    results = reader.readtext(image_bytes)
    
    # Extract text from results
    texts = []
    for (bbox, text, confidence) in results:
        if confidence > 0.5:  # Filter low confidence results
            texts.append(text.strip())
    
    return ' '.join(texts)


def _extract_with_tesseract(image_path: Path) -> str:
    """Extract text using Tesseract OCR."""
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("Tesseract not available")
    
    # Configure Tesseract for multilingual support
    config = '--oem 3 --psm 6 -l eng+hin+tel+tam+ben+mar+guj+kan+mal+pan+urd'
    
    # Extract text
    text = pytesseract.image_to_string(
        Image.open(image_path),
        config=config
    )
    
    return _clean_ocr_text(text)


def _extract_with_tesseract_bytes(image_bytes: bytes) -> str:
    """Extract text using Tesseract OCR from bytes."""
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("Tesseract not available")
    
    # Configure Tesseract for multilingual support
    config = '--oem 3 --psm 6 -l eng+hin+tel+tam+ben+mar+guj+kan+mal+pan+urd'
    
    # Open image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Extract text
    text = pytesseract.image_to_string(image, config=config)
    
    return _clean_ocr_text(text)


def _clean_ocr_text(text: str) -> str:
    """Clean and normalize OCR text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common OCR artifacts
    text = re.sub(r'[|\\/\[\]{}()]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text
