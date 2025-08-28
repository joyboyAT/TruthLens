"""
Text translation using Google Translate API for Indic languages.
"""

import re
import os
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import Google Translate
try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    logger.warning("Google Translate not available")

# Initialize translator (lazy loading)
_translator = None


def _get_translator():
    """Get or create Google Translate instance."""
    global _translator
    if _translator is None and GOOGLETRANS_AVAILABLE:
        _translator = Translator()
    return _translator


def translate_text(text: str, source_lang: Optional[str] = None, target_lang: str = "en") -> str:
    """
    Translate text to target language.
    
    Args:
        text: Text to translate
        source_lang: Source language code (auto-detect if None)
        target_lang: Target language code (default: 'en')
    
    Returns:
        Translated text
    """
    if not text.strip():
        return text
    
    if not GOOGLETRANS_AVAILABLE:
        logger.warning("Translation not available, returning original text")
        return text
    
    try:
        translator = _get_translator()
        if translator is None:
            return text
        
        # Detect source language if not provided
        if source_lang is None:
            detected = translator.detect(text)
            source_lang = detected.lang
        
        # Don't translate if already in target language
        if source_lang == target_lang:
            return text
        
        # Translate text
        result = translator.translate(text, src=source_lang, dest=target_lang)
        return result.text
    
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text  # Return original text on error


def detect_language(text: str) -> str:
    """
    Detect the language of text.
    
    Args:
        text: Text to analyze
    
    Returns:
        Language code (e.g., 'en', 'hi', 'ta')
    """
    if not text.strip():
        return "en"
    
    if not GOOGLETRANS_AVAILABLE:
        return _simple_language_detection(text)
    
    try:
        translator = _get_translator()
        if translator is None:
            return _simple_language_detection(text)
        
        detected = translator.detect(text)
        return detected.lang
    
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return _simple_language_detection(text)


def is_indic_language(text: str) -> bool:
    """
    Check if text is in an Indic language.
    
    Args:
        text: Text to check
    
    Returns:
        True if text is in an Indic language
    """
    lang = detect_language(text)
    indic_languages = ['hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'ur']
    return lang in indic_languages


def _simple_language_detection(text: str) -> str:
    """Simple language detection based on character sets."""
    text = text.lower()
    
    # Hindi/Devanagari
    if re.search(r'[\u0900-\u097F]', text):
        return "hi"
    
    # Tamil
    if re.search(r'[\u0B80-\u0BFF]', text):
        return "ta"
    
    # Telugu
    if re.search(r'[\u0C00-\u0C7F]', text):
        return "te"
    
    # Bengali
    if re.search(r'[\u0980-\u09FF]', text):
        return "bn"
    
    # Marathi (uses Devanagari)
    if re.search(r'[\u0900-\u097F]', text):
        return "mr"
    
    # Gujarati
    if re.search(r'[\u0A80-\u0AFF]', text):
        return "gu"
    
    # Kannada
    if re.search(r'[\u0C80-\u0CFF]', text):
        return "kn"
    
    # Malayalam
    if re.search(r'[\u0D00-\u0D7F]', text):
        return "ml"
    
    # Punjabi
    if re.search(r'[\u0A00-\u0A7F]', text):
        return "pa"
    
    # Urdu
    if re.search(r'[\u0600-\u06FF]', text):
        return "ur"
    
    # Default to English
    return "en"


def normalize_text(text: str) -> Dict[str, Any]:
    """
    Normalize text by detecting language and translating if needed.
    
    Args:
        text: Input text
    
    Returns:
        Dict containing normalized text and metadata
    """
    if not text.strip():
        return {
            "text": "",
            "original_language": "en",
            "translated": False,
            "metadata": {}
        }
    
    # Detect language
    detected_lang = detect_language(text)
    
    # Check if translation is needed
    if detected_lang != "en" and is_indic_language(text):
        translated_text = translate_text(text, source_lang=detected_lang, target_lang="en")
        return {
            "text": translated_text,
            "original_language": detected_lang,
            "translated": True,
            "metadata": {
                "original_text": text,
                "translation_confidence": 0.8  # Placeholder
            }
        }
    else:
        return {
            "text": text,
            "original_language": detected_lang,
            "translated": False,
            "metadata": {}
        }


def get_supported_languages() -> Dict[str, str]:
    """Get list of supported languages."""
    return {
        "en": "English",
        "hi": "Hindi",
        "ta": "Tamil",
        "te": "Telugu",
        "bn": "Bengali",
        "mr": "Marathi",
        "gu": "Gujarati",
        "kn": "Kannada",
        "ml": "Malayalam",
        "pa": "Punjabi",
        "ur": "Urdu"
    }
