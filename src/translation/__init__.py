"""
Multilingual translation for TruthLens.
Translates Indic languages to English using Google Translate API.
"""

from .translator import translate_text, detect_language, is_indic_language, normalize_text

__all__ = ["translate_text", "detect_language", "is_indic_language", "normalize_text"]
