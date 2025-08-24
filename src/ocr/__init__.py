"""
Optical Character Recognition (OCR) for TruthLens.
Extracts text from images with multilingual support.
"""

from .extractor import extract_text_from_image, extract_text_from_image_bytes

__all__ = ["extract_text_from_image", "extract_text_from_image_bytes"]
