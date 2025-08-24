"""
Input ingestion and type detection for TruthLens.
Handles URL, text, image, video, and audio inputs.
"""

from .detector import detect_input_type, validate_input
from .processor import process_input

__all__ = ["detect_input_type", "validate_input", "process_input"]
