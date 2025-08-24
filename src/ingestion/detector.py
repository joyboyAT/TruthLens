"""
Input type detection and validation for TruthLens.
"""

import re
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, Union
from enum import Enum


class InputType(Enum):
    """Supported input types."""
    TEXT = "text"
    URL = "url"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    FILE = "file"
    UNKNOWN = "unknown"


def detect_input_type(content: Union[str, bytes, Path]) -> InputType:
    """
    Detect the type of input content.
    
    Args:
        content: Input content (string, bytes, or file path)
    
    Returns:
        InputType: Detected input type
    """
    if isinstance(content, Path):
        return _detect_file_type(content)
    
    if isinstance(content, bytes):
        return _detect_binary_type(content)
    
    if isinstance(content, str):
        return _detect_text_type(content)
    
    return InputType.UNKNOWN


def _detect_text_type(text: str) -> InputType:
    """Detect type from text content."""
    text = text.strip()
    
    # Check if it's a URL
    if _is_url(text):
        return InputType.URL
    
    # Check if it's a file path
    if _is_file_path(text):
        return InputType.FILE
    
    # Default to text
    return InputType.TEXT


def _detect_file_type(file_path: Path) -> InputType:
    """Detect type from file path."""
    if not file_path.exists():
        return InputType.UNKNOWN
    
    # Get file extension
    ext = file_path.suffix.lower()
    
    # Image formats
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
        return InputType.IMAGE
    
    # Video formats
    if ext in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv']:
        return InputType.VIDEO
    
    # Audio formats
    if ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']:
        return InputType.AUDIO
    
    # Text formats
    if ext in ['.txt', '.md', '.json', '.xml', '.html', '.htm']:
        return InputType.TEXT
    
    # Default to file
    return InputType.FILE


def _detect_binary_type(data: bytes) -> InputType:
    """Detect type from binary data."""
    # Check for common file signatures
    if len(data) < 4:
        return InputType.UNKNOWN
    
    # Image signatures
    if data.startswith(b'\xff\xd8\xff'):  # JPEG
        return InputType.IMAGE
    if data.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
        return InputType.IMAGE
    if data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):  # GIF
        return InputType.IMAGE
    
    # Video signatures
    if data.startswith(b'\x00\x00\x00'):  # MP4
        return InputType.VIDEO
    
    # Audio signatures
    if data.startswith(b'ID3') or data.startswith(b'\xff\xfb'):  # MP3
        return InputType.AUDIO
    
    # Default to unknown
    return InputType.UNKNOWN


def _is_url(text: str) -> bool:
    """Check if text is a valid URL."""
    try:
        result = urllib.parse.urlparse(text)
        return all([result.scheme, result.netloc])
    except:
        return False


def _is_file_path(text: str) -> bool:
    """Check if text looks like a file path."""
    # Check for common path patterns
    path_patterns = [
        r'^[A-Za-z]:\\',  # Windows drive
        r'^[A-Za-z]:/',  # Windows with forward slash
        r'^/',  # Unix absolute path
        r'^\./',  # Unix relative path
        r'^[A-Za-z]:\\[^\\]+\.(jpg|jpeg|png|gif|bmp|tiff|webp|mp4|avi|mov|wmv|flv|webm|mkv|mp3|wav|flac|aac|ogg|m4a|txt|md|json|xml|html|htm)$',  # Windows with extension
        r'^[A-Za-z]:/[^/]+\.(jpg|jpeg|png|gif|bmp|tiff|webp|mp4|avi|mov|wmv|flv|webm|mkv|mp3|wav|flac|aac|ogg|m4a|txt|md|json|xml|html|htm)$',  # Windows with forward slash and extension
    ]
    
    return any(re.match(pattern, text, re.IGNORECASE) for pattern in path_patterns)


def validate_input(content: Union[str, bytes, Path], input_type: InputType) -> Dict[str, Any]:
    """
    Validate input content and return metadata.
    
    Args:
        content: Input content
        input_type: Detected input type
    
    Returns:
        Dict containing validation results and metadata
    """
    result = {
        "valid": True,
        "type": input_type.value,
        "size": 0,
        "metadata": {},
        "errors": []
    }
    
    try:
        if isinstance(content, str):
            result["size"] = len(content.encode('utf-8'))
            result["metadata"]["length"] = len(content)
            result["metadata"]["language"] = _detect_language(content)
            
        elif isinstance(content, bytes):
            result["size"] = len(content)
            
        elif isinstance(content, Path):
            if content.exists():
                result["size"] = content.stat().st_size
                result["metadata"]["filename"] = content.name
                result["metadata"]["extension"] = content.suffix
            else:
                result["valid"] = False
                result["errors"].append("File does not exist")
        
        # Validate size limits
        if result["size"] > 50 * 1024 * 1024:  # 50MB limit
            result["valid"] = False
            result["errors"].append("File too large (max 50MB)")
        
        # Type-specific validation
        if input_type == InputType.URL:
            if not _is_url(str(content)):
                result["valid"] = False
                result["errors"].append("Invalid URL format")
        
        elif input_type == InputType.TEXT:
            if len(str(content).strip()) == 0:
                result["valid"] = False
                result["errors"].append("Empty text content")
        
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Validation error: {str(e)}")
    
    return result


def _detect_language(text: str) -> str:
    """Simple language detection."""
    # Basic language detection based on character sets
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
    
    # Default to English
    return "en"
