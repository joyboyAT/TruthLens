"""
Input processing and normalization for TruthLens.
"""

import requests
from pathlib import Path
from typing import Dict, Any, Union, Optional
from .detector import InputType, detect_input_type, validate_input


def process_input(content: Union[str, bytes, Path]) -> Dict[str, Any]:
    """
    Process input content and normalize to text.
    
    Args:
        content: Input content (string, bytes, or file path)
    
    Returns:
        Dict containing processed text and metadata
    """
    # Detect input type
    input_type = detect_input_type(content)
    
    # Validate input
    validation = validate_input(content, input_type)
    if not validation["valid"]:
        return {
            "success": False,
            "text": "",
            "errors": validation["errors"],
            "metadata": validation["metadata"]
        }
    
    try:
        # Process based on type
        if input_type == InputType.URL:
            return _process_url(content)
        elif input_type == InputType.IMAGE:
            return _process_image(content)
        elif input_type == InputType.VIDEO:
            return _process_video(content)
        elif input_type == InputType.AUDIO:
            return _process_audio(content)
        elif input_type == InputType.TEXT:
            return _process_text(content)
        elif input_type == InputType.FILE:
            return _process_file(content)
        else:
            return {
                "success": False,
                "text": "",
                "errors": ["Unsupported input type"],
                "metadata": validation["metadata"]
            }
    
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "errors": [f"Processing error: {str(e)}"],
            "metadata": validation["metadata"]
        }


def _process_url(url: str) -> Dict[str, Any]:
    """Process URL and extract text content."""
    try:
        # Fetch URL content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Extract text from HTML
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return {
            "success": True,
            "text": text,
            "errors": [],
            "metadata": {
                "url": url,
                "content_type": response.headers.get('content-type', ''),
                "status_code": response.status_code,
                "length": len(text)
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "errors": [f"URL processing error: {str(e)}"],
            "metadata": {"url": url}
        }


def _process_image(image_path: Union[str, Path]) -> Dict[str, Any]:
    """Process image and extract text using OCR."""
    try:
        from .ocr import extract_text_from_image
        text = extract_text_from_image(image_path)
        
        return {
            "success": True,
            "text": text,
            "errors": [],
            "metadata": {
                "source": "image",
                "path": str(image_path),
                "length": len(text)
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "errors": [f"Image processing error: {str(e)}"],
            "metadata": {"source": "image", "path": str(image_path)}
        }


def _process_video(video_path: Union[str, Path]) -> Dict[str, Any]:
    """Process video and extract text from audio."""
    try:
        from .asr import transcribe_audio
        text = transcribe_audio(video_path)
        
        return {
            "success": True,
            "text": text,
            "errors": [],
            "metadata": {
                "source": "video",
                "path": str(video_path),
                "length": len(text)
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "errors": [f"Video processing error: {str(e)}"],
            "metadata": {"source": "video", "path": str(video_path)}
        }


def _process_audio(audio_path: Union[str, Path]) -> Dict[str, Any]:
    """Process audio and extract text."""
    try:
        from .asr import transcribe_audio
        text = transcribe_audio(audio_path)
        
        return {
            "success": True,
            "text": text,
            "errors": [],
            "metadata": {
                "source": "audio",
                "path": str(audio_path),
                "length": len(text)
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "errors": [f"Audio processing error: {str(e)}"],
            "metadata": {"source": "audio", "path": str(audio_path)}
        }


def _process_text(text: str) -> Dict[str, Any]:
    """Process text content."""
    # Normalize text
    text = text.strip()
    
    return {
        "success": True,
        "text": text,
        "errors": [],
        "metadata": {
            "source": "text",
            "length": len(text)
        }
    }


def _process_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Process file content."""
    try:
        file_path = Path(file_path)
        
        # Try to read as text first
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return _process_text(text)
        except UnicodeDecodeError:
            # If not text, try other processing methods
            pass
        
        # Try as image
        try:
            return _process_image(file_path)
        except:
            pass
        
        # Try as audio/video
        try:
            return _process_audio(file_path)
        except:
            pass
        
        return {
            "success": False,
            "text": "",
            "errors": ["Unable to process file content"],
            "metadata": {"source": "file", "path": str(file_path)}
        }
    
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "errors": [f"File processing error: {str(e)}"],
            "metadata": {"source": "file", "path": str(file_path)}
        }
