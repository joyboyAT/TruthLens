"""
Speech recognition using OpenAI Whisper for multilingual support.
"""

import io
import tempfile
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available")

# Try to import moviepy for video processing
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.warning("MoviePy not available")

# Initialize Whisper model (lazy loading)
_model = None


def _get_whisper_model():
    """Get or create Whisper model."""
    global _model
    if _model is None and WHISPER_AVAILABLE:
        # Use base model for faster processing
        _model = whisper.load_model("base")
    return _model


def transcribe_audio(audio_path: Union[str, Path]) -> str:
    """
    Transcribe audio file to text.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Transcribed text as string
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if not WHISPER_AVAILABLE:
        raise RuntimeError("Whisper not available")
    
    try:
        model = _get_whisper_model()
        result = model.transcribe(str(audio_path))
        return result["text"].strip()
    
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise RuntimeError(f"Audio transcription failed: {e}")


def transcribe_video(video_path: Union[str, Path]) -> str:
    """
    Transcribe video file to text by extracting audio first.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Transcribed text as string
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not WHISPER_AVAILABLE:
        raise RuntimeError("Whisper not available")
    
    if not MOVIEPY_AVAILABLE:
        raise RuntimeError("MoviePy not available for video processing")
    
    try:
        # Extract audio from video
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            # Load video and extract audio
            video = VideoFileClip(str(video_path))
            video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            video.close()
            
            # Transcribe the extracted audio
            text = transcribe_audio(temp_audio_path)
            
        finally:
            # Clean up temporary file
            Path(temp_audio_path).unlink(missing_ok=True)
        
        return text
    
    except Exception as e:
        logger.error(f"Video transcription failed: {e}")
        raise RuntimeError(f"Video transcription failed: {e}")


def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes to text.
    
    Args:
        audio_bytes: Audio data as bytes
    
    Returns:
        Transcribed text as string
    """
    if not WHISPER_AVAILABLE:
        raise RuntimeError("Whisper not available")
    
    try:
        # Save bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        try:
            # Transcribe the temporary file
            text = transcribe_audio(temp_audio_path)
            
        finally:
            # Clean up temporary file
            Path(temp_audio_path).unlink(missing_ok=True)
        
        return text
    
    except Exception as e:
        logger.error(f"Audio bytes transcription failed: {e}")
        raise RuntimeError(f"Audio bytes transcription failed: {e}")


def detect_language(audio_path: Union[str, Path]) -> str:
    """
    Detect the language of audio content.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Language code (e.g., 'en', 'hi', 'ta')
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if not WHISPER_AVAILABLE:
        raise RuntimeError("Whisper not available")
    
    try:
        model = _get_whisper_model()
        result = model.transcribe(str(audio_path), language=None)
        return result["language"]
    
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return "en"  # Default to English
