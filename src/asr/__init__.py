"""
Automatic Speech Recognition (ASR) for TruthLens.
Transcribes audio and video content to text.
"""

from .transcriber import transcribe_audio, transcribe_video

__all__ = ["transcribe_audio", "transcribe_video"]
