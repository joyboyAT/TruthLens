"""
TruthLens Phase 5: Manipulation Detection & User Education Module

This module implements the user explanation layer for detecting misleading content
and generating educational prebunking materials.

Components:
- Rule-based manipulation detection
- LLM-based pattern recognition
- Prebunk tip generation
- UX output formatting
"""

__version__ = '0.1.0'

from .rules import ManipulationRuleDetector
from .llm_detector import LLMManipulationDetector
from .prebunk_generator import PrebunkGenerator
from .output_builder import ManipulationOutputBuilder
from .pipeline import ManipulationPipeline

__all__ = [
    'ManipulationRuleDetector',
    'LLMManipulationDetector', 
    'PrebunkGenerator',
    'ManipulationOutputBuilder',
    'ManipulationPipeline'
]
