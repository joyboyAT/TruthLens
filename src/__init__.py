"""
TruthLens - A comprehensive fact-checking and misinformation detection system.
"""

from . import data_collection
from . import preprocessing
from . import ingestion
from . import ocr
from . import asr
from . import translation
from . import verification
from . import evidence_retrieval
from . import output_ux

__all__ = [
    "data_collection",
    "preprocessing", 
    "ingestion",
    "ocr",
    "asr",
    "translation",
    "verification",
    "evidence_retrieval",
    "output_ux"
]
