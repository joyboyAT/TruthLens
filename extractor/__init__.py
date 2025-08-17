# TruthLens extractor package

from .claim_detector import is_claim  # noqa: F401
from .claim_extractor import extract_claim_spans  # noqa: F401
from .atomicizer import to_atomic  # noqa: F401
from .context import analyze_context  # noqa: F401
from .ranker import score_claim  # noqa: F401
from .pipeline import process_text  # noqa: F401

__all__ = [
	"is_claim",
	"extract_claim_spans",
	"to_atomic",
	"analyze_context",
	"score_claim",
	"process_text",
]
