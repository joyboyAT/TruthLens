#!/usr/bin/env python3
"""
Enhanced TruthLens Pipeline
Uses dynamic evidence retrieval and NLI verification for any input text
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evidence_retrieval.dynamic_evidence_retriever import get_dynamic_evidence, DynamicEvidence
from src.verification.enhanced_verifier import verify_text_with_evidence, EnhancedVerifier

@dataclass
class EnhancedPipelineResult:
    """Result of the enhanced pipeline processing."""
    input_text: str
    evidence_retrieved: List[DynamicEvidence]
    verification_results: List[Dict[str, Any]]
    overall_verdict: Dict[str, Any]
    processing_time: float
    sources_checked: List[str]

class EnhancedTruthLensPipeline:
    """
    Enhanced TruthLens pipeline that works with any input text.
    Uses dynamic evidence retrieval and NLI verification.
    """
    
    def __init__(self):
        self.verifier = EnhancedVerifier()
        print("✅ Enhanced TruthLens Pipeline initialized")
    
    def process_text(self, text: str, max_evidence: int = 5) -> EnhancedPipelineResult:
        """
        Process any input text through the enhanced pipeline.
        
        Args:
            text: Input text to analyze (can be any text, not just claims)
            max_evidence: Maximum number of evidence items to retrieve
            
        Returns:
            EnhancedPipelineResult with all analysis results
        """
        start_time = time.time()
        
        print(f"🔍 Processing text: {text[:100]}...")
        
        # Step 1: Retrieve dynamic evidence
        print("📚 Retrieving evidence from external sources...")
        evidence_list = get_dynamic_evidence(text, max_evidence)
        
        # Step 2: Verify text against evidence using NLI
        print("🔍 Verifying text against evidence...")
        verification_results = self.verifier.verify_text_against_evidence(text, evidence_list)
        
        # Step 3: Get overall verdict
        print("⚖️ Computing overall verdict...")
        overall_verdict = self.verifier.get_overall_verdict(verification_results)
        
        # Step 4: Prepare results
        processing_time = time.time() - start_time
        sources_checked = list(set([ev.source for ev in evidence_list]))
        
        result = EnhancedPipelineResult(
            input_text=text,
            evidence_retrieved=evidence_list,
            verification_results=[self._verification_result_to_dict(vr) for vr in verification_results],
            overall_verdict=overall_verdict,
            processing_time=processing_time,
            sources_checked=sources_checked
        )
        
        print(f"✅ Processing completed in {processing_time:.2f}s")
        print(f"📊 Verdict: {overall_verdict['verdict']} (confidence: {overall_verdict['confidence']:.2f})")
        
        return result
    
    def _verification_result_to_dict(self, vr) -> Dict[str, Any]:
        """Convert VerificationResult to dictionary for JSON serialization."""
        return {
            "claim_text": vr.claim_text,
            "evidence_text": vr.evidence_text,
            "stance": vr.stance,
            "confidence_score": vr.confidence_score,
            "source": vr.source,
            "reasoning": vr.reasoning
        }
    
    def get_formatted_response(self, result: EnhancedPipelineResult) -> Dict[str, Any]:
        """Format the pipeline result for API response."""
        
        return {
            "input": {
                "text": result.input_text,
                "length": len(result.input_text)
            },
            "evidence_retrieval": {
                "total_evidence": len(result.evidence_retrieved),
                "sources_checked": result.sources_checked,
                "evidence": [
                    {
                        "id": ev.id,
                        "title": ev.title,
                        "content": ev.content,
                        "source": ev.source,
                        "url": ev.url,
                        "relevance_score": ev.relevance_score,
                        "source_type": ev.source_type,
                        "timestamp": ev.timestamp
                    } for ev in result.evidence_retrieved
                ]
            },
            "verification": {
                "total_verifications": len(result.verification_results),
                "verification_results": result.verification_results,
                "overall_verdict": result.overall_verdict
            },
            "summary": {
                "processing_time": result.processing_time,
                "verdict": result.overall_verdict["verdict"],
                "confidence": result.overall_verdict["confidence"],
                "reasoning": result.overall_verdict["reasoning"],
                "evidence_count": result.overall_verdict["evidence_count"]
            },
            "metadata": {
                "pipeline_version": "enhanced",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }

# Convenience function for easy integration
def process_text_enhanced(text: str, max_evidence: int = 5) -> Dict[str, Any]:
    """Process any text through the enhanced pipeline and return formatted results."""
    pipeline = EnhancedTruthLensPipeline()
    result = pipeline.process_text(text, max_evidence)
    return pipeline.get_formatted_response(result)
