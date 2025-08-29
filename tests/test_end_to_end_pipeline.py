#!/usr/bin/env python3
"""
End-to-End TruthLens Pipeline Test
A comprehensive, debuggable test for the complete TruthLens pipeline.
Can accept user input claims and provide detailed analysis output.
"""

import sys
import os
import time
import traceback
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for debugging (Windows compatible)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Structured result from the pipeline."""
    input_text: str
    phase1_result: Optional[Dict[str, Any]] = None
    phase2_result: Optional[Dict[str, Any]] = None
    phase3_result: Optional[Dict[str, Any]] = None
    phase4_result: Optional[Dict[str, Any]] = None
    phase5_result: Optional[Dict[str, Any]] = None
    final_verdict: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: float = 0.0
    errors: List[str] = None
    success: bool = False

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

class TruthLensPipelineTester:
    """Comprehensive pipeline tester with debugging capabilities."""
    
    def __init__(self, debug_mode: bool = True):
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(f"{__name__}.TruthLensPipelineTester")
        
        if debug_mode:
            self.logger.info("Debug mode enabled - detailed logging will be provided")
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components with error handling."""
        self.logger.info("Initializing TruthLens pipeline components...")
        
        try:
            # Import core components
            from src.enhanced_pipeline import EnhancedTruthLensPipeline
            self.enhanced_pipeline = EnhancedTruthLensPipeline()
            self.logger.info("Enhanced pipeline initialized")
            
        except ImportError as e:
            self.logger.error(f"Failed to import enhanced pipeline: {e}")
            self.enhanced_pipeline = None
        except Exception as e:
            self.logger.error(f"Error initializing enhanced pipeline: {e}")
            self.enhanced_pipeline = None
        
        try:
            # Import claim processing
            from src.claim_processing import ClaimProcessor
            self.claim_processor = ClaimProcessor()
            self.logger.info("Claim processor initialized")
            
        except ImportError as e:
            self.logger.error(f"Failed to import claim processor: {e}")
            self.claim_processor = None
    except Exception as e:
            self.logger.error(f"Error initializing claim processor: {e}")
            self.claim_processor = None
    
    def test_phase1_input_processing(self, input_text: str) -> Dict[str, Any]:
        """Test Phase 1: Input & Normalization with detailed logging."""
        self.logger.info("Phase 1: Input & Normalization")
        
        try:
            # Basic text normalization
            normalized_text = input_text.strip()
            
            # Detect language (if langdetect is available)
            language = "en"  # Default to English
            try:
                from langdetect import detect
                language = detect(normalized_text)
            except ImportError:
                self.logger.warning("langdetect not available, using default language")
            except Exception:
                self.logger.warning("Language detection failed, using default language")
            
            # Basic text analysis
            word_count = len(normalized_text.split())
            char_count = len(normalized_text)
            
            result = {
                "original_text": input_text,
                "normalized_text": normalized_text,
                "language": language,
                "word_count": word_count,
                "char_count": char_count,
                "success": True
            }
            
            self.logger.info(f"Phase 1 completed: {word_count} words, {char_count} characters")
            return result
            
        except Exception as e:
            error_msg = f"Phase 1 failed: {str(e)}"
            self.logger.error(error_msg)
            if self.debug_mode:
                self.logger.error(traceback.format_exc())
            return {"success": False, "error": error_msg}
    
    def test_phase2_claim_extraction(self, normalized_text: str) -> Dict[str, Any]:
        """Test Phase 2: Claim Extraction & Ranking."""
        self.logger.info("Phase 2: Claim Extraction & Ranking")
        
        if not self.claim_processor:
            error_msg = "Claim processor not available"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            # Process claims
            result = self.claim_processor.process_claims(normalized_text)
            
            # Convert to dictionary format
            claims_data = []
            for claim in result.atomic_claims:
                claims_data.append({
                    "text": claim.get("text", ""),
                    "checkworthiness": claim.get("checkworthiness", 0.0),
                    "claim_type": claim.get("claim_type", "unknown")
                })
            
            phase2_result = {
                "total_claims": result.total_claims,
                "checkworthy_claims": result.checkworthy_claims,
                "average_score": result.average_score,
                "processing_time": result.processing_time,
                "claims": claims_data,
                "success": True
            }
            
            self.logger.info(f"Phase 2 completed: {result.checkworthy_claims} checkworthy claims found")
            return phase2_result
        
    except Exception as e:
            error_msg = f"Phase 2 failed: {str(e)}"
            self.logger.error(error_msg)
            if self.debug_mode:
                self.logger.error(traceback.format_exc())
            return {"success": False, "error": error_msg}
    
    def test_phase3_evidence_retrieval(self, claim_text: str) -> Dict[str, Any]:
        """Test Phase 3: Evidence Retrieval."""
        self.logger.info("Phase 3: Evidence Retrieval")
        
        try:
            # Use enhanced pipeline for evidence retrieval
            if self.enhanced_pipeline:
                result = self.enhanced_pipeline.process_text(claim_text, max_evidence=5)
                
                evidence_data = []
                for ev in result.evidence_retrieved:
                    evidence_data.append({
                        "title": getattr(ev, 'title', 'Unknown'),
                        "snippet": getattr(ev, 'snippet', ''),
                        "url": getattr(ev, 'url', ''),
                        "source": getattr(ev, 'source', 'unknown'),
                        "relevance_score": getattr(ev, 'relevance_score', 0.0)
                    })
                
                phase3_result = {
                    "evidence_count": len(evidence_data),
                    "evidence": evidence_data,
                    "sources_checked": result.sources_checked,
                    "processing_time": result.processing_time,
                    "success": True
                }
                
                self.logger.info(f"Phase 3 completed: {len(evidence_data)} evidence items retrieved")
                return phase3_result
            else:
                # Fallback: mock evidence
        mock_evidence = [
            {
                        "title": "Fact-checking database",
                        "snippet": "This claim has been fact-checked by multiple sources.",
                        "url": "https://example.com/factcheck",
                        "source": "factcheck.org",
                        "relevance_score": 0.8
                    }
                ]
                
                return {
                    "evidence_count": 1,
                    "evidence": mock_evidence,
                    "sources_checked": ["factcheck.org"],
                    "processing_time": 0.1,
                    "success": True,
                    "note": "Using mock evidence (enhanced pipeline not available)"
                }
        
    except Exception as e:
            error_msg = f"Phase 3 failed: {str(e)}"
            self.logger.error(error_msg)
            if self.debug_mode:
                self.logger.error(traceback.format_exc())
            return {"success": False, "error": error_msg}
    
    def test_phase4_verification(self, claim_text: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test Phase 4: Verification & Scoring."""
        self.logger.info("Phase 4: Verification & Scoring")
        
        try:
            if self.enhanced_pipeline:
                # Use enhanced pipeline for verification
                result = self.enhanced_pipeline.process_text(claim_text, max_evidence=len(evidence))
                
                phase4_result = {
                    "verdict": result.overall_verdict.get("verdict", "UNKNOWN"),
                    "confidence": result.overall_verdict.get("confidence", 0.0),
                    "verification_results": result.verification_results,
                    "processing_time": result.processing_time,
                    "success": True
                }
                
                self.logger.info(f"Phase 4 completed: Verdict = {phase4_result['verdict']} (confidence: {phase4_result['confidence']:.3f})")
                return phase4_result
            else:
                # Fallback: simple verification logic
                # This is a basic heuristic - in production, use proper NLI models
                claim_lower = claim_text.lower()
                
                # Simple keyword-based verification
                refuting_keywords = ["false", "debunked", "no evidence", "disproven", "myth"]
                supporting_keywords = ["true", "confirmed", "verified", "proven", "fact"]
                
                refuting_count = sum(1 for keyword in refuting_keywords if keyword in claim_lower)
                supporting_count = sum(1 for keyword in supporting_keywords if keyword in claim_lower)
                
                if refuting_count > supporting_count:
                    verdict = "REFUTED"
                    confidence = 0.6
                elif supporting_count > refuting_count:
                    verdict = "SUPPORTED"
                    confidence = 0.6
                else:
                    verdict = "NOT ENOUGH INFO"
                    confidence = 0.3
                
                return {
                    "verdict": verdict,
                    "confidence": confidence,
                    "verification_results": [],
                    "processing_time": 0.1,
                    "success": True,
                    "note": "Using basic keyword verification (enhanced pipeline not available)"
                }
                
        except Exception as e:
            error_msg = f"Phase 4 failed: {str(e)}"
            self.logger.error(error_msg)
            if self.debug_mode:
                self.logger.error(traceback.format_exc())
            return {"success": False, "error": error_msg}
    
    def test_phase5_user_explanation(self, claim_text: str, verdict: str, confidence: float, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test Phase 5: User Explanation Layer."""
        self.logger.info("Phase 5: User Explanation Layer")
        
        try:
            # Generate explanation based on verdict
            if verdict == "REFUTED":
                explanation = f"This claim appears to be false based on available evidence. Multiple sources indicate that this information is incorrect or misleading."
                explanation_type = "refutation"
            elif verdict == "SUPPORTED":
                explanation = f"This claim appears to be supported by available evidence. Multiple sources confirm this information."
                explanation_type = "confirmation"
            else:
                explanation = f"There is insufficient evidence to determine the accuracy of this claim. More information is needed to verify this statement."
                explanation_type = "insufficient_info"
        
        # Generate manipulation cues
            manipulation_cues = []
            claim_lower = claim_text.lower()
            
            if any(word in claim_lower for word in ["urgent", "immediate", "act now"]):
                manipulation_cues.append({
                    "type": "urgency",
                    "description": "Uses urgent language to pressure quick action"
                })
            
            if any(word in claim_lower for word in ["conspiracy", "cover-up", "they don't want you to know"]):
                manipulation_cues.append({
                    "type": "conspiracy",
                    "description": "Suggests hidden agendas or cover-ups"
                })
            
            if any(word in claim_lower for word in ["100%", "guaranteed", "proven"]):
                manipulation_cues.append({
                    "type": "certainty",
                    "description": "Uses absolute certainty language"
                })
            
            # Format evidence for display
            formatted_evidence = []
            for ev in evidence:
                formatted_evidence.append({
                    "title": ev.get("title", "Unknown"),
                    "snippet": ev.get("snippet", "")[:200] + "..." if len(ev.get("snippet", "")) > 200 else ev.get("snippet", ""),
                    "source": ev.get("source", "unknown"),
                    "url": ev.get("url", "")
                })
            
            phase5_result = {
            "explanation": explanation,
                "explanation_type": explanation_type,
                "manipulation_cues": manipulation_cues,
                "formatted_evidence": formatted_evidence,
                "success": True
            }
            
            self.logger.info(f"Phase 5 completed: Generated {len(manipulation_cues)} manipulation cues")
            return phase5_result
            
    except Exception as e:
            error_msg = f"Phase 5 failed: {str(e)}"
            self.logger.error(error_msg)
            if self.debug_mode:
                self.logger.error(traceback.format_exc())
            return {"success": False, "error": error_msg}
    
    def run_complete_pipeline(self, input_text: str) -> PipelineResult:
        """Run the complete end-to-end pipeline with comprehensive error handling."""
        start_time = time.time()
        self.logger.info("Starting complete TruthLens pipeline test")
        self.logger.info(f"Input text: {input_text[:100]}...")
        
        result = PipelineResult(input_text=input_text)
        
        try:
            # Phase 1: Input Processing
            phase1_result = self.test_phase1_input_processing(input_text)
            result.phase1_result = phase1_result
            
            if not phase1_result.get("success", False):
                result.errors.append(f"Phase 1 failed: {phase1_result.get('error', 'Unknown error')}")
                result.success = False
                return result
            
            # Phase 2: Claim Extraction
            normalized_text = phase1_result["normalized_text"]
            phase2_result = self.test_phase2_claim_extraction(normalized_text)
            result.phase2_result = phase2_result
            
            if not phase2_result.get("success", False):
                result.errors.append(f"Phase 2 failed: {phase2_result.get('error', 'Unknown error')}")
                result.success = False
                return result
            
            # Get the most checkworthy claim for further processing
            claims = phase2_result.get("claims", [])
            if claims:
                top_claim = max(claims, key=lambda x: x.get("checkworthiness", 0))
                claim_text = top_claim["text"]
            else:
                # Use original text if no claims extracted
                claim_text = normalized_text
    
    # Phase 3: Evidence Retrieval
            phase3_result = self.test_phase3_evidence_retrieval(claim_text)
            result.phase3_result = phase3_result
            
            if not phase3_result.get("success", False):
                result.errors.append(f"Phase 3 failed: {phase3_result.get('error', 'Unknown error')}")
                result.success = False
                return result
            
            # Phase 4: Verification
            evidence = phase3_result.get("evidence", [])
            phase4_result = self.test_phase4_verification(claim_text, evidence)
            result.phase4_result = phase4_result
            
            if not phase4_result.get("success", False):
                result.errors.append(f"Phase 4 failed: {phase4_result.get('error', 'Unknown error')}")
                result.success = False
                return result
            
            # Phase 5: User Explanation
            verdict = phase4_result.get("verdict", "UNKNOWN")
            confidence = phase4_result.get("confidence", 0.0)
            phase5_result = self.test_phase5_user_explanation(claim_text, verdict, confidence, evidence)
            result.phase5_result = phase5_result
            
            if not phase5_result.get("success", False):
                result.errors.append(f"Phase 5 failed: {phase5_result.get('error', 'Unknown error')}")
                result.success = False
                return result
            
            # Set final results
            result.final_verdict = verdict
            result.confidence = confidence
            result.success = True
            
    except Exception as e:
            error_msg = f"Pipeline failed with exception: {str(e)}"
            self.logger.error(error_msg)
            if self.debug_mode:
                self.logger.error(traceback.format_exc())
            result.errors.append(error_msg)
            result.success = False
        
        finally:
            result.processing_time = time.time() - start_time
            self.logger.info(f"Total processing time: {result.processing_time:.2f} seconds")
        
        return result
    
    def print_results(self, result: PipelineResult):
        """Print formatted results to console."""
        print("\n" + "="*80)
        print("TRUTHLENS PIPELINE TEST RESULTS")
        print("="*80)
        
        print(f"Input Text: {result.input_text}")
        print(f"Processing Time: {result.processing_time:.2f} seconds")
        print(f"Success: {result.success}")
        
        if result.errors:
            print(f"Errors: {len(result.errors)}")
            for error in result.errors:
                print(f"   - {error}")
        
        if result.success:
            print(f"\nFinal Verdict: {result.final_verdict}")
            print(f"Confidence: {result.confidence:.3f}")
            
            # Phase 1 results
            if result.phase1_result:
                p1 = result.phase1_result
                print(f"\nPhase 1 - Input Processing:")
                print(f"   Language: {p1.get('language', 'unknown')}")
                print(f"   Word Count: {p1.get('word_count', 0)}")
                print(f"   Character Count: {p1.get('char_count', 0)}")
            
            # Phase 2 results
            if result.phase2_result:
                p2 = result.phase2_result
                print(f"\nPhase 2 - Claim Extraction:")
                print(f"   Total Claims: {p2.get('total_claims', 0)}")
                print(f"   Checkworthy Claims: {p2.get('checkworthy_claims', 0)}")
                print(f"   Average Score: {p2.get('average_score', 0):.3f}")
                
                claims = p2.get("claims", [])
                if claims:
                    print("   Top Claims:")
                    for i, claim in enumerate(claims[:3], 1):
                        print(f"     {i}. {claim.get('text', '')[:60]}... (score: {claim.get('checkworthiness', 0):.3f})")
            
            # Phase 3 results
            if result.phase3_result:
                p3 = result.phase3_result
                print(f"\nPhase 3 - Evidence Retrieval:")
                print(f"   Evidence Count: {p3.get('evidence_count', 0)}")
                print(f"   Sources Checked: {', '.join(p3.get('sources_checked', []))}")
                
                if p3.get("note"):
                    print(f"   Note: {p3['note']}")
            
            # Phase 4 results
            if result.phase4_result:
                p4 = result.phase4_result
                print(f"\nPhase 4 - Verification:")
                print(f"   Verdict: {p4.get('verdict', 'UNKNOWN')}")
                print(f"   Confidence: {p4.get('confidence', 0):.3f}")
                
                if p4.get("note"):
                    print(f"   Note: {p4['note']}")
            
            # Phase 5 results
            if result.phase5_result:
                p5 = result.phase5_result
                print(f"\nPhase 5 - User Explanation:")
                print(f"   Explanation: {p5.get('explanation', '')}")
                print(f"   Explanation Type: {p5.get('explanation_type', 'unknown')}")
                
                cues = p5.get("manipulation_cues", [])
                if cues:
                    print(f"   Manipulation Cues ({len(cues)}):")
                    for cue in cues:
                        print(f"     - {cue.get('type', 'unknown')}: {cue.get('description', '')}")
        
        print("\n" + "="*80)

def get_user_input() -> str:
    """Get claim input from user."""
    print("\nTruthLens End-to-End Pipeline Test")
    print("="*50)
    print("Enter a claim to test the complete pipeline:")
    print("(Or press Enter to use a default test claim)")
    
    user_input = input("\nClaim: ").strip()
    
    if not user_input:
        # Default test claims
        default_claims = [
            "COVID-19 vaccines cause autism in children according to recent studies.",
            "5G towers emit dangerous radiation that causes cancer in nearby residents.",
            "Water boils at 100 degrees Celsius at sea level under normal atmospheric pressure.",
            "The Earth is flat and NASA has been hiding this fact for decades.",
            "Drinking 8 glasses of water per day is essential for good health."
        ]
        
        print("\nUsing default test claims:")
        for i, claim in enumerate(default_claims, 1):
            print(f"{i}. {claim}")
        
        choice = input("\nSelect a claim (1-5) or press Enter for claim 1: ").strip()
        try:
            choice_idx = int(choice) - 1 if choice else 0
            if 0 <= choice_idx < len(default_claims):
                return default_claims[choice_idx]
        except ValueError:
            pass
        
        return default_claims[0]
    
    return user_input

def main():
    """Main function to run the end-to-end pipeline test."""
    try:
        # Get user input
        input_text = get_user_input()
        
        # Initialize tester
        tester = TruthLensPipelineTester(debug_mode=True)
        
        # Run complete pipeline
        result = tester.run_complete_pipeline(input_text)
        
        # Print results
        tester.print_results(result)
        
        # Save results to file
        output_file = "pipeline_test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        
        if result.success:
            print("\nEnd-to-end pipeline test completed successfully!")
            return 0
        else:
            print("\nEnd-to-end pipeline test completed with errors!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
            return 1
    except Exception as e:
        print(f"\nTest failed with exception: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
