"""
Demo script for Phase 4 Verification with Coupled Confidence Calibration.

This script demonstrates the enhanced calibration approach that combines:
1. Bayesian fusion for conservative probability estimation
2. Variance-based confidence estimation
3. Coupled fusion that balances both approaches
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from pipeline import VerificationPipeline
from coupled_calibrator import (
    calibrate_confidence_coupled,
    calibrate_confidence_weighted_coupled,
    test_coupled_calibration
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_evidence():
    """Create realistic mock evidence for testing."""
    return [
        {
            "evidence_id": "ev_001",
            "url": "https://who.int/covid-vaccines",
            "snippet": "WHO confirms that COVID-19 vaccines are safe and effective. No evidence of infertility has been found in clinical trials or post-marketing surveillance.",
            "domain": "who.int",
            "age_days": 30,
            "scores": {"cosine": 0.85, "cross": 0.92, "rrf": 0.88}
        },
        {
            "evidence_id": "ev_002", 
            "url": "https://cdc.gov/vaccine-safety",
            "snippet": "CDC monitoring shows no increased risk of infertility among vaccinated individuals. Studies with over 100,000 participants found no fertility issues.",
            "domain": "cdc.gov",
            "age_days": 45,
            "scores": {"cosine": 0.78, "cross": 0.85, "rrf": 0.81}
        },
        {
            "evidence_id": "ev_003",
            "url": "https://nejm.org/vaccine-studies", 
            "snippet": "Large-scale study published in NEJM found no correlation between COVID-19 vaccination and fertility problems in both men and women.",
            "domain": "nejm.org",
            "age_days": 60,
            "scores": {"cosine": 0.72, "cross": 0.79, "rrf": 0.75}
        },
        {
            "evidence_id": "ev_004",
            "url": "https://blogspot.com/conspiracy",
            "snippet": "Some people claim vaccines cause infertility, but these claims lack scientific evidence and are not supported by medical research.",
            "domain": "blogspot.com",
            "age_days": 90,
            "scores": {"cosine": 0.65, "cross": 0.68, "rrf": 0.66}
        }
    ]


def run_coupled_demo():
    """Run the coupled calibration demo."""
    print("=== Phase 4 Verification with Coupled Confidence Calibration ===\n")
    
    # Test the coupled calibrator directly
    print("1. Testing Coupled Calibrator:")
    test_coupled_calibration()
    print()
    
    # Create mock evidence
    evidence = create_mock_evidence()
    claim = "COVID-19 vaccines cause infertility"
    
    print(f"2. Testing with Claim: '{claim}'")
    print(f"   Evidence count: {len(evidence)}")
    print()
    
    # Initialize pipeline
    pipeline = VerificationPipeline()
    
    # Run standard pipeline
    print("3. Running Standard Pipeline:")
    standard_result = pipeline.run(claim, evidence, top_k=3, similarity_min=0.4, temperature=1.5)
    
    print(f"   Verdict: {standard_result.verdict['verdict']}")
    print(f"   Label: {standard_result.verdict['label']}")
    print(f"   p_calibrated: {standard_result.verdict['p_calibrated_top']:.3f}")
    print(f"   p_raw: {standard_result.verdict['p_raw_top']:.3f}")
    print(f"   Discrepancy: {standard_result.verdict['discrepancy']}")
    print()
    
    # Test coupled calibration on stance results
    print("4. Testing Coupled Calibration on Stance Results:")
    
    # Extract stance results for coupled calibration
    stance_results = []
    for ev in evidence[:3]:  # Top 3 evidence
        # Simulate NLI results (in real pipeline, these come from stance_classifier)
        if "who.int" in ev["domain"] or "cdc.gov" in ev["domain"]:
            # High confidence REFUTED
            logits = [-2.0, 3.0, -1.0]  # REFUTED
            probs = [0.1, 0.85, 0.05]
        elif "nejm.org" in ev["domain"]:
            # Medium confidence REFUTED
            logits = [-1.5, 2.0, -0.5]  # REFUTED
            probs = [0.15, 0.75, 0.1]
        else:
            # Low confidence NEI
            logits = [-0.5, -0.5, 1.0]  # NEI
            probs = [0.3, 0.3, 0.4]
        
        stance_results.append({
            "evidence_id": ev["evidence_id"],
            "logits": logits,
            "probs": probs,
            "scores": ev["scores"],
            "domain": ev["domain"],
            "age_days": ev["age_days"]
        })
    
    # Test basic coupled calibration
    coupled_result = calibrate_confidence_coupled(stance_results, T=1.5, beta=0.4)
    print("   Basic Coupled Calibration:")
    print(f"     SUPPORTED: {coupled_result['SUPPORTED']:.3f}")
    print(f"     REFUTED: {coupled_result['REFUTED']:.3f}")
    print(f"     NOT ENOUGH INFO: {coupled_result['NOT ENOUGH INFO']:.3f}")
    print(f"     Confidence: {coupled_result['confidence']:.3f}")
    print()
    
    # Test weighted coupled calibration
    weighted_result = calibrate_confidence_weighted_coupled(stance_results, T=1.5, beta=0.4)
    print("   Weighted Coupled Calibration:")
    print(f"     SUPPORTED: {weighted_result['SUPPORTED']:.3f}")
    print(f"     REFUTED: {weighted_result['REFUTED']:.3f}")
    print(f"     NOT ENOUGH INFO: {weighted_result['NOT ENOUGH INFO']:.3f}")
    print(f"     Confidence: {weighted_result['confidence']:.3f}")
    print(f"     Weights: {[f'{w:.3f}' for w in weighted_result['weights']]}")
    print()
    
    # Compare approaches
    print("5. Comparison of Calibration Approaches:")
    approaches = {
        "Standard": {
            "SUPPORTED": standard_result.verdict['p_calibrated_top'],
            "REFUTED": 1 - standard_result.verdict['p_calibrated_top'] - 0.1,  # Approximate
            "NOT ENOUGH INFO": 0.1  # Approximate
        },
        "Coupled": coupled_result,
        "Weighted Coupled": weighted_result
    }
    
    for name, probs in approaches.items():
        if name == "Standard":
            print(f"   {name}: SUPPORTED={probs['SUPPORTED']:.3f}, REFUTED={probs['REFUTED']:.3f}, NEI={probs['NOT ENOUGH INFO']:.3f}")
        else:
            print(f"   {name}: SUPPORTED={probs['SUPPORTED']:.3f}, REFUTED={probs['REFUTED']:.3f}, NEI={probs['NOT ENOUGH INFO']:.3f}, Conf={probs['confidence']:.3f}")
    
    print()
    print("6. Key Benefits of Coupled Calibration:")
    print("   - Bayesian fusion provides conservative probability estimates")
    print("   - Variance-based confidence captures evidence agreement")
    print("   - Coupling parameter (β) balances strict vs. average fusion")
    print("   - Weighted version accounts for evidence quality differences")
    print("   - Better uncertainty quantification for fact-checking")


if __name__ == "__main__":
    run_coupled_demo()
