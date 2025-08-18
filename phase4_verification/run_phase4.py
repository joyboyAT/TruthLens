import logging
from typing import List, Dict, Any

from src.pipeline import VerificationPipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    claim = "COVID-19 vaccines cause infertility"
    evidence: List[Dict[str, Any]] = [
        {"id": "1", "title": "WHO: COVID-19 vaccines are safe", "snippet": "No evidence linking vaccines to infertility.", "url": "https://who.int/a"},
        {"id": "2", "title": "CDC safety update", "snippet": "Studies show no increased risk of infertility.", "url": "https://cdc.gov/b"},
        {"id": "3", "title": "Random blog", "snippet": "I think vaccines are harmful", "url": "https://blog.example/c"},
        {"id": "4", "title": "Systematic review", "snippet": "Comprehensive analysis finds no association with infertility.", "url": "https://nejm.org/d"},
        {"id": "5", "title": "Unrelated topic", "snippet": "Stock market news.", "url": "https://news.example/e"},
    ]

    # Test standard calibration
    pipe = VerificationPipeline()
    out = pipe.run(claim, evidence, top_k=3, similarity_min=0.4, temperature=1.5)

    print("\n=== Phase 4 Verification (Standard Calibration) ===")
    print("Claim:", out.claim)
    print("Verdict:", out.verdict["verdict"], "p_calibrated=", round(out.verdict["p_calibrated_top"], 3), "p_raw=", round(out.verdict["p_raw_top"], 3))
    print("Discrepancy:", out.verdict["discrepancy"])
    print("Notes:", out.verdict.get("explanations", {}).get("notes", []))
    print("\nCitations (top-3):")
    for c in out.verdict["citations"]:
        print("-", c["url"])

    # Test coupled calibration
    pipe_coupled = VerificationPipeline()
    pipe_coupled.use_coupled_calibration = True
    out_coupled = pipe_coupled.run(claim, evidence, top_k=3, similarity_min=0.4, temperature=1.5)

    print("\n=== Phase 4 Verification (Coupled Calibration) ===")
    print("Claim:", out_coupled.claim)
    print("Verdict:", out_coupled.verdict["verdict"], "p_calibrated=", round(out_coupled.verdict["p_calibrated_top"], 3), "p_raw=", round(out_coupled.verdict["p_raw_top"], 3))
    print("Discrepancy:", out_coupled.verdict["discrepancy"])
    print("Notes:", out_coupled.verdict.get("explanations", {}).get("notes", []))
    print("\nCitations (top-3):")
    for c in out_coupled.verdict["citations"]:
        print("-", c["url"]) 


if __name__ == "__main__":
    main()


