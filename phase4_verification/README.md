Phase 4 — Verification

Overview
- Input: Claims from Phase 2 and evidence from Phase 3
- Output: Final verdict per claim with calibrated confidence and citations

Modules
- src/evidence_selection.py: Embedding similarity search to filter top-3 evidence snippets per claim
- src/stance_classifier.py: NLI model to classify support/contradiction/neutral
- src/confidence_calibrator.py: Temperature scaling and score fusion
- src/verdict_mapper.py: Maps confidence to traffic-light UX and structured output
- src/pipeline.py: Orchestrates claim → verdict

Data Folders
- data/claims: Input claims
- data/evidence: Retrieved evidence bundle (Phase 3)
- data/stance_results: Raw stance logits + probabilities
- data/verdicts: Final verdict JSON with calibrated score + citations

Quickstart (Step 1: Evidence Selection)
1) Install requirements
   pip install -r phase4_verification/requirements.txt

2) Run tests for evidence selection
   pytest -k test_evidence_selection -v

Config
- config/model_config.yaml: Model names for embeddings/NLI and calibration params
- config/thresholds.yaml: Similarity and confidence thresholds
- config/ux_mapping.yaml: Traffic-light mapping rules

Logs
- logs/evidence.log, logs/stance.log, logs/calibration.log

## TruthLens — Phase 4: Verification

This phase takes a claim (from earlier phases) and a set of retrieved evidence (from Phase 3), then verifies the claim using an NLI model and calibrated confidence. The overall flow:

- Evidence selection (embedding similarity) → Stance classification (NLI) → Confidence calibration → Verdict mapping (traffic light)

This repo currently includes Step 1 (Evidence Selection).

### Step 1: Evidence Selection
- Input: claim string + list of evidence snippets
- Model: sentence-transformers/all-mpnet-base-v2 (cosine similarity)
- Output: top-3 evidence above a threshold (default 0.6), with similarity scores

Run tests:

```bash
pip install -r phase4_verification/requirements.txt
pytest phase4_verification/tests/test_evidence_selection.py -q
```


