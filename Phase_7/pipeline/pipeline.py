from __future__ import annotations

from typing import Dict, List
import logging

from .extractor import ClaimExtractor
from .context import ContextProcessor
from .retriever import EvidenceRetriever
from .verifier import Verifier
from .manipulation import ManipulationCueDetector
from .prebunk import PrebunkGenerator
from phase6_output_ux.evidence_cards import build_evidence_cards
from phase6_output_ux.share_card import build_share_card


logger = logging.getLogger("Phase_7.pipeline")
if not logger.handlers:
	logging.basicConfig(level=logging.INFO)


def run_pipeline(text: str) -> Dict[str, object]:
	"""End-to-end pipeline over a single input sentence.

	Returns a dict with all intermediate and final outputs.
	"""
	logger.info("Pipeline started")
	logger.info("Input text: %s", text)
	# Configure per-run parameters
	evidence_corpus: List[Dict[str, str]] = [
		{"text": "WHO confirms vaccines are safe", "url": "https://pib.gov/some/path", "source": "PIB"}
	]
	cue_names: List[str] = ["Fear Appeal", "Conspiracy"]
	prebunk_tips: Dict[str, str] = {
		"Fear Appeal": "Fear is used to manipulate. Verify with WHO/CDC.",
		"Conspiracy": "Conspiracies lack proof. Check transparent sources.",
	}

	# Instantiate components
	extractor = ClaimExtractor()
	ctx_proc = ContextProcessor()
	retriever = EvidenceRetriever(evidence_corpus)
	verifier = Verifier()
	cue_det = ManipulationCueDetector(cue_names)
	prebunk = PrebunkGenerator(cue_names, prebunk_tips)

	# Stage 1: claim extraction produces list of {claim, context}
	items = extractor.process({"claim": "", "context": text})
	logger.info("Stage: extract -> %d item(s)", len(items))
	for i, it in enumerate(items):
		logger.info("  [extract %d] %s", i + 1, it)

	results: List[Dict[str, object]] = []
	for idx, it in enumerate(items):
		logger.info("Processing item %d: %s", idx + 1, it.get("claim", ""))
		it1 = ctx_proc.process(it)
		logger.info("  [context] %s", it1.get("context_info"))

		it2 = retriever.process(it1)
		logger.info("  [retriever] evidence=%s", it2.get("evidence"))

		# Evidence cards
		evid_cards = build_evidence_cards(it2.get("evidence") or [])  # type: ignore[arg-type]
		logger.info("  [evidence_cards] %s", evid_cards)
		it2b = {**it2, "evidence_cards": evid_cards}

		it3 = verifier.process(it2b)
		logger.info("  [verify] %s", it3.get("verification"))

		it4 = cue_det.process(it3)
		logger.info("  [cues] %s", it4.get("cue_badges"))

		it5 = prebunk.process(it4)
		logger.info("  [prebunk] %s", it5.get("prebunk"))

		# Share card
		share_text = build_share_card(
			it5.get("claim", ""),
			it5["verification"]["verdict"].split(" — ")[0],  # type: ignore[index]
			(it5.get("evidence_cards") or [""])[0],
			(it5.get("prebunk", {}).get("tips") or [""])[0],  # type: ignore[call-arg]
		)
		logger.info("  [share] %s", share_text)
		final = {**it5, "share": share_text}
		results.append(final)

	out = {"results": results}
	logger.info("Pipeline finished with %d result(s)", len(results))
	return out
