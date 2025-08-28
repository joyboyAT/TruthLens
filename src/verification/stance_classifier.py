import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


MNLI_LABELS = {
    0: "REFUTED",             # contradiction
    1: "NOT ENOUGH INFO",     # neutral
    2: "SUPPORTED",           # entailment
}


@dataclass
class StanceResult:
    label: str
    probabilities: Dict[str, float]
    raw_logits: Optional[List[float]] = None
    evidence_id: Optional[str] = None
    scores: Optional[Dict[str, float]] = None


class StanceClassifier:
    """NLI-based stance classification using roberta-large-mnli.

    Usage:
        sc = StanceClassifier()
        result = sc.classify_one(claim, evidence_text)
        batch_results = sc.classify_batch(claim, [ev1, ev2, ...])
    """

    def __init__(self, model_name: str = "roberta-large-mnli", device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._offline = os.environ.get("TRUTHLENS_FORCE_OFFLINE", "0") == "1"
        if self._offline:
            self.tokenizer = None  # type: ignore
            self.model = None  # type: ignore
            logger.warning("TRUTHLENS_FORCE_OFFLINE=1 → using offline heuristic NLI")
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Loaded NLI model {model_name} on {self.device}")
            except Exception as e:
                # Offline fallback: simple lexical heuristics for logits
                self._offline = True
                self.tokenizer = None  # type: ignore
                self.model = None  # type: ignore
                logger.warning(f"Failed to load NLI model '{model_name}'. Using offline heuristic. Error: {e}")

    def _predict_logits(self, premises: Sequence[str], hypotheses: Sequence[str]) -> torch.Tensor:
        if self._offline:
            # Heuristic logits based on keyword overlap; order [ref, nei, sup]
            batch = []
            for prem, hyp in zip(premises, hypotheses):
                ps = set((prem or "").lower().split())
                hs = set((hyp or "").lower().split())
                inter = len(ps & hs)
                # crude signals
                has_neg = any(w in prem.lower() for w in ["no", "not", "fake", "false", "deny", "hoax"]) 
                z_sup = 0.2 + 0.15 * inter
                z_ref = (0.3 if has_neg else 0.1) + 0.1 * (len(ps) > 0)
                z_nei = 0.2
                batch.append([float(z_ref), float(z_nei), float(z_sup)])
            return torch.tensor(batch, dtype=torch.float32)
        inputs = self.tokenizer(
            list(premises),
            list(hypotheses),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits  # [batch, 3]
        return logits.detach().cpu()

    def classify_batch(self, claim: str, evidence_texts: List[str], evidence_ids: Optional[List[str]] = None, evidence_scores: Optional[List[Dict[str, float]]] = None, evidence_meta: Optional[List[Dict[str, Any]]] = None) -> List[StanceResult]:
        if not evidence_texts:
            return []
        premises = evidence_texts  # evidence as premise
        hypotheses = [claim] * len(evidence_texts)
        logits = self._predict_logits(premises, hypotheses)
        probs = torch.softmax(logits, dim=1).tolist()
        preds = torch.argmax(logits, dim=1).tolist()

        results: List[StanceResult] = []
        for i, p in enumerate(preds):
            prob_map = {
                MNLI_LABELS[0]: float(probs[i][0]),
                MNLI_LABELS[1]: float(probs[i][1]),
                MNLI_LABELS[2]: float(probs[i][2]),
            }
            results.append(
                StanceResult(
                    label=MNLI_LABELS[int(p)],
                    probabilities=prob_map,
                    raw_logits=[float(v) for v in logits[i].tolist()],
                    evidence_id=(evidence_ids[i] if evidence_ids and i < len(evidence_ids) else None),
                    scores=(evidence_scores[i] if evidence_scores and i < len(evidence_scores) else None),
                )
            )
        return results

    def classify_one(self, claim: str, evidence_text: str) -> StanceResult:
        res = self.classify_batch(claim, [evidence_text])
        return res[0] if res else StanceResult(label=MNLI_LABELS[1], probabilities={l: 0.0 for l in MNLI_LABELS.values()}, raw_logits=None)


def classify_stance(
    claim: str,
    evidence: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]],
    model_name: str = "roberta-large-mnli",
) -> Union[StanceResult, List[StanceResult]]:
    """Functional wrapper.

    - If `evidence` is a single string or dict, returns one StanceResult
    - If list, returns list of StanceResult in same order
    Dict evidence can use keys: snippet | text | full_text | title
    """
    clf = StanceClassifier(model_name=model_name)

    def _text_from_ev(ev: Union[str, Dict[str, Any]]) -> str:
        if isinstance(ev, str):
            return ev
        return str(ev.get("snippet") or ev.get("text") or ev.get("full_text") or ev.get("title") or "")

    if isinstance(evidence, list):
        texts = [_text_from_ev(e) for e in evidence]
        ev_ids = [e.get("id") if isinstance(e, dict) else None for e in evidence]
        ev_scores = [e.get("scores") if isinstance(e, dict) else None for e in evidence]
        return clf.classify_batch(claim, texts, evidence_ids=ev_ids, evidence_scores=ev_scores)
    else:
        text = _text_from_ev(evidence)
        return clf.classify_one(claim, text)


