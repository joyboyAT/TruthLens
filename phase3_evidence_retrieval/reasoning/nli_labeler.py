"""
NLI labeling for TruthLens Phase 3.

- Models:
  - English: facebook/bart-large-mnli
  - Multilingual: MoritzLaurer/multilingual-MiniLMv2-L6-mnli
- For each (claim, chunk_text), classify entail/contradict/neutral
- Aggregate per source (document) by majority or max-confidence
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Schemas
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from schemas.evidence import TextChunk  # type: ignore

EN_MODEL = "facebook/bart-large-mnli"
MULTI_MODEL = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli"


@dataclass
class NLIConfig:
	use_multilingual: bool = False
	batch_size: int = 8
	max_length: int = 512
	device: Optional[str] = None


class NLIClassifier:
	"""MNLI classifier for entailment/contradiction/neutral."""

	def __init__(self, config: Optional[NLIConfig] = None):
		self.config = config or NLIConfig()
		self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
		model_name = MULTI_MODEL if self.config.use_multilingual else EN_MODEL
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
		self.model.eval()
		# label mapping from model config
		self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
		self.label2id = {v.lower(): int(k) for k, v in self.model.config.label2id.items()}

	@torch.no_grad()
	def classify_pairs(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float | str]]:
		"""Classify list of (premise=chunk, hypothesis=claim) or (claim, chunk).
		We treat claim as hypothesis and chunk as premise.
		Returns list of dicts with 'label' and confidence per label.
		"""
		results: List[Dict[str, float | str]] = []
		bsz = self.config.batch_size
		for i in range(0, len(pairs), bsz):
			batch = pairs[i : i + bsz]
			premises = [p[1] for p in batch]  # chunk_text as premise
			hypotheses = [p[0] for p in batch]  # claim as hypothesis
			enc = self.tokenizer(
				premises,
				hypotheses,
				padding=True,
				truncation=True,
				max_length=self.config.max_length,
				return_tensors="pt",
			).to(self.device)
			logits = self.model(**enc).logits  # [B, 3]
			probs = F.softmax(logits, dim=-1)
			for j in range(probs.size(0)):
				p = probs[j].detach().cpu()
				# Derive human labels
				label_idx = int(torch.argmax(p).item())
				label_name = self.id2label.get(label_idx, "ENTAILMENT").lower()
				# Normalize to entail/contradict/neutral names
				if "entail" in label_name:
					label = "entail"
				elif "contrad" in label_name:
					label = "contradict"
				else:
					label = "neutral"
				results.append(
					{
						"label": label,
						"entail": float(p[self.label2id.get("entailment", label_idx)].item()),
						"neutral": float(p[self.label2id.get("neutral", label_idx)].item()),
						"contradict": float(p[self.label2id.get("contradiction", label_idx)].item()),
					}
				)
		return results

	def classify_chunks(self, claim: str, chunks: List[TextChunk]) -> List[Dict[str, float | str]]:
		pairs = [(claim, c.text) for c in chunks]
		return self.classify_pairs(pairs)

	def aggregate_by_source(self, claim: str, source_to_chunks: Dict[str, List[TextChunk]], method: str = "majority") -> Dict[str, Dict[str, float | str]]:
		"""Aggregate chunk-level labels per source (document).

		Returns mapping of source_id -> {label, entail, neutral, contradict} using either majority label or max-confidence.
		"""
		out: Dict[str, Dict[str, float | str]] = {}
		for source_id, chunks in source_to_chunks.items():
			preds = self.classify_chunks(claim, chunks)
			if not preds:
				continue
			if method == "majority":
				counts = {"entail": 0, "contradict": 0, "neutral": 0}
				avg = {"entail": 0.0, "contradict": 0.0, "neutral": 0.0}
				for pr in preds:
					lbl = str(pr["label"]).lower()
					counts[lbl] += 1
					for k in ("entail", "contradict", "neutral"):
						avg[k] += float(pr[k])
					total = float(len(preds))
				for k in avg:
					avg[k] /= total
				final_label = max(counts.items(), key=lambda x: x[1])[0]
				out[source_id] = {"label": final_label, **avg}
			else:  # max-confidence
				# pick the chunk with highest max(prob)
				best = max(preds, key=lambda pr: max(float(pr["entail"]), float(pr["contradict"]), float(pr["neutral"])) )
				out[source_id] = best
		return out
