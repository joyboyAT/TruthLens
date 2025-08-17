from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, TypedDict
import os


class Span(TypedDict):
	text: str
	start: int
	end: int
	conf: float


# --- Fallback: regex-based noun/verb chunk heuristic -------------------------
_FALLBACK_VERB = r"\b(bought|buy|buys|acquired|acquires|acquire|fired|firing|fires|causes?|caused|leads?\s+to|led\s+to|results?\s+in)\b"


def _fallback_extract(sentence: str) -> List[Span]:
	import re
	m = re.search(_FALLBACK_VERB, sentence, flags=re.IGNORECASE)
	if not m:
		return []
	# expand to end of clause
	start = max(0, sentence.rfind(' ', 0, m.start()))
	end = len(sentence)
	for p in ['.', '!', '?', ';', ',']:
		idx = sentence.find(p, m.end())
		if idx != -1:
			end = min(end, idx)
	text = sentence[start:end].strip()
	return [Span(text=text, start=start, end=end, conf=0.5)]


# --- ML path (optional) ------------------------------------------------------
_MODEL_NAME = "roberta-base"
_LABELS: List[str] = ["O", "B", "I"]
_NUM_LABELS = len(_LABELS)
_LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(_LABELS)}
_ID2LABEL: Dict[int, str] = {i: l for l, i in _LABEL2ID.items()}
_FINETUNED_DIR = Path("extractor/_models/claim-extractor-bio")


@dataclass
class LabeledSpan:
	start: int
	end: int


@dataclass
class FineTuneExample:
	text: str
	spans: List[LabeledSpan]


def _device():
	try:
		import torch  # type: ignore
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	except Exception:
		return None


def _tokenize_with_labels(tokenizer, text: str, spans: List[LabeledSpan]):
	enc = tokenizer(text, return_offsets_mapping=True, truncation=True, padding=False)
	offsets = enc["offset_mapping"]
	labels: List[int] = []
	def in_any_span(s: int, e: int):
		for idx, sp in enumerate(spans):
			if not (e <= sp.start or s >= sp.end):
				return True, idx
		return False, -1
	tag_started = set()
	for (s, e) in offsets:
		if s == 0 and e == 0:
			labels.append(-100)
			continue
		inside, span_idx = in_any_span(s, e)
		if not inside:
			labels.append(_LABEL2ID["O"])
			continue
		if span_idx not in tag_started:
			labels.append(_LABEL2ID["B"])
			tag_started.add(span_idx)
		else:
			labels.append(_LABEL2ID["I"])
	enc["labels"] = labels
	return enc


def _build_dataset(tokenizer, examples: Sequence[FineTuneExample]):
	from datasets import Dataset  # type: ignore
	encoded_batches = [_tokenize_with_labels(tokenizer, ex.text, ex.spans) for ex in examples]
	return Dataset.from_list(encoded_batches)


def _tiny_bootstrap_examples() -> List[FineTuneExample]:
	text = "The company fired 100 employees last month."
	start = text.find("fired 100 employees last month")
	end = start + len("fired 100 employees last month")
	return [
		FineTuneExample(text=text, spans=[LabeledSpan(start=start, end=end)]),
		FineTuneExample(text="What a nice day!", spans=[]),
	]


def _ensure_finetuned_model():
	try:
		from transformers import AutoModelForTokenClassification, AutoTokenizer  # type: ignore
	except Exception:
		return None, None

	if _FINETUNED_DIR.exists():
		try:
			tokenizer = AutoTokenizer.from_pretrained(str(_FINETUNED_DIR))
			model = AutoModelForTokenClassification.from_pretrained(str(_FINETUNED_DIR))
			return tokenizer, model
		except Exception:
			pass
	# minimal fine-tune
	from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments  # type: ignore
	tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
	model = AutoModelForTokenClassification.from_pretrained(
		_MODEL_NAME,
		num_labels=_NUM_LABELS,
		id2label=_ID2LABEL,
		label2id=_LABEL2ID,
	)
	ds = _build_dataset(tokenizer, _tiny_bootstrap_examples())
	ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])  # type: ignore
	args = TrainingArguments(output_dir="./.claim_extractor_ft", num_train_epochs=2, per_device_train_batch_size=8, learning_rate=5e-5, report_to=[], save_strategy="no")
	trainer = Trainer(model=model, args=args, train_dataset=ds)
	trainer.train()
	_FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
	model.save_pretrained(str(_FINETUNED_DIR))
	tokenizer.save_pretrained(str(_FINETUNED_DIR))
	return tokenizer, model


def extract_claim_spans(sentence: str) -> List[Span]:
	"""Extract claim spans using BIO model if available, else fallback regex."""
	# Allow disabling ML path via environment for tests/lightweight runs
	if os.getenv("TRUTHLENS_DISABLE_ML") == "1":
		return _fallback_extract(sentence)
	device = _device()
	if device is None:
		return _fallback_extract(sentence)
	try:
		from transformers import AutoTokenizer  # type: ignore
		from torch import softmax  # type: ignore
		tokenizer, model = _ensure_finetuned_model()
		if tokenizer is None or model is None:
			return _fallback_extract(sentence)
		model.to(device)
		enc = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt", truncation=True, padding=False)
		enc = {k: v.to(device) for k, v in enc.items()}
		offsets = enc["offset_mapping"][0].tolist()
		logits = model(**{k: v for k, v in enc.items() if k != "offset_mapping"}).logits
		probs = softmax(logits, dim=-1)[0]
		pred_ids = probs.argmax(dim=-1).tolist()
		spans: List[Span] = []
		current_tokens: List[int] = []
		for idx, (s, e) in enumerate(offsets):
			if s == 0 and e == 0:
				continue
			label = _ID2LABEL.get(pred_ids[idx], "O")
			if label == "B":
				if current_tokens:
					start_char = offsets[current_tokens[0]][0]
					end_char = offsets[current_tokens[-1]][1]
					conf = float(probs[current_tokens, 1:3].sum(dim=1).mean().item())
					spans.append(Span(text=sentence[start_char:end_char], start=start_char, end=end_char, conf=conf))
				current_tokens = [idx]
			elif label == "I":
				current_tokens.append(idx)
			else:
				if current_tokens:
					start_char = offsets[current_tokens[0]][0]
					end_char = offsets[current_tokens[-1]][1]
					conf = float(probs[current_tokens, 1:3].sum(dim=1).mean().item())
					spans.append(Span(text=sentence[start_char:end_char], start=start_char, end=end_char, conf=conf))
					current_tokens = []
		if current_tokens:
			start_char = offsets[current_tokens[0]][0]
			end_char = offsets[current_tokens[-1]][1]
			conf = float(probs[current_tokens, 1:3].sum(dim=1).mean().item())
			spans.append(Span(text=sentence[start_char:end_char], start=start_char, end=end_char, conf=conf))
		return spans
	except Exception:
		return _fallback_extract(sentence)
