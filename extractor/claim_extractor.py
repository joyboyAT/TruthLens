from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, TypedDict

import torch
from datasets import Dataset
from transformers import (
	AutoModelForTokenClassification,
	AutoTokenizer,
	Trainer,
	TrainingArguments,
)


_MODEL_NAME = "roberta-base"
_LABELS: List[str] = ["O", "B", "I"]
_NUM_LABELS = len(_LABELS)
_LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(_LABELS)}
_ID2LABEL: Dict[int, str] = {i: l for l, i in _LABEL2ID.items()}
_FINETUNED_DIR = Path("extractor/_models/claim-extractor-bio")


class Span(TypedDict):
	text: str
	start: int
	end: int
	conf: float


@dataclass
class LabeledSpan:
	start: int
	end: int


@dataclass
class FineTuneExample:
	text: str
	spans: List[LabeledSpan]  # character-level [start, end) spans for claims


def _device() -> torch.device:
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tokenize_with_labels(
	tokenizer: AutoTokenizer,
	text: str,
	spans: List[LabeledSpan],
):
	enc = tokenizer(
		text,
		return_offsets_mapping=True,
		truncation=True,
		padding=False,
	)
	offsets = enc["offset_mapping"]
	labels: List[int] = []

	def in_any_span(s: int, e: int) -> Tuple[bool, int]:
		for idx, sp in enumerate(spans):
			if e <= sp.start or s >= sp.end:
				continue
			return True, idx
		return False, -1

	tag_started = set()
	for i, (s, e) in enumerate(offsets):
		# Special tokens may have (0, 0); label as -100 for loss masking
		if s == 0 and e == 0:
			labels.append(-100)
			continue
		inside, span_idx = in_any_span(s, e)
		if not inside:
			labels.append(_LABEL2ID["O"])
			continue
		# Begin tag if first token touching this span, else Inside
		if span_idx not in tag_started:
			labels.append(_LABEL2ID["B"])
			tag_started.add(span_idx)
		else:
			labels.append(_LABEL2ID["I"])

	enc["labels"] = labels
	return enc


def _build_dataset(tokenizer: AutoTokenizer, examples: Sequence[FineTuneExample]) -> Dataset:
	encoded_batches = [
		_tokenize_with_labels(tokenizer, ex.text, ex.spans) for ex in examples
	]
	# Collate lists of fields into a Dataset
	return Dataset.from_list(encoded_batches)


def _tiny_bootstrap_examples() -> List[FineTuneExample]:
	text = "The company fired 100 employees last month."
	start = text.find("fired 100 employees last month")
	end = start + len("fired 100 employees last month")
	return [
		FineTuneExample(text=text, spans=[LabeledSpan(start=start, end=end)]),
		FineTuneExample(text="What a nice day!", spans=[]),
		FineTuneExample(
			text="GDP rose by 2% in 2020.",
			spans=[LabeledSpan(start=0, end=len("GDP rose by 2% in 2020"))],
		),
	]


def _ensure_finetuned_model() -> Tuple[AutoTokenizer, AutoModelForTokenClassification]:
	if _FINETUNED_DIR.exists():
		try:
			tokenizer = AutoTokenizer.from_pretrained(str(_FINETUNED_DIR))
			model = AutoModelForTokenClassification.from_pretrained(
				str(_FINETUNED_DIR)
			)
			return tokenizer, model
		except Exception:
			pass

	tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
	model = AutoModelForTokenClassification.from_pretrained(
		_MODEL_NAME,
		num_labels=_NUM_LABELS,
		id2label=_ID2LABEL,
		label2id=_LABEL2ID,
	)

	ds = _build_dataset(tokenizer, _tiny_bootstrap_examples())
	ds.set_format(
		type="torch",
		columns=["input_ids", "attention_mask", "labels"],
	)

	args = TrainingArguments(
		output_dir="./.claim_extractor_ft",
		num_train_epochs=3,
		per_device_train_batch_size=8,
		learning_rate=5e-5,
		weight_decay=0.01,
		report_to=[],
		save_strategy="no",
		logging_steps=50,
	)

	trainer = Trainer(model=model, args=args, train_dataset=ds)
	trainer.train()

	_FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
	model.save_pretrained(str(_FINETUNED_DIR))
	tokenizer.save_pretrained(str(_FINETUNED_DIR))
	return tokenizer, model


@torch.inference_mode()
def extract_claim_spans(sentence: str) -> List[Span]:
	"""Extract claim spans using a BIO tagger.

	Returns a list of {text, start, end, conf}.
	"""
	tokenizer, model = _ensure_finetuned_model()
	model.to(_device())

	enc = tokenizer(
		sentence,
		return_offsets_mapping=True,
		return_tensors="pt",
		truncation=True,
		padding=False,
	)
	enc = {k: v.to(_device()) for k, v in enc.items()}
	offsets = enc["offset_mapping"][0].tolist()

	logits = model(**{k: v for k, v in enc.items() if k != "offset_mapping"}).logits
	probs = torch.softmax(logits, dim=-1)[0]  # [seq_len, num_labels]
	pred_ids = probs.argmax(dim=-1).tolist()

	spans: List[Span] = []
	current_tokens: List[int] = []
	for idx, (s, e) in enumerate(offsets):
		if s == 0 and e == 0:
			continue
		label = _ID2LABEL.get(pred_ids[idx], "O")
		if label == "B":
			# close previous
			if current_tokens:
				start_char = offsets[current_tokens[0]][0]
				end_char = offsets[current_tokens[-1]][1]
				span_probs = [float((probs[i, _LABEL2ID['B']] + probs[i, _LABEL2ID['I']]).item()) for i in current_tokens]
				conf = sum(span_probs) / max(len(span_probs), 1)
				spans.append(
					Span(text=sentence[start_char:end_char], start=start_char, end=end_char, conf=conf)
				)
			current_tokens = [idx]
		elif label == "I":
			if current_tokens:
				current_tokens.append(idx)
			else:
				current_tokens = [idx]
		else:  # "O"
			if current_tokens:
				start_char = offsets[current_tokens[0]][0]
				end_char = offsets[current_tokens[-1]][1]
				span_probs = [float((probs[i, _LABEL2ID['B']] + probs[i, _LABEL2ID['I']]).item()) for i in current_tokens]
				conf = sum(span_probs) / max(len(span_probs), 1)
				spans.append(
					Span(text=sentence[start_char:end_char], start=start_char, end=end_char, conf=conf)
				)
				current_tokens = []

	# flush
	if current_tokens:
		start_char = offsets[current_tokens[0]][0]
		end_char = offsets[current_tokens[-1]][1]
		span_probs = [float((probs[i, _LABEL2ID['B']] + probs[i, _LABEL2ID['I']]).item()) for i in current_tokens]
		conf = sum(span_probs) / max(len(span_probs), 1)
		spans.append(Span(text=sentence[start_char:end_char], start=start_char, end=end_char, conf=conf))

	return spans
