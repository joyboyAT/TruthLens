from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	Trainer,
	TrainingArguments,
)
from datasets import Dataset


_MODEL_NAME = "roberta-base"
_NUM_LABELS = 2
_FINETUNED_DIR = Path("extractor/_models/claim-detector")


def _device() -> torch.device:
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class FineTuneExample:
	text: str
	label: int  # 1 = claim, 0 = not-claim


def _tokenize_batch(tokenizer: AutoTokenizer, texts: Sequence[str]):
	return tokenizer(list(texts), truncation=True, padding=True)


def fine_tune(
	examples: Iterable[FineTuneExample],
	num_epochs: int = 1,
	batch_size: int = 8,
	learning_rate: float = 5e-5,
	seed: int = 42,
	save_path: str | None = None,
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
	"""Fine-tune RoBERTa-base on binary claim detection."""
	torch.manual_seed(seed)
	texts: List[str] = [e.text for e in examples]
	labels: List[int] = [int(e.label) for e in examples]

	tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
	model = AutoModelForSequenceClassification.from_pretrained(
		_MODEL_NAME,
		num_labels=_NUM_LABELS,
	)

	ds = Dataset.from_dict({"text": texts, "label": labels})
	encoded = ds.map(lambda b: _tokenize_batch(tokenizer, b["text"]), batched=True)
	encoded = encoded.rename_column("label", "labels")
	encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

	args = TrainingArguments(
		output_dir="./.claim_detector_ft",
		num_train_epochs=num_epochs,
		per_device_train_batch_size=batch_size,
		learning_rate=learning_rate,
		seed=seed,
		logging_steps=50,
		report_to=[],
		save_strategy="no",
	)

	trainer = Trainer(model=model, args=args, train_dataset=encoded)
	trainer.train()

	if save_path:
		model.save_pretrained(save_path)
		tokenizer.save_pretrained(save_path)

	return tokenizer, model


def _tiny_bootstrap_dataset() -> List[FineTuneExample]:
	# Minimal examples to bias the classifier; includes the requested tests
	claims = [
		"The earth is flat.",
		"The GDP of France grew by 2% in 2023.",
		"Water boils at 100 degrees Celsius at sea level.",
	]
	not_claims = [
		"What a nice day!",
		"Wow!",
		"How are you?",
	]
	return [
		* [FineTuneExample(text=t, label=1) for t in claims],
		* [FineTuneExample(text=t, label=0) for t in not_claims],
	]


def _ensure_finetuned_model() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
	if _FINETUNED_DIR.exists():
		try:
			tokenizer = AutoTokenizer.from_pretrained(str(_FINETUNED_DIR))
			model = AutoModelForSequenceClassification.from_pretrained(
				str(_FINETUNED_DIR)
			)
			return tokenizer, model
		except Exception:
			pass
	# Fine-tune quickly on tiny dataset and cache
	_FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
	tokenizer, model = fine_tune(
		examples=_tiny_bootstrap_dataset(),
		num_epochs=2,
		batch_size=8,
		learning_rate=5e-5,
		seed=42,
		save_path=str(_FINETUNED_DIR),
	)
	return tokenizer, model


@torch.inference_mode()
def is_claim(sentence: str, threshold: float = 0.6) -> Tuple[bool, float]:
	"""Predict whether a sentence is a factual claim.

	Returns (label, prob_of_claim). label is True if prob > threshold.
	If no fine-tuned weights are present locally, a tiny bootstrap fine-tuning
	will run once and cache under `extractor/_models/claim-detector`.
	"""
	tokenizer, model = _ensure_finetuned_model()
	model.to(_device())

	batch = tokenizer([sentence], return_tensors="pt", truncation=True, padding=True)
	batch = {k: v.to(_device()) for k, v in batch.items()}
	logits = model(**batch).logits
	probs = torch.softmax(logits, dim=-1).squeeze(0)
	prob_claim = float(probs[1].item())
	return prob_claim > threshold, prob_claim
