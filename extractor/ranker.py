from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import List

import numpy as np

try:
	from xgboost import XGBClassifier
	xgb_available = True
except Exception:  # pragma: no cover
	xgb_available = False


_MODEL_DIR = Path("extractor/_models")
_MODEL_PATH = _MODEL_DIR / "claim-ranker.json"

_CAUSAL_PATTERNS = [
	r"\bcauses?\b",
	r"\bcaused\b",
	r"\bleads?\s+to\b",
	r"\bled\s+to\b",
	r"\bresults?\s+in\b",
	r"\btrigger(s|ed)?\b",
]

_NEGATION_PATTERNS = [r"\bno\b", r"\bnot\b", r"n't\b", r"\bnever\b"]

_KEYWORDS = {
	"medical": ["covid", "vaccine", "vaccines", "virus", "flu", "disease", "illness"],
	"technology": ["5g", "tower", "towers", "radiation", "chip"],
	"misinfo": ["flat earth", "chemtrails", "microchip", "hoax", "plandemic"],
}

_STOPWORDS = {
	"a",
	"an",
	"the",
	"is",
	"are",
	"was",
	"were",
	"be",
	"to",
	"of",
	"and",
	"in",
	"that",
	"it",
	"on",
	"for",
	"as",
}


def _normalize(text: str) -> str:
	return re.sub(r"\s+", " ", text or "").strip()


def _tokenize(text: str) -> List[str]:
	return re.findall(r"[A-Za-z0-9%]+|[^\w\s]", text)


def _count_regex(text: str, pattern: str) -> int:
	return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _feature_vector(text: str) -> np.ndarray:
	text = _normalize(text)
	lower = text.lower()
	tokens = _tokenize(text)
	alpha_tokens = [t for t in tokens if re.search(r"[A-Za-z]", t)]
	length_chars = len(text)
	n_tokens = len(tokens)
	n_alpha = len(alpha_tokens)
	n_digits = sum(ch.isdigit() for ch in text)
	has_number = 1.0 if re.search(r"\d", text) else 0.0
	has_percent = 1.0 if "%" in text or re.search(r"\bpercent\b", lower) else 0.0
	has_year = 1.0 if re.search(r"\b(19|20)\d{2}\b", lower) else 0.0
	has_causal = 1.0 if any(re.search(p, lower) for p in _CAUSAL_PATTERNS) else 0.0
	n_neg = sum(1 for p in _NEGATION_PATTERNS if re.search(p, lower))
	has_exclaim = 1.0 if "!" in text else 0.0
	has_question = 1.0 if "?" in text else 0.0
	titlecase_frac = 0.0
	if n_alpha:
		titlecase_frac = sum(1 for t in alpha_tokens if t[:1].isupper()) / max(1, n_alpha)
	uppercase_frac = 0.0
	if n_alpha:
		uppercase_frac = sum(1 for t in alpha_tokens if t.isupper()) / max(1, n_alpha)
	stop_frac = 0.0
	if n_alpha:
		stop_frac = sum(1 for t in alpha_tokens if t.lower() in _STOPWORDS) / max(1, n_alpha)

	# Domain keywords
	has_med = 1.0 if any(k in lower for k in _KEYWORDS["medical"]) else 0.0
	has_tech = 1.0 if any(k in lower for k in _KEYWORDS["technology"]) else 0.0
	has_misinfo = 1.0 if any(k in lower for k in _KEYWORDS["misinfo"]) else 0.0

	features = np.array(
		[
			length_chars,
			n_tokens,
			n_digits,
			has_number,
			has_percent,
			has_year,
			has_causal,
			n_neg,
			has_exclaim,
			has_question,
			titlecase_frac,
			uppercase_frac,
			stop_frac,
			has_med,
			has_tech,
			has_misinfo,
		],
		dtype=float,
	)
	# Basic log scaling for count-like features to keep ranges stable
	features[0] = math.log1p(features[0])  # length_chars
	features[1] = math.log1p(features[1])  # n_tokens
	features[2] = math.log1p(features[2])  # n_digits
	return features


def _bootstrap_training_data():
	pos = [
		"5G towers cause COVID-19.",
		"Vaccines cause autism.",
		"Radiation from 5G towers leads to illness.",
		"Chemtrails result in disease.",
		"The earth is flat.",
	]
	neg = [
		"I love pizza.",
		"What a nice day!",
		"Let's meet tomorrow?",
		"I enjoy coding in Python.",
		"This is a great movie.",
	]
	X = np.vstack([_feature_vector(t) for t in pos + neg])
	y = np.array([1] * len(pos) + [0] * len(neg), dtype=int)
	return X, y


def _train_and_save(model_path: Path):
	X, y = _bootstrap_training_data()
	if not xgb_available:
		# Nothing to save; heuristic fallback will be used
		return None
	clf = XGBClassifier(
		objective="binary:logistic",
		n_estimators=300,
		max_depth=3,
		learning_rate=0.1,
		subsample=0.9,
		colsample_bytree=0.9,
		eval_metric="logloss",
		random_state=42,
	)
	clf.fit(X, y)
	_MODEL_DIR.mkdir(parents=True, exist_ok=True)
	clf.save_model(str(model_path))
	return clf


def _load_model(model_path: Path):
	if not xgb_available:
		return None
	clf = XGBClassifier()
	clf.load_model(str(model_path))
	return clf


def _ensure_model():
	if _MODEL_PATH.exists():
		try:
			return _load_model(_MODEL_PATH)
		except Exception:
			pass
	return _train_and_save(_MODEL_PATH)


def _heuristic_score(text: str) -> float:
	# Lightweight score in case XGBoost is unavailable
	f = _feature_vector(text)
	score = 0.0
	# Weight cues
	score += 0.25 * f[6]  # has_causal
	score += 0.10 * f[5]  # has_year
	score += 0.20 * f[13]  # has_med
	score += 0.15 * f[14]  # has_tech
	score += 0.20 * f[15]  # has_misinfo
	score -= 0.10 * f[9]   # question
	score = max(0.0, min(1.0, score))
	return float(score)


def score_claim(claim: str) -> float:
	"""Score how "claim-like" a text is in [0,1] using engineered features + XGBoost.

	On first run, a tiny bootstrap model is trained and cached under
	`extractor/_models/claim-ranker.json`.
	"""
	features = _feature_vector(claim)
	model = _ensure_model()
	if model is None:
		return _heuristic_score(claim)
	proba = model.predict_proba(features.reshape(1, -1))[0, 1]
	# Clamp to [0,1] just in case
	return float(max(0.0, min(1.0, proba)))
