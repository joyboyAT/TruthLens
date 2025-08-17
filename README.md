# TruthLens — Phase 1

Phase 1 establishes scaffolding for data collection and preprocessing.

Project structure:
- src/
  - __init__.py
  - data_collection.py
  - preprocessing.py
- data/
  - raw/
  - processed/
- notebooks/
- requirements.txt

Getting started (PowerShell):
1. python -m venv .venv
2. .venv\Scripts\Activate.ps1
3. pip install -r requirements.txt

Usage:
- from src.data_collection import collect_from_urls
- from src.preprocessing import build_dataframe, html_to_text

Notes:
- data/raw/ and data/processed/ include placeholders to keep them in version control.
