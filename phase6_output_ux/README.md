# Phase 6 — Output UX

Utilities to render verdicts, evidence, cue badges, prebunk tips, shareable text, and collect feedback.

## Install

Use the project venv and install requirements:

```
& .venv\Scripts\python.exe -m pip install -r requirements.txt
```

## API

- verdict_mapping.map_verdict(score) -> str
- evidence_cards.build_evidence_cards(list[dict]) -> list[dict]
- cue_badges.generate_cue_badges(list[str]) -> list[str]
- prebunk_card.build_prebunk_card(cues, tips) -> dict
- share_card.build_share_card(claim, verdict, evidence, tip) -> str
- feedback.collect_feedback(claim_id, user_choice) -> str

## Tests

```
& .venv\Scripts\python.exe -m pytest -q phase6_output_ux\tests
```

## Demos

```
& .venv\Scripts\python.exe phase6_output_ux\demo\run_verdict_demo.py
& .venv\Scripts\python.exe phase6_output_ux\demo\run_full_pipeline.py
```
