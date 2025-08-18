# SatyaLens (Phase 7)

FastAPI service integrating the TruthLens pipeline and Phase 6 output UX.

## Setup

Use the project venv and install API deps:

```
& .venv\Scripts\python.exe -m pip install -r Phase_7\requirements.txt
```

## Run API

```
& .venv\Scripts\python.exe -m uvicorn Phase_7.app:app --reload
```

POST /verify

```
{
  "text": "5G towers cause COVID-19.",
  "evidence": [{"text":"WHO confirms vaccines are safe","url":"https://pib.gov","source":"PIB"}],
  "cues": ["Fear Appeal","Conspiracy"],
  "prebunk_tips": {"Fear Appeal":"Verify with WHO/CDC.","Conspiracy":"Check transparent sources."}
}
```
