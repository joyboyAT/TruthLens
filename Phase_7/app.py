from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Phase_7.pipeline.pipeline import run_pipeline


class VerifyRequest(BaseModel):
	text: str
	evidence: Optional[List[Dict[str, str]]] = None
	cues: Optional[List[str]] = None
	prebunk_tips: Optional[Dict[str, str]] = None


app = FastAPI(title="SatyaLens API")

# Enable simple CORS for local dev UIs
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.post("/verify")
async def verify(req: VerifyRequest) -> Dict[str, Any]:
	# The pipeline now takes only text; optional fields reserved for future wiring
	result = run_pipeline(req.text)
	return result
