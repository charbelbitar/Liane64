import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import traceback
from datetime import datetime, timezone
import json
import pathlib

from main import rag_pipeline
from cache import purge_invalid_entries

from prometheus_fastapi_instrumentator import Instrumentator

purge_invalid_entries()

app = FastAPI(title="RAG Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
    ],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


class Message(BaseModel):
    role: str     
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class Metadata(BaseModel):
    role_detecte: Optional[str] = None
    phase: Optional[str] = None
    urgence: Optional[str] = None
    language: Optional[str] = None
    niveau_langue: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    metadata: Optional[Metadata] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        # Convert Pydantic messages to plain dicts that main.py expects
        history = [{"role": m.role, "content": m.content} for m in req.history]

        answer, parsed = rag_pipeline(req.message, history)

        sources = parsed.get("sources", []) if parsed else []

        metadata = None
        if parsed:
            metadata = Metadata(
                role_detecte=parsed.get("role_detecte"),
                phase=parsed.get("phase"),
                urgence=parsed.get("urgence"),
                language=parsed.get("language"),
                niveau_langue=parsed.get("niveau_langue"),
            )

        return ChatResponse(answer=answer, sources=sources, metadata=metadata)

    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


FEEDBACK_PATH = pathlib.Path("/app/feedback/feedback.jsonl")
FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

class FeedbackRequest(BaseModel):
    rating: int
    mcq_answer: Optional[str] = None
    message_count: Optional[int] = None

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rating": req.rating,
        "mcq_answer": req.mcq_answer,
        "message_count": req.message_count,
    }
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"status": "ok"}