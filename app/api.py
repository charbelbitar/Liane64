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

FEEDBACK_PATH = pathlib.Path("/app/feedback/feedback.jsonl")
FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)


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

class EventItem(BaseModel):
    nom: Optional[str] = None
    date: Optional[str] = None
    adresse: Optional[str] = None
    ville: Optional[str] = None
    sujet: Optional[str] = None
    lien_inscription: Optional[str] = None
    structure_nom: Optional[str] = None
    public_enfants: Optional[bool] = None
    public_age_minimum: Optional[int] = None
    public_age_maximum: Optional[int] = None
    public_parents: Optional[bool] = None
    public_futurs_parents: Optional[bool] = None

class ServiceItem(BaseModel):
    nom: Optional[str] = None
    type_service: Optional[str] = None
    adresse: Optional[str] = None
    ville: Optional[str] = None
    telephone: Optional[str] = None
    email: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    metadata: Optional[Metadata] = None
    events: List[EventItem] = []
    services: List[ServiceItem] = []

class FeedbackRequest(BaseModel):
    rating: int
    mcq_answer: Optional[str] = None
    message_count: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        # Convert Pydantic messages to plain dicts that main.py expects
        history = [{"role": m.role, "content": m.content} for m in req.history]

        answer, parsed, raw_events, raw_services = rag_pipeline(req.message, history)

        sources = parsed.get("sources", []) if parsed else []

        VALID_LANGUAGES = {"francais", "basque", "occitan", "anglais", "espagnol"}

        metadata = None
        if parsed:
            language = parsed.get("language", "francais")
            if language not in VALID_LANGUAGES:
                print(f"[LANG] Invalid language value '{language}' — defaulting to 'francais'")
                language = "francais"
            metadata = Metadata(
                role_detecte=parsed.get("role_detecte"),
                phase=parsed.get("phase"),
                urgence=parsed.get("urgence"),
                language=language,
                niveau_langue=parsed.get("niveau_langue"),
            )

        events = []
        for e in (raw_events or []):
            m = e.get("metadata", {})
            events.append(EventItem(
            nom=m.get("nom_evenement"),
            date=m.get("date_evenement"),
            adresse=m.get("adresse"),
            ville=m.get("ville"),
            sujet=m.get("sujet"),
            lien_inscription=m.get("lien_inscription") or None,
            structure_nom=m.get("structure_nom") or None,
            public_enfants=m.get("public_enfants"),
            public_age_minimum=m.get("public_age_minimum"),
            public_age_maximum=m.get("public_age_maximum"),
            public_parents=m.get("public_parents"),
            public_futurs_parents=m.get("public_futurs_parents"),
        ))

        services = []
        for s in (raw_services or []):
            m = s.get("metadata", {})
            nom = m.get("nom") or m.get("qui") or m.get("structure_nom")
            services.append(ServiceItem(
                nom=nom,
                type_service=m.get("type") or m.get("quoi"),
                adresse=m.get("adresse"),
                ville=m.get("ville"),
                telephone=m.get("telephone"),
                email=m.get("email") if m.get("email") and m.get("email") != "/" else None,
            ))

        return ChatResponse(
            answer=answer,
            sources=sources,
            metadata=metadata,
            events=events,
            services=services,
        )

    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


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
