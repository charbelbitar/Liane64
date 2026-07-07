import numpy as np
import re
from config import client, chroma_client, REFUSAL_RESPONSES, is_refusal
import uuid
import json
import pathlib
from embeddings import embed
from metrics import CACHE_HITS, CACHE_MISSES

# Create or get cache collection
cache_collection = chroma_client.get_or_create_collection(
    name="semantic_cache_BGE",
    metadata={"hnsw:space": "cosine"}
)

SIMILARITY_THRESHOLD = 0.82
EXACT_DUPLICATE_THRESHOLD = 0.97

# A first-name gazetteer, lowercased, loaded once at import time.
# Source: INSEE's public "Liste de tous les prénoms" dataset (open data, not copyrighted —
# it's a factual list of legal first names). Download once, store as prenoms_fr.csv
# next to this file with one name per line.
_PRENOMS_PATH = pathlib.Path(__file__).parent / "prenoms_fr.csv"

def _load_prenoms() -> frozenset[str]:
    if not _PRENOMS_PATH.exists():
        return frozenset()
    with open(_PRENOMS_PATH, encoding="utf-8") as f:
        return frozenset(line.strip().lower() for line in f if line.strip())

_KNOWN_PRENOMS = _load_prenoms()

# Relaxed: name no longer needs to be capitalized — the relational/title
# context word ("mon mari", "Dr.", etc.) is the real signal here.
_NAME_PREFIX_RE = re.compile(
    r"""
    (?:
        mme\.?                  |
        m\.?\s*(?:me|onsieur)?  |
        dr\.?  | docteur        |
        pr\.?  | professeur     |
        sage[-\s]femme          |
        infirmier(?:e)?         |
        mon\s+(?:mari|médecin|docteur|fils|frère|père|beau[-\s]père)  |
        ma\s+(?:femme|épouse|fille|sœur|mère|belle[-\s]mère|sage[-\s]femme) |
        son\s+(?:mari|médecin|docteur|fils|frère|père) |
        sa\s+(?:femme|fille|sœur|mère)
    )\s+
    ([A-ZÀÂÉÈÊËÎÏÔÙÛÜ][a-zàâéèêëîïôùûü]+(?:\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ][a-zàâéèêëîïôùûü]+)*)
    """,
    re.VERBOSE | re.IGNORECASE,
)
 
 # Bare full names with NO context: keep the capitalized pattern as one path...
_FULL_NAME_RE = re.compile(
    r'\b([A-ZÀÂÉÈÊËÎÏÔÙÛÜ][a-zàâéèêëîïôùûü]{2,}'
    r'(?:\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ][a-zàâéèêëîïôùûü]{2,})+)\b'
)
 
# Shared word pattern used by the gazetteer-based (lowercase) checks below.
_WORD_RE = r"[a-zàâéèêëîïôùûü]+(?:-[a-zàâéèêëîïôùûü]+)*"
 
_SIMPLE_PII = [
    (re.compile(r'\b0[1-9](?:[\s.\-]?\d{2}){4}\b'),                                   "[TEL]"),
    (re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', re.IGNORECASE),                    "[EMAIL]"),
    (re.compile(r'\b(?:né(?:e)?\s+(?:le|en)\s+)?\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b'), "[DATE]"),
]
 
# If any of these remain after scrubbing, the query is still too identifying
_RESIDUAL_PII_RE = re.compile(r'\[(?:NOM|TEL|EMAIL|DATE)\]')
 

# Prénoms that are common nouns in everyday/medical French — excluded from the
# standalone-token check to avoid over-redacting ordinary words.
# (They're still caught by _NAME_PREFIX_RE and the pair-based check, where
# context disambiguates them — e.g. "mon mari Pierre" still gets redacted.)
_AMBIGUOUS_PRENOMS = frozenset({
    "pierre", "rose", "marguerite", "iris", "lys", "lou", "olive",
    "violette", "perle", "jade", "capucine", "églantine",
})

# Function words / short tokens that should never be treated as a name
# even if they happen to coincide with a rare prénom.
_MIN_STANDALONE_NAME_LEN = 3

_STANDALONE_TOKEN_RE = re.compile(rf'\b({_WORD_RE})\b')


def _scrub_standalone_first_names(text: str) -> str:
    if not _KNOWN_PRENOMS:
        return text

    def _replace(m: re.Match) -> str:
        token = m.group(1)
        low = token.lower()
        if (
            low in _KNOWN_PRENOMS
            and low not in _AMBIGUOUS_PRENOMS
            and len(low) >= _MIN_STANDALONE_NAME_LEN
        ):
            return "[NOM]"
        return token

    return _STANDALONE_TOKEN_RE.sub(_replace, text)


def scrub_pii(text: str) -> str:

    def _replace_named(m: re.Match) -> str:
        full = m.group(0)
        name = m.group(1)
        return full[: full.index(name)] + "[NOM]"
 
    text = _NAME_PREFIX_RE.sub(_replace_named, text)
    text = _FULL_NAME_RE.sub("[NOM]", text)
    text = _scrub_standalone_first_names(text)  # single token, e.g. "lucas pleure"
 
    for pattern, placeholder in _SIMPLE_PII:
        text = pattern.sub(placeholder, text)
 
    return text.strip()
 
 
def has_residual_pii(text: str) -> bool:
    return bool(_RESIDUAL_PII_RE.search(text))
 
 


def embed_query(query: str, verbose: bool = False) -> list:
    # emb = embedding_model.encode(query).tolist()
    emb = embed(query)
    if verbose:
        arr = np.array(emb)
        print(f"[EMBEDDING] dim={len(arr)}, mean={arr.mean():.4f}, "
              f"std={arr.std():.4f}, norm={np.linalg.norm(arr):.4f}")
        print(f"[EMBEDDING] first 8 dims: {np.round(arr[:8], 4).tolist()}")
    return emb
    

def get_cached_answer(query: str) -> str | None:
    if cache_collection.count() == 0:
        print("[CACHE] Collection is empty, skipping lookup")
        return None

    # query_emb = embed_query(query)
    scrubbed = scrub_pii(query)
    query_emb = embed_query(scrubbed)
 
    results = cache_collection.query(
        query_embeddings=[query_emb],
        n_results=1,
        include=["documents", "metadatas", "distances"]
    )
 
    if not results["documents"] or not results["documents"][0]:
        return None
 
    distance   = results["distances"][0][0]
    similarity = 1 - distance
    metadata   = results["metadatas"][0][0]
 
    print(f"[CACHE] best_similarity={similarity:.3f}, "
          f"cached_query='{metadata.get('query', '')}'")
 
    if similarity < 0:
        print(f"[CACHE] Negative similarity ({similarity:.3f}) — possible corrupted entry, skipping")
        return None

    if similarity < SIMILARITY_THRESHOLD:
        print("[CACHE MISS]")
        CACHE_MISSES.inc()
        return None
       
 
    print("[CACHE HIT]")
    CACHE_HITS.inc()
    return results["documents"][0][0]


def add_to_cache(query: str, answer: str) -> None:
    # Validate answer is proper structured JSON before caching
    try:
        parsed = json.loads(answer)
        if not isinstance(parsed, dict) or "reponse" not in parsed:
            print("[CACHE SKIP] Answer is not a valid structured response")
            return
    except (json.JSONDecodeError, TypeError):
        print("[CACHE SKIP] Answer is not JSON — refusing to cache plain text")
        return
        
    scrubbed = scrub_pii(query)
 
    if has_residual_pii(scrubbed):
        print("[CACHE SKIP] Identifier detected in query — not caching")
        return
 
    query_emb = embed_query(scrubbed)
 
    # Skip near-duplicates
    if cache_collection.count() > 0:
        existing = cache_collection.query(
            query_embeddings=[query_emb],
            n_results=1,
            include=["distances"]
        )
        if existing["distances"] and existing["distances"][0]:
            if 1 - existing["distances"][0][0] >= EXACT_DUPLICATE_THRESHOLD:
                print("[CACHE SKIP] Nearly identical entry exists")
                return
 
    cache_collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[query_emb],
        documents=[answer],
        metadatas=[{"query": scrubbed, "type": "qa_cache"}]
    )
    print(f"[CACHE ADD] query='{scrubbed[:80]}...'")


def purge_invalid_entries() -> None:
    all_entries = cache_collection.get(include=["documents", "metadatas"])
    ids_to_delete = []

    for doc_id, doc, meta in zip(
        all_entries["ids"],
        all_entries["documents"],
        all_entries["metadatas"]
    ):
        reason = None

        if not doc or len(doc.strip()) < 10:
            reason = "empty/malformed answer"
        elif doc.strip().startswith("<html") or "Gateway Time-out" in doc:
            reason = "HTML error page"
        else:
            # Check for legacy plain-text or non-standard JSON structure
            try:
                inner = json.loads(doc)
                if not isinstance(inner, dict) or "reponse" not in inner:
                    reason = "non-standard JSON structure (legacy)"
                else:
                    reponse = inner.get("reponse", "")
                    if is_refusal(reponse):
                        reason = "cached refusal response"
            except (json.JSONDecodeError, TypeError):
                reason = "plain-text entry (pre-JSON format)"

        if reason is None and has_residual_pii(meta.get("query", "")):
            reason = "residual PII in query key (legacy entry)"

        if reason:
            print(f"[CACHE PURGE] {reason} — "
                  f"id={doc_id}, query='{meta.get('query', '')[:60]}'")
            ids_to_delete.append(doc_id)

    if ids_to_delete:
        cache_collection.delete(ids=ids_to_delete)
        print(f"[CACHE PURGE] Removed {len(ids_to_delete)} entries")
    else:
        print("[CACHE PURGE] No invalid entries found")