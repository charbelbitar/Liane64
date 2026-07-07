from config import collection, events_collection, services_collection
from embeddings import embed
import numpy as np
import re
from datetime import date

# Maps every stade value to the keywords a user query typically contains
_STADE_KEYWORDS: dict[str, list[str]] = {
    "grossesse":    ["grossesse", "enceinte", "enceint", "trimestre", "fœtus", "foetus",
                     "amniocentèse", "échographie", "nausée", "accouchement"],
    "nouveau_né":   ["nouveau-né", "nouveau_né", "naissance", "nouveau né", "neonatal",
                     "néonatal", "allaitement", "lait maternel", "maternité", "nourrisson"],
    "bebe":         ["bébé", "bebe", "nourrisson", "4 mois", "6 mois", "8 mois",
                     "12 mois", "diversification"],
    "enfance":      ["enfant", "enfants", "enfance", "école", "maternelle", "primaire",
                     "apprentissage", "scolarité"],
    "adolescence":  ["adolescent", "ado", "ados", "puberte", "puberté", "collège", "lycée"],
}

_RISQUE_WEIGHTS: dict[str, float] = {
    "élevé": 0.08,
    "eleve": 0.08,
    "moyen": 0.04,
    "faible": 0.0,
}

STADE_BOOST   = 0.07   # added to score when stade matches the query
KEYWORD_BOOST = 0.05   # added per matched mot-clé


# Return the most likely stade based on keywords in the query or none
def _infer_stade_from_query(query: str) -> str | None:
    q = query.lower()
    best_stade, best_count = None, 0
    for stade, keywords in _STADE_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in q)
        if count > best_count:
            best_count, best_stade = count, stade
    return best_stade if best_count > 0 else None


# Returns a score bonus in [0, ~0.20] based on stade match, mots_clés overlap, risque level
def _metadata_boost(meta: dict, query: str, inferred_stade: str | None) -> float:
    boost = 0.0
    q_lower = query.lower()

    # stade match
    chunk_stade = meta.get("stade", "")
    if inferred_stade and chunk_stade == inferred_stade:
        boost += STADE_BOOST

    # mots_clés overlap
    raw_kw = meta.get("mots_clés", meta.get("mots_cles", "[]"))
    try:
        import json as _json
        keywords: list[str] = _json.loads(raw_kw) if isinstance(raw_kw, str) else raw_kw
    except Exception:
        keywords = []
    if any(kw.lower() in q_lower for kw in keywords):
        boost += KEYWORD_BOOST  

    # risque 
    risque = meta.get("risque", "faible").lower()
    boost += _RISQUE_WEIGHTS.get(risque, 0.0)

    return boost


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


GEO_BOOST = 0.15  # added when an item's ville/cp matches one detected in the query


def _load_known_locations() -> tuple[dict[str, str], set[str]]:
    villes_lower: dict[str, str] = {}
    cps: set[str] = set()
    for coll in (events_collection, services_collection):
        try:
            data = coll.get(include=["metadatas"])
        except Exception as e:
            print(f"[GEO] Could not load metadata for gazetteer from {coll}: {e}")
            continue
        for meta in data.get("metadatas", []) or []:
            ville = (meta.get("ville") or "").strip()
            if ville:
                villes_lower[ville.lower()] = ville
            cp = str(meta.get("cp") or "").strip()
            if cp:
                cps.add(cp)
    return villes_lower, cps


def _build_ville_regex(villes_lower: dict[str, str]):
    if not villes_lower:
        return None
    return re.compile(
        r'\b(' + '|'.join(re.escape(v) for v in sorted(villes_lower, key=len, reverse=True)) + r')\b',
        re.IGNORECASE,
    )

_KNOWN_VILLES_LOWER, _KNOWN_CPS = _load_known_locations()

_VILLE_RE = _build_ville_regex(_KNOWN_VILLES_LOWER)
_CP_RE = re.compile(r'\b(\d{5})\b')


def refresh_geo_gazetteer() -> None:
    global _KNOWN_VILLES_LOWER, _KNOWN_CPS, _VILLE_RE
    _KNOWN_VILLES_LOWER, _KNOWN_CPS = _load_known_locations()
    _VILLE_RE = _build_ville_regex(_KNOWN_VILLES_LOWER)
    print(f"[GEO] Gazetteer refreshed — {len(_KNOWN_VILLES_LOWER)} villes, {len(_KNOWN_CPS)} cps")


def detect_location(query: str) -> tuple[str | None, str | None]:
    ville_match = None
    if _VILLE_RE:
        m = _VILLE_RE.search(query)
        if m:
            ville_match = _KNOWN_VILLES_LOWER.get(m.group(1).lower())

    cp_match = None
    m = _CP_RE.search(query)
    if m and m.group(1) in _KNOWN_CPS:
        cp_match = m.group(1)

    return ville_match, cp_match


def _geo_where_clause(ville: str | None, cp: str | None) -> dict | None:
    if ville and cp:
        return {"$or": [{"ville": ville}, {"cp": cp}]}
    if ville:
        return {"ville": ville}
    if cp:
        return {"cp": cp}
    return None


def _run_geo_query(coll, query_emb: list, n_results: int, where: dict | None):
    kwargs = dict(
        query_embeddings=[query_emb],
        n_results=n_results,
        include=["documents", "embeddings", "metadatas"],
    )
    if where:
        kwargs["where"] = where
    return coll.query(**kwargs)


def retrieve_and_rerank(query: str, n_results=15, candidate_count=50, metadata_filter: dict = None, query_emb: list = None):
    
    if query_emb is None:
        query_emb = embed(query)
    elif not isinstance(query_emb, list):
        query_emb = query_emb.tolist() 

    query_kwargs = dict(
        query_embeddings=[query_emb],
        n_results=candidate_count,
        include=["documents", "embeddings", "metadatas"]
    )
    if metadata_filter:
        query_kwargs["where"] = metadata_filter
 
    try:
        results = collection.query(**query_kwargs)
    except Exception as e:
        print(f"[RERANK] Query error: {e}")
        return []
 
    docs       = results["documents"][0]    
    embeddings = results["embeddings"][0]     
    metadatas  = results["metadatas"][0]       
 
    if not docs:
        return []
 
    inferred_stade = _infer_stade_from_query(query)
    if inferred_stade:
        print(f"  [RERANK] inferred_stade='{inferred_stade}'")

    scored = [
        {
            "content":  doc,
            "metadata": meta,
            "score":    cosine_similarity(query_emb, emb)
                        + _metadata_boost(meta, query, inferred_stade)
        }
        for doc, emb, meta in zip(docs, embeddings, metadatas)
    ]
 
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:n_results]
 
    for i, item in enumerate(top):
        src = item["metadata"].get("source", "?")
        print(f"  [RERANK #{i+1}] score={item['score']:.3f}  source={src}")
 
    return top
 
 
def _is_upcoming(meta: dict) -> bool:
    raw_date = (meta.get("date_evenement") or "").strip()
    if not raw_date:
        return True
    try:
        event_date = date.fromisoformat(raw_date[:10])
    except ValueError:
        return True
    return event_date >= date.today()


def retrieve_events(query: str, n_results=3, score_threshold=0.50, query_emb: list = None):
    if query_emb is None:
        query_emb = embed(query)
    elif not isinstance(query_emb, list):
        query_emb = query_emb.tolist()

    ville, cp = detect_location(query)
    where = _geo_where_clause(ville, cp)
    if where:
        print(f"[EVENTS] Detected location in query — ville={ville!r} cp={cp!r}")

    try:
        results = _run_geo_query(events_collection, query_emb, min(n_results * 6, 25), where)
        if where and not results["documents"][0]:
            print(f"[EVENTS] No events for location filter {where} — falling back to soft boost")
            results = _run_geo_query(events_collection, query_emb, min(n_results * 6, 25), None)
            where = None
    except Exception as e:
        print(f"[EVENTS] Query error: {e}")
        return []
 
    docs       = results["documents"][0]
    embeddings = results["embeddings"][0]
    metadatas  = results["metadatas"][0]
 
    if not docs:
        return []

    upcoming = [
        (doc, emb, meta) for doc, emb, meta in zip(docs, embeddings, metadatas)
        if _is_upcoming(meta)
    ]
    n_dropped = len(docs) - len(upcoming)
    if n_dropped:
        print(f"  [EVENTS] Dropped {n_dropped} past event(s)")
    if not upcoming:
        return []
    docs, embeddings, metadatas = zip(*upcoming)

 
    scored = []
    for doc, emb, meta in zip(docs, embeddings, metadatas):
        score = cosine_similarity(query_emb, emb)
        
        if not where and (ville or cp):
            if ville and meta.get("ville", "").lower() == ville.lower():
                score += GEO_BOOST
            elif cp and str(meta.get("cp", "")) == cp:
                score += GEO_BOOST
        scored.append({"text": doc, "metadata": meta, "score": score})
 
    scored.sort(key=lambda x: x["score"], reverse=True)
 
    filtered = [e for e in scored if e["score"] >= score_threshold][:n_results]
 
    for i, item in enumerate(filtered):
        print(f"  [EVENTS #{i+1}] score={item['score']:.3f}  nom='{item['metadata'].get('nom_evenement', '?')}'")
 
    if not filtered:
        print(f"  [EVENTS] No events above threshold {score_threshold}")
 
    return filtered
 
 
def retrieve_services(query: str, n_results=3, score_threshold=0.45, query_emb: list = None):
    if query_emb is None:
        query_emb = embed(query)
    elif not isinstance(query_emb, list):
        query_emb = query_emb.tolist()

    ville, cp = detect_location(query)
    where = _geo_where_clause(ville, cp)
    if where:
        print(f"[SERVICES] Detected location in query — ville={ville!r} cp={cp!r}")

    try:
        results = _run_geo_query(services_collection, query_emb, min(n_results * 6, 25), where)
        if where and not results["documents"][0]:
            print(f"[SERVICES] No services for location filter {where} — falling back to soft boost")
            results = _run_geo_query(services_collection, query_emb, min(n_results * 6, 25), None)
            where = None
    except Exception as e:
        print(f"[SERVICES] Query error: {e}")
        return []
 
    docs       = results["documents"][0]
    embeddings = results["embeddings"][0]
    metadatas  = results["metadatas"][0]
 
    if not docs:
        return []
 
    scored = []
    for doc, emb, meta in zip(docs, embeddings, metadatas):
        score = cosine_similarity(query_emb, emb)
        if not where and (ville or cp):
            if ville and meta.get("ville", "").lower() == ville.lower():
                score += GEO_BOOST
            elif cp and str(meta.get("cp", "")) == cp:
                score += GEO_BOOST
        scored.append({"text": doc, "metadata": meta, "score": score})
 
    scored.sort(key=lambda x: x["score"], reverse=True)
 
    filtered = [s for s in scored if s["score"] >= score_threshold][:n_results]
 
    for i, item in enumerate(filtered):
        nom = item["metadata"].get("nom", item["metadata"].get("qui", "?"))
        print(f"  [SERVICES #{i+1}] score={item['score']:.3f}  nom='{nom}'")
 
    if not filtered:
        print(f"  [SERVICES] No services above threshold {score_threshold}")
 
    return filtered