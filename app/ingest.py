import os
import sys
import json
import argparse
from pathlib import Path
 
import chromadb
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
MAIN_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "MPJ_MPEDIA_PAPOTO_CAF_BGE")
EVENTS_COLLECTION_NAME = "events_caf64"
SERVICES_COLLECTION_NAME = "servicess"
 
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
BATCH_SIZE = 500  # upsert in chunks rather than one giant request
 
 
# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] {path}:{line_num} skipped (bad JSON): {e}")
 
 
def find_one(dir_path: Path, pattern: str) -> Path:
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    matches = sorted(dir_path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {dir_path}")
    if len(matches) > 1:
        print(f"  [WARN] multiple matches for '{pattern}' in {dir_path}, using {matches[0].name}")
    return matches[0]
 
 
def sanitize_metadata(meta: dict, extra: dict | None = None) -> dict:
    """Chroma metadata values must be str/int/float/bool (no lists/dicts/None).
    Lists/dicts get JSON-encoded — this is exactly what main.py expects for
    'mots_clés', since it does json.loads() on that field when building
    chunk_meta_summary."""
    clean = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (list, dict)):
            clean[k] = json.dumps(v, ensure_ascii=False)
        else:
            clean[k] = v
    if extra:
        clean.update(extra)
    return clean
 
 
def load_records(paths: list[Path]):
    ids, docs, embeddings, metadatas = [], [], [], []
    seen_ids = set()
    for path in paths:
        count = 0
        for rec in load_jsonl(path):
            rid = rec.get("id")
            embedding = rec.get("embedding")
            if rid is None or embedding is None:
                print(f"  [WARN] {path}: skipping record missing id/embedding")
                continue
            if rid in seen_ids:
                print(f"  [WARN] {path}: duplicate id '{rid}' — later one wins (upsert)")
            seen_ids.add(rid)
 
            meta = sanitize_metadata(rec.get("metadata", {}), extra={"doc_type": rec.get("type", "")})
            ids.append(rid)
            docs.append(rec.get("text", ""))
            embeddings.append(embedding)
            metadatas.append(meta)
            count += 1
        print(f"  loaded {count} records from {path}")
    return ids, docs, embeddings, metadatas
 
 
# ---------------------------------------------------------------------------
# Chroma client
# ---------------------------------------------------------------------------
def get_chroma_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
 
 
def get_or_reset_collection(client, name: str, reset: bool):
    if reset:
        try:
            client.delete_collection(name=name)
            print(f"[RESET] deleted existing collection '{name}'")
        except Exception:
            pass  # didn't exist yet — fine
    return client.get_or_create_collection(name=name)
 
 
def upsert_in_batches(collection, ids, docs, embeddings, metadatas, batch_size=BATCH_SIZE):
    total = len(ids)
    if total == 0:
        print("  nothing to upsert")
        return
    for i in range(0, total, batch_size):
        sl = slice(i, i + batch_size)
        collection.upsert(
            ids=ids[sl],
            documents=docs[sl],
            embeddings=embeddings[sl],
            metadatas=metadatas[sl],
        )
        print(f"  upserted {min(i + batch_size, total)}/{total}")
 
 
# ---------------------------------------------------------------------------
# Per-collection ingestion
# ---------------------------------------------------------------------------
def ingest_main_kb(client, reset: bool):
    print(f"\n=== Ingesting main KB ({MAIN_COLLECTION_NAME}) ===")
    source_dirs = ["1000_premiers_jours", "mpedia", "papoto", "CAF64_articles"]
    paths = [find_one(DATA_DIR / d, "*_embeddings.jsonl") for d in source_dirs]
 
    ids, docs, embeddings, metadatas = load_records(paths)
    collection = get_or_reset_collection(client, MAIN_COLLECTION_NAME, reset)
    upsert_in_batches(collection, ids, docs, embeddings, metadatas)
    print(f"'{MAIN_COLLECTION_NAME}' now has {collection.count()} documents.")
 
 
def ingest_events(client, reset: bool):
    print(f"\n=== Ingesting events ({EVENTS_COLLECTION_NAME}) ===")
    path = DATA_DIR / "CAF64_events" / "embedded_events_caf64.jsonl"
 
    ids, docs, embeddings, metadatas = load_records([path])
    collection = get_or_reset_collection(client, EVENTS_COLLECTION_NAME, reset)
    upsert_in_batches(collection, ids, docs, embeddings, metadatas)
    print(f"'{EVENTS_COLLECTION_NAME}' now has {collection.count()} documents.")
 
 
def ingest_services(client, reset: bool):
    print(f"\n=== Ingesting services ({SERVICES_COLLECTION_NAME}) ===")
    paths = [
        DATA_DIR / "CD64_services" / "sdsei" / "embedded_sdsei.jsonl",
        DATA_DIR / "CD64_services" / "vifs" / "embedded_vifs.jsonl",
    ]
 
    ids, docs, embeddings, metadatas = load_records(paths)
    collection = get_or_reset_collection(client, SERVICES_COLLECTION_NAME, reset)
    upsert_in_batches(collection, ids, docs, embeddings, metadatas)
    print(f"'{SERVICES_COLLECTION_NAME}' now has {collection.count()} documents.")
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Populate Chroma collections from pre-embedded JSONL files.")
    parser.add_argument(
        "--collection",
        choices=["main", "events", "services", "all"],
        required=True,
        help="Which collection(s) to ingest.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the target collection(s) before ingesting.",
    )
    args = parser.parse_args()
 
    print(f"Connecting to Chroma at {CHROMA_HOST}:{CHROMA_PORT}")
    print(f"Reading source data from {DATA_DIR}")
    client = get_chroma_client()
    client.heartbeat()  # fail fast with a clear error if chroma isn't reachable
 
    if args.collection in ("main", "all"):
        ingest_main_kb(client, args.reset)
    if args.collection in ("events", "all"):
        ingest_events(client, args.reset)
    if args.collection in ("services", "all"):
        ingest_services(client, args.reset)
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)