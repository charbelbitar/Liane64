import requests
import os
from metrics import EMBED_DURATION

EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://embeddings:80").rstrip("/")

def embed(text: str) -> list[float]:
    with EMBED_DURATION.time():
        response = requests.post(
            f"{EMBEDDING_URL}/embed",
            json={"inputs": text},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
    return result[0]