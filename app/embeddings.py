import requests
import os

EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://embeddings:80").rstrip("/")

def embed(text: str) -> list[float]:
    """Embed a single string via the `embeddings` (TEI / BGE-m3) container.
 
    Returns a flat list[float] — callers should use it directly (e.g. pass
    straight to chromadb or np.dot); there is no numpy array here, so do NOT
    call .tolist() on the result.
    """
    response = requests.post(
        f"{EMBEDDING_URL}/embed",
        json={"inputs": text},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    # TEI's /embed endpoint always returns one embedding per input, even for
    # a single string — unwrap to the single flat vector callers expect.
    return result[0]