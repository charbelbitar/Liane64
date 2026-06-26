import os
from pathlib import Path
from openai import OpenAI
import chromadb


########### Paths
# BASE_DIR = Path(__file__).resolve().parent.parent
# load_dotenv(dotenv_path=BASE_DIR / ".env")
ILaaS_API_KEY = os.getenv("ILaaS_API_KEY")
ILaaS_BASE_URL = os.getenv("ILaaS_BASE_URL", "https://llm.ilaas.fr/v1")


########### ILaaS LLM
LLM_MODEL = os.getenv("LLM_MODEL")


########### Local BGE Embedding
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "C:/Users/al-bitar/.cache/huggingface/hub/models--BAAI--bge-m3")
# embedding_model = SentenceTransformer(EMBEDDING_MODEL)


########### ChromaDB
# CHROMA_PATH = BASE_DIR / "data" / "chroma_db_BGE"
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "MPJ_MPEDIA_PAPOTO_CAF_BGE")

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

collection          = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)
events_collection   = chroma_client.get_or_create_collection(name="events_caf64")
services_collection = chroma_client.get_or_create_collection(name="servicess")


chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


########### Client
client = OpenAI(
    api_key=ILaaS_API_KEY,
    base_url=ILaaS_BASE_URL,
    timeout=300.0
)


# Refusal helpers (shared across modules)
REFUSAL_RESPONSES = {
    "Cette question est hors périmètre.",
    "Pas de ressources disponibles.",
}

def is_refusal(text: str) -> bool:
    return text.strip() in REFUSAL_RESPONSES