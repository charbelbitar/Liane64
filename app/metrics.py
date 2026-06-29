from prometheus_client import Counter, Histogram

GROUNDING_DISCARDED = Counter(
    "rag_grounding_discarded_total",
    "Number of answers discarded by the grounding check"
)
GROUNDING_LLM_CHECK_CALLS = Counter(
    "rag_grounding_llm_check_total",
    "Number of times the borderline-overlap case escalated to an LLM grounding check"
)
CACHE_HITS = Counter("rag_cache_hits_total", "Semantic cache hits")
CACHE_MISSES = Counter("rag_cache_misses_total", "Semantic cache misses")

# ── Pipeline stage timing ──────────────────────────────────────────────
EMBED_DURATION = Histogram(
    "rag_embed_duration_seconds",
    "Time spent calling the embeddings (TEI) service per request"
)
TRANSLATE_DURATION = Histogram(
    "rag_translate_duration_seconds",
    "Time spent on LLM translation calls (urgency/retrieval translation)"
)
REWRITE_DURATION = Histogram(
    "rag_rewrite_duration_seconds",
    "Time spent on query rewrite/topic-similarity LLM calls"
)
LLM_GENERATION_DURATION = Histogram(
    "rag_llm_generation_duration_seconds",
    "Time spent on the main answer-generation LLM call"
)
GROUNDING_LLM_DURATION = Histogram(
    "rag_grounding_llm_duration_seconds",
    "Time spent on the LLM grounding-check tiebreaker call"
)
RETRIEVAL_DURATION = Histogram(
    "rag_retrieval_duration_seconds",
    "Time spent on the concurrent docs/events/services retrieval block"
)
PIPELINE_TOTAL_DURATION = Histogram(
    "rag_pipeline_total_duration_seconds",
    "End-to-end time for a full rag_pipeline() call"
)