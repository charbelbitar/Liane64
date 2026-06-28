from prometheus_client import Counter

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