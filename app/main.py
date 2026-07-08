from config import client, LLM_MODEL, collection, REFUSAL_RESPONSES, is_refusal
from retriever import retrieve_and_rerank, retrieve_events, retrieve_services, detect_location
from prompt import build_prompt
from embeddings import embed
from cache import get_cached_answer, add_to_cache, purge_invalid_entries
import numpy as np
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import pathlib, json
import re
import concurrent.futures
import time        
import threading
from langdetect import detect_langs
from prometheus_client import Counter
from metrics import (
    TRANSLATE_DURATION, REWRITE_DURATION, LLM_GENERATION_DURATION,
    GROUNDING_LLM_DURATION, RETRIEVAL_DURATION, PIPELINE_TOTAL_DURATION,
    GROUNDING_DISCARDED, GROUNDING_LLM_CHECK_CALLS,
)
from datetime import datetime, timezone


_tokenizer = MistralTokenizer.v3()
_tok = _tokenizer.instruct_tokenizer.tokenizer


def count_tokens(text: str) -> int:
    req = ChatCompletionRequest(messages=[UserMessage(content=text)])
    tokens = _tokenizer.encode_chat_completion(req).tokens
    return len(tokens)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


FOLLOWUP_THRESHOLD_HIGH = 0.75  # followup → rewrite the query directly
FOLLOWUP_THRESHOLD_LOW  = 0.40  # new topic → skip rewriting


URGENCY_KEYWORDS = [
    "maltraite", "maltraitent", "maltraitance",
    "abuse", "abusent",
    "violence", "violent", "violente",
    "enfance en danger",
    "signalement",
    "blessé", "blessée",
    "brutalise", "brutalise",
    "négligence", "négligé",
]


URGENCY_FALSE_POSITIVES = [
    "taper sur les nerfs", "taper dans l'oeil", "taper dans le mille", "frapper à la porte", "battre des records", "battre en retraite", "se battre pour", "cogne pas",
    "coups de foudre", "coups de chance", "se battre contre le temps", "se battre avec les mots", "battre son propre record", "battre la mesure", "battre le pavé", 
    "battre froid", "battre en brèche", "battre des cils", "frapper les esprits", "frapper fort", "frapper un grand coup", "frapper l'imagination", "coup de théâtre", 
    "coup de cœur", "coup de tête", "coup de balai", "coup de main", "coup de bol", "coup de vieux", "coup d'œil", "taper sur le système", "taper du pied", 
    "taper la discute", "taper dans la caisse", "taper dans l'œil du public", "taper un sprint", "frapper à toutes les portes", "frapper juste", 
    "frapper un grand coup médiatique", "frapper les consciences", "frapper l’attention", "frapper fort les esprits", "battre la chamade", "battre le rappel", 
    "battre en cadence", "battre à l’unisson", "battre le fer", "battre à plates coutures", "battre tous les records", "battre la campagne", "battre des mains", 
    "battre le fer tant qu'il est chaud", "se battre comme un lion", "se battre jusqu'au bout", "se battre contre vents et marées", "se battre pour ses idées",
    "se battre contre soi-même", "se battre pour survivre", "coup de sang", "coup de chaud", "coup de froid", "coup de vent", "coup de feu", "coup de blues",
    "coup de poker", "coup de maître", "coup de tonnerre", "coup de massue", "coup de grâce", "coup de folie", "coup de stress", "coup de génie"
]


URGENCY_ACTION_VERBS = re.compile(
    r'\b(tape|tapent|frappe|frappent|bat\b|battent|cogne|cognent|'
    r'donne\s+des\s+coups?|reçoit\s+des\s+coups?|coups?\s+sur)\b',
    re.IGNORECASE
)


VIOLENCE_TARGETS = re.compile(
    r'\b(enfant|bébé|fils|fille|gosse|gamin|enfants|bébés|lui|elle|le|la)\b',
    re.IGNORECASE
)


_SERVICE_OR_EVENT_INTENT_KEYWORDS = [
    "service", "aide", "accompagnement", "mode de garde", "garde", "crèche", "creche",
    "assistante sociale", "pmi", "sage-femme", "sage femme", "pédiatre", "pediatre",
    "atelier", "consultation", "rendez-vous", "rendez vous", "allocation", "caf",
    "association", "centre", "antenne", "permanence", "soutien", "écoute", "ecoute",
    "orientation", "structure", "événement", "evenement", "réunion", "reunion",
    "rencontre", "formation", "conférence", "conference", "séance", "seance",
    "groupe de parole", "stage", "inscription",
]


_AGE_REQUIRED_KEYWORDS = [
    "dort", "dodo", "sommeil", "mange", "alimentation", "repas", "diversification",
    "développement", "apprentissage", "marche", "parle", "langage", "vaccin",
    "taille", "poids", "croissance", "pleure", "dent", "allaitement", "biberon",
    "écran", "autonomie", "propreté", "sevrage"
]


_LOCATION_REQUIRED_KEYWORDS = [
    "atelier", "événement", "activité", "groupe", "rencontre", "sortie",
    "service", "accompagnement", "soutien", "aide", "pmr", "crèche",
    "relais", "rpe", "maison", "centre", "association"
]


_PERSONAL_INDICATORS = [
    "mon ", "ma ", "notre ", "son ", "sa ",
    "bébé", "enfant", "fille", "fils", "petit", "petite",
    "nourrisson", "ado", "adolescent",
    "il ", "elle ", "le bébé", "la bébé",
    "je suis enceinte", "ma grossesse", "ma femme", "mon mari",
]


def _extract_age_from_history(query: str, chat_history: list) -> bool:
    import re
    age_pattern = re.compile(
        r'\b(\d+)\s*(?:semaine|sem|mois|month|an(?:s)?|year)',
        re.IGNORECASE
    )
    if age_pattern.search(query): # current query 
        return True
    for msg in chat_history: # previous messages in conversation
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if age_pattern.search(content):
            return True
    return False


def _needs_age_clarification(query: str, chat_history: list) -> bool:
    q = query.lower()
    if _extract_age_from_history(query, chat_history):
        return False
    if not any(ind in q for ind in _PERSONAL_INDICATORS):
        return False
    return any(kw in q for kw in _AGE_REQUIRED_KEYWORDS)


def _needs_location_clarification(query: str, chat_history: list) -> bool:
    q = query.lower()
    full_text = query + " ".join(m.get("content","") for m in chat_history)
    from retriever import detect_location
    ville, cp = detect_location(full_text)
    if ville or cp:
        return False 
    return any(kw in q for kw in _LOCATION_REQUIRED_KEYWORDS)


def _has_service_or_event_intent(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _SERVICE_OR_EVENT_INTENT_KEYWORDS)


_VERB_CONJUGATIONS: dict[str, list[str]] = {
    "taper":   ["tape", "tapes", "tapent", "tapais", "tapait", "tapions", "tapiez",
                "tapaient", "tapant", "tapé", "tapée", "tapés", "tapées", "tapa", "tapèrent"],
    "frapper": ["frappe", "frappes", "frappent", "frappais", "frappait", "frappions",
                "frappiez", "frappaient", "frappant", "frappé", "frappée", "frappés",
                "frappées", "frappa", "frappèrent"],
    "battre":  ["bats", "bat", "battent", "battais", "battait", "battions", "battiez",
                "battaient", "battant", "battu", "battue", "battus", "battues"],
    "cogner":  ["cogne", "cognes", "cognent", "cognais", "cognait", "cognions", "cogniez",
                "cognaient", "cognant", "cogné", "cognée", "cognés", "cognées"],
}


def _expand_false_positive(phrase: str) -> list[str]:
    """Return [phrase] plus a variant for each common conjugation of its lead verb
    (handling a leading reflexive 'se ' like in 'se battre pour')."""
    reflexive = ""
    body = phrase
    if phrase.startswith("se "):
        reflexive, body = "se ", phrase[3:]

    first_word, _, remainder = body.partition(" ")
    conjugations = _VERB_CONJUGATIONS.get(first_word)
    if not conjugations:
        return [phrase]

    variants = [phrase]
    for conj in conjugations:
        variant = f"{reflexive}{conj} {remainder}".strip() if remainder else f"{reflexive}{conj}"
        variants.append(variant)
    return variants


_EXPANDED_FALSE_POSITIVES = [
    variant
    for fp in URGENCY_FALSE_POSITIVES
    for variant in _expand_false_positive(fp)
]

_FP_RE = re.compile(
    "|".join(re.escape(fp) for fp in _EXPANDED_FALSE_POSITIVES),
    re.IGNORECASE
)


def detect_urgency(query: str) -> bool:
    q = query.lower()

    if _FP_RE.search(q):
        return False

    for kw in URGENCY_KEYWORDS:
        pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        if pattern.search(q):
            return True

    verb_match = URGENCY_ACTION_VERBS.search(q)
    if verb_match:
        start = max(0, verb_match.start() - 60)
        end   = min(len(q), verb_match.end() + 60)
        window = q[start:end]
        if VIOLENCE_TARGETS.search(window):
            return True

    return False
 
URGENCY_RESPONSE = """⚠️ Ce que vous décrivez semble indiquer une situation d'enfance en danger.
 
Si vous pensez qu'un enfant est en danger immédiat, contactez sans attendre :
 
• 🚨 **119** — Numéro national de l'enfance en danger (gratuit, 24h/24)
• 🚑 **15** — SAMU (urgence médicale)
• 👮 **17** — Police (urgence immédiate)
• 🚒 **18** — Pompiers
 
Vous pouvez signaler de façon anonyme. Un signalement vaut mieux qu'une absence de signalement : les professionnels évalueront la situation."""
 
CLOSING_PHRASES = [
    "merci", "ok merci", "okay merci", "thank you", "thanks", "ok", "bye", "d'accord",
    "ok thank you", "okay thank you", "c'est bon", "c'est tout", "gracias",
    "ça marche", "parfait", "super merci", "au revoir", "bonne journée"
]


def is_closing_message(query: str):
    q = query.strip().lower()
    if q in CLOSING_PHRASES:
        return True
    if len(q.split()) <= 4 and any(phrase in q for phrase in CLOSING_PHRASES):
        return True
    return False


SHORT_QUERY_WORD_LIMIT = 3

AFFIRMATIVE_TOKENS = {"oui", "yes", "ok", "okay", "d'accord", "bien sûr",
                      "allez", "vas-y", "go", "carrément", "absolument"}

def is_short_followup(query: str) -> bool:
    words = query.strip().lower().split()
    return len(words) <= SHORT_QUERY_WORD_LIMIT


NEGATIVE_TOKENS = {
    "non", "no", "nope", "nan", "pas vraiment",
    "ça va", "c'est bon", "laisse tomber", "laisse tombez",
    "pas besoin", "non merci", "no thanks", "no merci",
}


def is_negative_followup(query: str) -> bool:
    return query.strip().lower() in NEGATIVE_TOKENS


def _llm(messages: list, temperature: float = 0, max_retries: int = 3, max_tokens: int | None = None) -> str:
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = dict(model=LLM_MODEL, messages=messages, temperature=temperature)
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except Exception as e:
            last_exc = e
            wait = 2 ** attempt  # 2s, 4s, 8s
            print(f"[LLM] Attempt {attempt}/{max_retries} failed: {e} — retrying in {wait}s")
            time.sleep(wait)

    print(f"[LLM] All {max_retries} attempts failed: {last_exc}")
    raise RuntimeError(f"LLM unavailable after {max_retries} attempts: {last_exc}") from last_exc



def _detect_query_language(text: str) -> str:
    if len(text.split()) < 3:
        return "fr"
    try:
        top = detect_langs(text)[0]
        if top.prob >= 0.70:
            return top.lang
    except Exception as e:
        print(f"[LANG] Detection failed, defaulting to fr: {e}")
    return "fr"


# Translate a non-French query to French for retrieval/caching ONLY
def translate_to_french(query: str) -> str:
    system_msg = {
        "role": "system",
        "content": (
            "Traduis le texte suivant en français, en conservant exactement le sens, "
            "les noms propres, les villes, les codes postaux et les chiffres. "
            "Réponds UNIQUEMENT avec la traduction française, sans aucun texte avant ou après."
        )
    }
    try:
        with TRANSLATE_DURATION.time():
            translated = _llm([system_msg, {"role": "user", "content": query}], temperature=0, max_tokens=200)
        return translated.strip() or query
    except RuntimeError as e:
        print(f"[TRANSLATE] Failed, falling back to original query for retrieval: {e}")
        return query


def rewrite_query(query, chat_history) -> tuple[str, list | None]:
    if len(chat_history) < 2:
        return query, None

    user_messages = [m["content"] for m in chat_history if m["role"] == "user"]
    if not user_messages:
        return query, None
    history_text = "\n".join(user_messages[-4:])

    if is_short_followup(query):
        if is_negative_followup(query):
            print(f"[NEGATIVE FOLLOWUP] '{query}' — user declined")
            return "__NEGATIVE__", None
        last_assistant = next(
            (m["content"] for m in reversed(chat_history) if m["role"] == "assistant"),
            ""
        )
        print(f"[SHORT QUERY] '{query}' — rewriting from history")
        return _do_rewrite(query, history_text, last_assistant=last_assistant), None

    query_emb   = embed(query)
    history_emb = embed(history_text)
    score = cosine_similarity(query_emb, history_emb)
    print(f"[TOPIC SIMILARITY] {score:.3f}")

    if score < FOLLOWUP_THRESHOLD_LOW:
        print("[TOPIC SHIFT] new topic detected, skipping rewrite")
        return query, query_emb 

    if score > FOLLOWUP_THRESHOLD_HIGH:
        last_assistant = next(
            (m["content"] for m in reversed(chat_history) if m["role"] == "assistant"),
            ""
        )
        return _do_rewrite(query, history_text, last_assistant=last_assistant), None

    print("[AMBIGUOUS] asking LLM to decide")
    prompt = (
        "L'historique de conversation suivant et la nouvelle question sont-ils sur le même sujet ?\n"
        "Réponds uniquement par OUI ou NON.\n\n"
        f"Historique :\n{history_text}\n\n"
        f"Nouvelle question :\n{query}\n\nRéponse :"
    )
    try:
        with REWRITE_DURATION.time():
            content = _llm([{"role": "user", "content": prompt}])
    except RuntimeError as e:
        print(f"[REWRITE] Ambiguous-topic check failed, treating as same topic: {e}")
        content = "OUI" 

    if not content.strip().upper().startswith("OUI"):
        return query, query_emb

    last_assistant = next(
        (m["content"] for m in reversed(chat_history) if m["role"] == "assistant"),
        ""
    )
    return _do_rewrite(query, history_text, last_assistant=last_assistant), None

    
def _do_rewrite(query: str, history_text: str, last_assistant: str = "") -> str:
    context = ""
    if last_assistant:
        context = f"\nDernière réponse du chatbot :\n{last_assistant}\n"

    prompt = (
        "Tu es un assistant qui reformule des questions de suivi en questions autonomes.\n\n"
        "RÈGLE CRITIQUE : La nouvelle question DOIT inclure TOUT le contexte pertinent "
        "de l'historique (âge de l'enfant, stade, sujet principal) pour qu'elle soit compréhensible sans l'historique.\n\n"
        "Exemples :\n"
        "- Historique: 'mon bébé de 6 mois dort mal' → Suivi: 'il mange bien' "
        "→ Reformulation: 'un bébé de 6 mois qui dort mal mais mange bien — "
        "y a-t-il un lien entre alimentation et sommeil à cet âge ?'\n"
        "- Historique: 'grossesse à 7 mois' → Suivi: 'et pour les douleurs de dos ?' "
        "→ Reformulation: 'quelles sont les causes et remèdes pour les douleurs de dos au 7ème mois de grossesse ?'\n\n"
        f"Historique de la conversation :\n{history_text}\n"
        f"{context}\n"
        f"Question de suivi : {query}\n\n"
        "Question reformulée (autonome, avec tout le contexte) :"
    )
    try:
        with REWRITE_DURATION.time():
            return _llm([{"role": "user", "content": prompt}])
    except RuntimeError as e:
        print(f"[REWRITE] Failed, falling back to original query: {e}")
        return query


def _extract_reponse_field(raw: str) -> str | None:
    match = re.search(r'"reponse"\s*:\s*"(.*)', raw, re.DOTALL)
    if not match:
        return None
    content = match.group(1)
    end = re.search(r'"\s*,\s*"sources"', content)
    if end:
        content = content[:end.start()]
    content = content.replace('\\n', '\n').replace('\\"', '"').replace('\\t', '\t')
    content = re.sub(r'```\s*$', '', content).strip()
    return content or None


def parse_llm_response(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
 
    try:
        parsed = json.loads(cleaned, strict=False)
        if is_refusal(parsed.get("reponse", "")):
            parsed.update({
                "language": "ambigu", 
                "niveau_langue": "ambigu",
                "role_detecte": "ambigu", 
                "phase": "ambigu", 
                "urgence": "non",
                "sources": [],
            })
        return parsed
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return parsed
            except json.JSONDecodeError:
                pass

        extracted = _extract_reponse_field(cleaned)
        if extracted:
            return {
                "language": "ambigu", "niveau_langue": "ambigu",
                "role_detecte": "ambigu", "phase": "ambigu",
                "urgence": "non", "reponse": extracted, "sources": [],
            }
    
        return {
            "language": "ambigu", 
            "niveau_langue": "ambigu",
            "role_detecte": "ambigu", 
            "phase": "ambigu",
            "urgence": "non", 
            "reponse": raw,
        }


def is_valid_answer(answer: str) -> bool:
    if not answer or len(answer.strip()) < 10:
        return False
    if answer.strip().startswith("<html") or "504" in answer or "Gateway Time-out" in answer:
        return False
    return True

  
ROLE_LABELS = {
    "parent":        "👨‍👩‍👧 Parent",
    "professionnel": "🏥 Professionnel de santé/social",
    "ambigu":        "❓ Rôle indéterminé",
}

PHASE_LABELS = {
    "grossesse":     "🤰 Grossesse",
    "post-natalite": "🍼 Post-natalité",
    "bebe":          "👶 Bébé",
    "enfance":       "🧒 Enfance",
    "adolescence":   "🧑 Adolescence",
    "ambigu":        "❓ Phase indéterminée",
}

URGENCE_LABELS = {
    "oui": "🚨 OUI", 
    "non": "✅ Non",
}


def print_response(parsed: dict):
    reponse = parsed.get("reponse", "").strip()
 
    if is_refusal(reponse):
        print("\nRéponse :\n")
        print(reponse)
        print()
        return
 
    role    = ROLE_LABELS.get(parsed.get("role_detecte", "ambigu"), "❓")
    phase   = PHASE_LABELS.get(parsed.get("phase", "ambigu"), "❓")
    urgence = URGENCE_LABELS.get(parsed.get("urgence", "non"), "✅ Non")
    langue  = parsed.get("language", "ambigu").capitalize()
    niveau  = parsed.get("niveau_langue", "ambigu").capitalize()
    sources = parsed.get("sources", [])
 
    print("\n" + "─" * 60)
    print(f"  Rôle détecté   : {role}")
    print(f"  Phase          : {phase}")
    print(f"  Urgence        : {urgence}")
    print(f"  Langue         : {langue}  |  Niveau : {niveau}")

    chunk_meta = parsed.get("chunk_meta", [])
    if chunk_meta:
        print("─" * 60)
        print("  Chunks utilisés :")
        for cm in chunk_meta:
            stade   = cm.get("stade", "?")
            risque  = cm.get("risque", "?")
            kws     = ", ".join(cm.get("mots_clés", [])) or "—"
            risque_icon = {"élevé": "🔴", "eleve": "🔴", "moyen": "🟡", "faible": "🟢"}.get(risque.lower(), "⚪")
            print(f"    • stade={stade}  {risque_icon} risque={risque}  🔑 {kws}")
            
    print("─" * 60)
    print("\nRéponse :\n")
    print(reponse)
    if sources:
        print("\n📚 Sources :")
        for url in sources:
            print(f"   • {url}")
    print()


_BOILERPLATE_PATTERNS = [
    r"🚨.*?119.*?danger",
    r"Le Fil des parents.*?Tipi",
    r"monenfant\.fr",
    r"SAMU.*?15.*?Police.*?17.*?[Pp]ompiers.*?18",
]

def _strip_boilerplate(text: str) -> str:
    for pat in _BOILERPLATE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def _compute_overlap(answer: str, context_docs: list[str]) -> float:
    context_text = " ".join(context_docs).lower()
    context_words = set(re.findall(r"\w{4,}", context_text))
    answer_words_list = re.findall(r"\w{4,}", _strip_boilerplate(answer).lower())
    if not answer_words_list:
        return 1.0
    matched = sum(1 for w in answer_words_list if w in context_words)
    return matched / len(answer_words_list)


def llm_grounding_check(answer: str, context_docs: list[str]) -> bool:
    context_text = "\n\n".join(context_docs)[:6000]
    prompt = (
        "Voici un CONTEXTE et une RÉPONSE générée à partir de ce contexte.\n"
        "Réponds UNIQUEMENT par OUI si chaque affirmation factuelle de la réponse "
        "est présente dans le contexte ou en découle directement. "
        "Réponds NON si la réponse contient une information absente du contexte.\n\n"
        f"CONTEXTE:\n{context_text}\n\nRÉPONSE:\n{answer}\n\nVerdict (OUI/NON):"
    )
    try:
        with GROUNDING_LLM_DURATION.time():
            verdict = _llm([{"role": "user", "content": prompt}], temperature=0, max_tokens=5)
        return verdict.strip().upper().startswith("OUI")
    except RuntimeError as e:
        print(f"[GROUNDING-LLM] Check failed, defaulting to trust heuristic: {e}")
        return True


def _is_followup(chat_history: list) -> bool:
    return len([m for m in chat_history if m.get("role") == "assistant"]) > 0


# main logic 
def _append_turn(chat_history: list, query: str, answer: str) -> None:
    chat_history.append({"role": "user",      "content": query})
    chat_history.append({"role": "assistant", "content": answer})

def rag_pipeline(query: str, chat_history):
    with PIPELINE_TOTAL_DURATION.time():
        result = _rag_pipeline_impl(query, chat_history)
        return result 
    
def _rag_pipeline_impl(query: str, chat_history):

    # Urgency detection must work regardless of input language
    urgency_text = query
    query_lang_for_urgency = _detect_query_language(query)
    if query_lang_for_urgency != "fr":
        urgency_text = translate_to_french(query)

    if detect_urgency(query) or detect_urgency(urgency_text):
        print("[URGENCY] Emergency keywords detected — bypassing pipeline")
        _append_turn(chat_history, query, URGENCY_RESPONSE)
        return URGENCY_RESPONSE, {
            "language": "francais", "niveau_langue": "ambigu",
            "role_detecte": "ambigu", "phase": "enfance",
            "urgence": "oui", "reponse": URGENCY_RESPONSE,
        }, [], []

    if is_closing_message(query):
        farewell = "De rien, n'hésitez pas à revenir si vous avez d'autres questions. Bonne continuation ! 😊"
        _append_turn(chat_history, query, farewell)
        return farewell, {}, [], []


    # Clarification gates
    if not _is_followup(chat_history):
    # Only ask clarification on FIRST message, never on follow-ups
        needs_age = _needs_age_clarification(query, chat_history)
        needs_loc = _needs_location_clarification(query, chat_history)

        if needs_age:
            clarification = (
                "Pour vous donner une réponse précise et adaptée, "
                "pourriez-vous me préciser l'âge de votre enfant "
                "(en semaines, mois ou années) ?"
            )
            print(f"[CLARIFICATION] Age required for query: {query[:60]}")
            _append_turn(chat_history, query, clarification)
            return clarification, {
                "language": "francais", "niveau_langue": "ambigu",
                "role_detecte": "parent", "phase": "ambigu",
                "urgence": "non", "reponse": clarification, "sources": []
            }, [], []

        if needs_loc:
            clarification = (
                "Pour vous proposer des événements et services près de chez vous, "
                "pourriez-vous m'indiquer votre ville ou code postal ?"
            )
            print(f"[CLARIFICATION] Location required for query: {query[:60]}")
            _append_turn(chat_history, query, clarification)
            return clarification, {
                "language": "francais", "niveau_langue": "ambigu",
                "role_detecte": "parent", "phase": "ambigu",
                "urgence": "non", "reponse": clarification, "sources": []
            }, [], []

    # Full pipeline

    rewritten_query, query_emb = rewrite_query(query, chat_history)

    if rewritten_query == "__NEGATIVE__":
        reply = "Pas de problème ! N'hésitez pas si vous avez d'autres questions. 😊"
        _append_turn(chat_history, query, reply)
        return reply, {}, [], []


    query_lang = _detect_query_language(rewritten_query)
    if query_lang != "fr":
        if rewritten_query == query and urgency_text != query:
            retrieval_query = urgency_text
        else:
            print(f"[LANG] Non-French query detected (lang={query_lang}) — translating for retrieval/caching")
            retrieval_query = translate_to_french(rewritten_query)
        query_emb = None 
    else:
        retrieval_query = rewritten_query
    detected_ville, detected_cp = detect_location(retrieval_query)

    cached_answer = get_cached_answer(retrieval_query)
    if cached_answer:
        try:
            parsed = json.loads(cached_answer)
            if not isinstance(parsed, dict) or "reponse" not in parsed:
                raise ValueError("Cached entry missing expected structure")
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"[CACHE] Invalid or legacy entry, discarding: {e}")
            parsed = None
        if parsed:
            answer = parsed.get("reponse", "")
            if answer and not is_refusal(answer):
                _append_turn(chat_history, query, answer)
                return answer, parsed, [], []
            print("[CACHE] Cached answer was a refusal or empty — retrying")


    # Main document retrieval 
    print("[PIPELINE] Retrieving documents, events & services concurrently...")
    if query_emb is None:
        retrieval_query_emb = embed(retrieval_query)
    else:
        retrieval_query_emb = query_emb if isinstance(query_emb, list) else query_emb.tolist()

    with RETRIEVAL_DURATION.time():
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_docs     = executor.submit(retrieve_and_rerank, retrieval_query, 15, 50, None, retrieval_query_emb)
            future_events   = executor.submit(retrieve_events,   retrieval_query, 3, 0.50, retrieval_query_emb)
            future_services = executor.submit(retrieve_services, retrieval_query, 3, 0.45, retrieval_query_emb)

        ranked_docs        = future_docs.result()
        relevant_events    = future_events.result()
        relevant_services  = future_services.result()

    MIN_RERANK_SCORE = 0.45
    print(f"[RERANK] Scores before filtering: {[round(d.get('score', 0), 3) for d in ranked_docs]}")
    ranked_docs = [d for d in ranked_docs if d.get("score", 0) >= MIN_RERANK_SCORE]

    # Only refuse outright if NOTHING relevant was found anywhere 
    if not ranked_docs and not relevant_events and not relevant_services:
        if not (detected_ville or detected_cp) and _has_service_or_event_intent(retrieval_query):
            LOOSE_FLOOR = 0.35
            loose_events   = retrieve_events(retrieval_query, n_results=1, score_threshold=LOOSE_FLOOR, query_emb=retrieval_query_emb)
            loose_services = retrieve_services(retrieval_query, n_results=1, score_threshold=LOOSE_FLOOR, query_emb=retrieval_query_emb)
            if loose_events or loose_services:
                print("[RAG] No location given, but a plausible event/service exists below threshold — asking for location")
                out = ("Pour vous orienter vers les services ou événements adaptés, "
                       "pourriez-vous préciser votre ville ou votre code postal ?")
                _append_turn(chat_history, query, out)
                return out, {
                    "language": "francais", "niveau_langue": "ambigu",
                    "role_detecte": "ambigu", "phase": "ambigu",
                    "urgence": "non", "reponse": out,
                    "sources": []
                }, [], []
 
        print("[RAG] No sufficiently relevant docs/events/services — returning refusal without LLM call")
        out = "Pas de ressources disponibles."
        _append_turn(chat_history, query, out)
        return out, {
            "language": "ambigu", "niveau_langue": "ambigu",
            "role_detecte": "ambigu", "phase": "ambigu",
            "urgence": "non", "reponse": out,
            "sources": []
        }, [], []

    # DEBUG: print retrieved chunks
    print(f"\n{'='*60}")
    print(f"[CHUNKS] {len(ranked_docs)} chunks passed score filter")
    print(f"{'='*60}")
    for i, doc in enumerate(ranked_docs):
        print(f"\n--- Chunk #{i+1} | score={doc['score']:.3f} | source={doc['metadata'].get('source', '?')} ---")
        print(f"URL   : {doc['metadata'].get('url', 'N/A')}")
        print(f"Title : {doc['metadata'].get('title', 'N/A')}")
        print(f"Content:\n{doc['content'][:500]}{'...' if len(doc['content']) > 500 else ''}")
    print(f"\n{'='*60}\n")


    MAX_CONTEXT_TOKENS = 10000
    context_parts: list[str] = []
    context_tokens = 0
 
    for doc in ranked_docs:
        content   = doc["content"]
        url       = doc["metadata"].get("url", "")
        title     = doc["metadata"].get("title", "")
        source    = doc["metadata"].get("source", "")

        # feeds the metadata directly into the context window so the LLM sees it alongside the chunk text
        mots_clés = doc["metadata"].get("mots_clés", doc["metadata"].get("mots_cles", ""))
        risque    = doc["metadata"].get("risque", "")
        stade     = doc["metadata"].get("stade", "")

        meta_line = " | ".join(filter(None, [
            f"stade={stade}"       if stade    else "",
            f"risque={risque}"     if risque   else "",
            f"mots-clés={mots_clés}" if mots_clés else "",
        ]))
        annotated = f"[{source} — {title}]"
        if meta_line:
            annotated += f"\n[{meta_line}]"
        annotated += f"\n{content}\n[URL: {url}]"  

        chunk_tokens = count_tokens(annotated)
        if context_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
            remaining = MAX_CONTEXT_TOKENS - context_tokens
            if remaining > 100:
                ids       = _tok.encode(annotated, bos=False, eos=False)
                truncated = _tok.decode(ids[:remaining])

                # Cut at the last complete sentence to avoid mid-word/mid-token garbling
                last_period = max(
                    truncated.rfind("."),
                    truncated.rfind("!"),
                    truncated.rfind("?"),
                )
                if last_period != -1 and last_period > len(truncated) * 0.5:
                    truncated = truncated[: last_period + 1]

                context_parts.append(truncated + " […]")
            break
        context_parts.append(annotated)
        context_tokens += chunk_tokens

    real_sources = []
    for doc in ranked_docs:
        url = doc["metadata"].get("url", "")
        if url and url not in real_sources:
            real_sources.append(url)


    # Collect unique chunk metadata for display
    chunk_meta_summary = []
    seen_stades = set()
    for doc in ranked_docs:
        meta = doc["metadata"]
        stade    = meta.get("stade", "")
        risque   = meta.get("risque", "")
        raw_kw   = meta.get("mots_clés", meta.get("mots_cles", "[]"))
        try:
            import json as _json
            keywords = _json.loads(raw_kw) if isinstance(raw_kw, str) else raw_kw
        except Exception:
            keywords = []
        key = stade or "?"
        if key not in seen_stades:
            seen_stades.add(key)
            chunk_meta_summary.append({
                "stade":   stade,
                "risque":  risque,
                "mots_clés": keywords,
            })
    
    prompt = build_prompt(
        rewritten_query,
        context_parts,
        events=relevant_events if relevant_events else None,
        services=relevant_services if relevant_services else None,
        location_known=bool(detected_ville or detected_cp),
        chat_history=chat_history
    )

    system_msg = {
        "role": "system",
        "content": (
            "Tu es un assistant expert en périnatalité. "
            "RÈGLE ABSOLUE : Tu ne génères AUCUNE information qui ne figure pas mot pour mot dans le contexte fourni. "
            "Si le contexte est insuffisant, tu réponds UNIQUEMENT : 'Pas de ressources disponibles.' "
            "Tu ne te bases JAMAIS sur tes connaissances générales. Jamais. "
            "Tu réponds UNIQUEMENT en JSON valide, sans texte avant ou après."
        )
    }
    recent_turns = chat_history[-6:]
    messages = [system_msg] + recent_turns + [{"role": "user", "content": prompt}]


    try:
        with LLM_GENERATION_DURATION.time():
            raw = _llm(messages, temperature=0.2, max_tokens=1800)
    except RuntimeError as e:
        print(f"[PIPELINE] LLM call failed: {e}")
        out = "Le service est temporairement indisponible. Veuillez réessayer dans quelques instants."
        _append_turn(chat_history, query, out)
        return out, {}, [], []

    parsed = parse_llm_response(raw)
    answer = parsed.get("reponse", raw)

    NO_RESOURCE_SENTINELS = {"Pas de ressources disponibles.", "Cette question est hors périmètre."}
    if answer.strip() in NO_RESOURCE_SENTINELS or is_refusal(answer):
        parsed["sources"] = []
        parsed["chunk_meta"] = []
    else:
        parsed["sources"] = real_sources
        parsed["chunk_meta"] = chunk_meta_summary
 
    # Grounding check 
    extra_words = sum(len(e.get("text", "").split()) for e in (relevant_events or []))
    extra_words += sum(len(s.get("text", "").split()) for s in (relevant_services or []))
 
    if context_parts:
        overlap = _compute_overlap(answer, context_parts)
        print(f"[GROUNDING] Lexical overlap score: {overlap:.2f}")
        if overlap < 0.15:
            grounded = False
        elif overlap > 0.40:
            grounded = True
        else:
            print("[GROUNDING] Borderline — escalating to LLM check")
            GROUNDING_LLM_CHECK_CALLS.inc()
            grounded = llm_grounding_check(answer, context_parts)

        if not grounded:
            GROUNDING_DISCARDED.inc()
            print("[GROUNDING] Answer may contain hallucinated content — discarding")
            out = "Pas de ressources disponibles."
            _append_turn(chat_history, query, out)
            return out, {
                "language": parsed.get("language", "ambigu"),
                "niveau_langue": parsed.get("niveau_langue", "ambigu"),
                "role_detecte": parsed.get("role_detecte", "ambigu"),
                "phase": parsed.get("phase", "ambigu"),
                "urgence": parsed.get("urgence", "non"),
                "reponse": out,
                "sources": []
            }, [], []
    else:
        print("[GROUNDING] No document context — skipping overlap check (events/services only answer)")

    
    if is_valid_answer(answer) and not is_refusal(answer) and not relevant_events and not relevant_services:
        threading.Thread(
            target=add_to_cache,
            args=(retrieval_query, json.dumps(parsed, ensure_ascii=False)),
            daemon=True,
        ).start()
 
    _append_turn(chat_history, query, answer)
    return answer, parsed, relevant_events, relevant_services



if __name__ == "__main__":
    chat_history: list = []
    purge_invalid_entries()

    while True:
        query = input("\nPosez votre question (ou bien 'sortir'): ")
        if not query:
            continue
        if query.lower() == "sortir":
            break

        try:
            answer, parsed, _, _  = rag_pipeline(query, chat_history)
            if parsed:
                print_response(parsed)
            else:
                print("\nRéponse :\n", answer)
        except Exception as e:
            print(f"\n[ERREUR] Une erreur inattendue s'est produite : {e}")
