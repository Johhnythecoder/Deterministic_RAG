from __future__ import annotations

import json
import math
import os
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader

from core_pipeline import get_llm_call_count, load_pdf, reset_llm_call_count, run_query

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
SAMPLE_DIR = BASE_DIR / "sample_docs"
UPLOAD_DIR = BASE_DIR / "uploads"
PREPARED_INDEX_PATH = BASE_DIR / "prepared_docs.json"
TOPOLOGY_TELEMETRY_PATH = BASE_DIR / "topology_telemetry.jsonl"

SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class PipelineCache:
    def __init__(self, max_items: int = 3) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self._max_items = max(1, int(max_items))

    def _prune_locked(self) -> None:
        if len(self._cache) <= self._max_items:
            return
        # Evict least-recently-used entries to bound memory + open resources.
        victims = sorted(self._cache.items(), key=lambda kv: float(kv[1].get("last_used", 0.0)))
        for key, _ in victims[: max(0, len(self._cache) - self._max_items)]:
            self._cache.pop(key, None)

    def get(self, pdf_path: Path):
        key = str(pdf_path.resolve())
        mtime = pdf_path.stat().st_mtime
        with self._lock:
            item = self._cache.get(key)
            if item and item["mtime"] == mtime:
                item["last_used"] = time.time()
                return item["pipeline"]

        pipeline = load_pdf(pdf_path)

        with self._lock:
            self._cache[key] = {"mtime": mtime, "pipeline": pipeline, "last_used": time.time()}
            self._prune_locked()
        return pipeline

    def has(self, pdf_path: Path) -> bool:
        key = str(pdf_path.resolve())
        mtime = pdf_path.stat().st_mtime
        with self._lock:
            item = self._cache.get(key)
            return bool(item and item["mtime"] == mtime)

    def set(self, pdf_path: Path, pipeline) -> None:
        key = str(pdf_path.resolve())
        mtime = pdf_path.stat().st_mtime
        with self._lock:
            self._cache[key] = {"mtime": mtime, "pipeline": pipeline, "last_used": time.time()}
            self._prune_locked()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "items": len(self._cache),
                "max_items": self._max_items,
            }


cache = PipelineCache()
app = FastAPI(title="Deterministic RAG Demo", version="1.0.0")

app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

_prep_jobs: dict[str, dict[str, Any]] = {}
_job_lock = Lock()
_prep_runtime_lock = Lock()
_prep_runtime_model: dict[str, Any] = {
    "sec_per_page_ema": 2.6,
    "samples": 0,
}
_prep_phase_model: dict[str, Any] = {
    "setup_sec_ema": 10.0,
    "sec_per_chunk_ema": 2.2,
    "post_base_sec_ema": 6.0,
    "post_sec_per_1k_nodes_ema": 10.0,
    "post_sec_per_llm_call_ema": 1.2,
    "samples": 0,
}
_doc_tokens: dict[str, Path] = {}
_prepared_index: list[dict[str, Any]] = []
_telemetry_lock = Lock()
_access_code = (os.getenv("APP_ACCESS_CODE") or "").strip()
_rate_window_sec = max(10, int(os.getenv("API_RATE_WINDOW_SEC") or 60))
_rate_limit_general = max(10, int(os.getenv("API_RATE_LIMIT_REQS") or 120))
_rate_limit_ask = max(5, int(os.getenv("API_RATE_LIMIT_ASK_REQS") or 60))
_rate_limit_prepare = max(1, int(os.getenv("API_RATE_LIMIT_PREPARE_REQS") or 8))
_rate_events: dict[str, deque[float]] = defaultdict(deque)
_rate_lock = Lock()
_server_started_at = time.time()
_ask_debug_lock = Lock()
_ask_debug: dict[str, Any] = {
    "active": 0,
    "total_started": 0,
    "total_finished": 0,
    "last_started_at": None,
    "last_finished_at": None,
    "last_latency_ms": None,
    "last_error": None,
}


def _client_ip(request: Request) -> str:
    xff = (request.headers.get("x-forwarded-for") or "").strip()
    if xff:
        return xff.split(",")[0].strip()
    return (request.client.host if request.client else "unknown") or "unknown"


def _enforce_access(request: Request, access_code: str | None = None) -> None:
    """Optional server-side access control when APP_ACCESS_CODE is set."""
    if not _access_code:
        return
    provided = (
        (access_code or "").strip()
        or (request.headers.get("x-access-code") or "").strip()
        or (request.query_params.get("access_code") or "").strip()
    )
    if provided != _access_code:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid access code.")


def _enforce_rate_limit(request: Request, bucket: str, limit: int, window_sec: int | None = None) -> None:
    """Simple in-memory per-IP rate limit."""
    window = window_sec or _rate_window_sec
    now = time.time()
    key = f"{bucket}:{_client_ip(request)}"
    with _rate_lock:
        dq = _rate_events[key]
        while dq and now - dq[0] > window:
            dq.popleft()
        if len(dq) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please retry shortly.")
        dq.append(now)


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", name).strip("._")
    return cleaned or "upload.pdf"


def _collect_demo_docs() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for p in sorted(SAMPLE_DIR.glob("*.pdf")):
        docs.append({"name": p.name, "path": str(p)})
    return docs


def _has_prebuilt_artifacts(pdf_path: Path) -> bool:
    nodes_path = pdf_path.parent / f"{pdf_path.stem}_typed_nodes.json"
    graph_path = pdf_path.parent / f"{pdf_path.stem}_graph.json"
    return nodes_path.exists() and graph_path.exists()


def _ensure_prepared_token_for_pdf(pdf_path: Path) -> str:
    resolved = str(pdf_path.resolve())
    for item in _prepared_index:
        if item.get("path") == resolved and item.get("token"):
            token = str(item.get("token") or "")
            if token:
                _doc_tokens[token] = pdf_path
                return token
    token = uuid4().hex
    _register_prepared_doc(token, pdf_path)
    return token


def _bootstrap_prepared_index_from_artifacts() -> None:
    # Ensure sample docs with baked artifacts are instantly selectable after restart/deploy.
    for p in sorted(SAMPLE_DIR.glob("*.pdf")):
        if _has_prebuilt_artifacts(p):
            _ensure_prepared_token_for_pdf(p)


def _load_prepared_index() -> None:
    global _prepared_index
    if not PREPARED_INDEX_PATH.exists():
        _prepared_index = []
    else:
        try:
            raw = json.loads(PREPARED_INDEX_PATH.read_text())
            if isinstance(raw, list):
                _prepared_index = raw
            else:
                _prepared_index = []
        except Exception:
            _prepared_index = []

    for item in _prepared_index:
        token = str(item.get("token") or "")
        p = Path(str(item.get("path") or ""))
        if token and p.exists():
            _doc_tokens[token] = p
    _bootstrap_prepared_index_from_artifacts()


def _save_prepared_index() -> None:
    PREPARED_INDEX_PATH.write_text(json.dumps(_prepared_index, indent=2))


def _register_prepared_doc(token: str, pdf_path: Path) -> None:
    resolved = str(pdf_path.resolve())
    now = int(time.time())
    existing = None
    for item in _prepared_index:
        if item.get("path") == resolved:
            existing = item
            break
    if existing:
        existing["token"] = token
        existing["name"] = pdf_path.name
        existing["last_prepared"] = now
    else:
        _prepared_index.append(
            {
                "token": token,
                "name": pdf_path.name,
                "path": resolved,
                "last_prepared": now,
            }
        )
    _doc_tokens[token] = pdf_path
    _save_prepared_index()


def _list_saved_docs() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    keep: list[dict[str, Any]] = []
    for item in _prepared_index:
        token = str(item.get("token") or "")
        p = Path(str(item.get("path") or ""))
        if not token or not p.exists():
            continue
        keep.append(item)
        docs.append(
            {
                "token": token,
                "name": str(item.get("name") or p.name),
                "path": str(p),
                "last_prepared": int(item.get("last_prepared") or 0),
            }
        )
    docs.sort(key=lambda x: x["last_prepared"], reverse=True)
    if len(keep) != len(_prepared_index):
        _prepared_index.clear()
        _prepared_index.extend(keep)
        _save_prepared_index()
    return docs


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        return max(1, len(PdfReader(str(pdf_path)).pages))
    except Exception:
        return 12


def _estimate_prep_seconds(pdf_path: Path) -> int:
    """ETA model using page count + adaptive per-page runtime from recent jobs."""
    if cache.has(pdf_path):
        return 5
    n_pages = _pdf_page_count(pdf_path)
    with _prep_runtime_lock:
        sec_per_page = float(_prep_runtime_model.get("sec_per_page_ema") or 2.6)
        samples = int(_prep_runtime_model.get("samples") or 0)
    baseline = 12 if samples < 3 else 10
    # Slightly increase for very large files where graph build and retries dominate.
    large_doc_penalty = 0.0 if n_pages < 80 else 0.15 * (n_pages - 79)
    est = int(baseline + sec_per_page * n_pages + large_doc_penalty)
    # Permit larger ETAs for long contracts while avoiding runaway values.
    return max(12, min(1200, est))


def _record_prep_runtime(n_pages: int, elapsed_sec: float) -> None:
    """Update in-memory ETA model from observed completed jobs."""
    if n_pages <= 0 or elapsed_sec <= 0:
        return
    observed = float(elapsed_sec) / float(max(1, n_pages))
    observed = max(0.8, min(20.0, observed))
    with _prep_runtime_lock:
        prev = float(_prep_runtime_model.get("sec_per_page_ema") or 2.6)
        samples = int(_prep_runtime_model.get("samples") or 0)
        alpha = 0.25 if samples < 8 else 0.12
        _prep_runtime_model["sec_per_page_ema"] = prev * (1.0 - alpha) + observed * alpha
        _prep_runtime_model["samples"] = samples + 1


def _record_phase_runtime(
    total_chunks: int,
    started_at: float,
    first_extract_at: float | None,
    done_extract_at: float | None,
    nodes_created: int,
    llm_calls_actual: int,
    finished_at: float,
) -> None:
    """Learn phase timing model: setup + extraction (sec/chunk) + post tail."""
    if started_at <= 0 or finished_at <= started_at:
        return
    elapsed = max(1.0, finished_at - started_at)
    tc = max(1, int(total_chunks or 0))

    if first_extract_at and first_extract_at > started_at:
        setup_sec = max(0.5, min(120.0, first_extract_at - started_at))
    else:
        setup_sec = max(0.5, min(120.0, elapsed * 0.18))

    if done_extract_at and first_extract_at and done_extract_at > first_extract_at:
        extract_sec = max(1.0, min(elapsed, done_extract_at - first_extract_at))
    else:
        extract_sec = max(1.0, min(elapsed, elapsed * 0.66))

    post_sec = max(0.5, min(elapsed, elapsed - setup_sec - extract_sec))
    sec_per_chunk = max(0.3, min(30.0, extract_sec / tc))

    with _prep_runtime_lock:
        smp = int(_prep_phase_model.get("samples") or 0)
        alpha = 0.28 if smp < 8 else 0.12
        _prep_phase_model["setup_sec_ema"] = float(_prep_phase_model.get("setup_sec_ema") or 10.0) * (1.0 - alpha) + setup_sec * alpha
        _prep_phase_model["sec_per_chunk_ema"] = float(_prep_phase_model.get("sec_per_chunk_ema") or 2.2) * (1.0 - alpha) + sec_per_chunk * alpha
        _prep_phase_model["post_base_sec_ema"] = float(_prep_phase_model.get("post_base_sec_ema") or 6.0) * (1.0 - alpha) + (post_sec * 0.35) * alpha
        if nodes_created > 0:
            obs_node_coef = max(0.2, min(120.0, (post_sec * 1000.0) / float(nodes_created)))
            _prep_phase_model["post_sec_per_1k_nodes_ema"] = float(_prep_phase_model.get("post_sec_per_1k_nodes_ema") or 10.0) * (1.0 - alpha) + obs_node_coef * alpha
        if llm_calls_actual > 0:
            obs_llm_coef = max(0.2, min(40.0, post_sec / float(llm_calls_actual)))
            _prep_phase_model["post_sec_per_llm_call_ema"] = float(_prep_phase_model.get("post_sec_per_llm_call_ema") or 1.2) * (1.0 - alpha) + obs_llm_coef * alpha
        _prep_phase_model["samples"] = smp + 1


def _llm_preflight_error() -> str | None:
    """Return error string when no viable LLM backend is configured."""
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    if has_openai:
        return None

    # No API key configured; verify Ollama fallback is reachable.
    raw_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    tags_url = raw_url.rsplit("/", 1)[0] + "/tags" if "/" in raw_url else "http://localhost:11434/api/tags"
    try:
        r = requests.get(tags_url, timeout=1.5)
        if r.status_code < 500:
            return None
    except Exception:
        pass
    return (
        "No OpenAI key found and Ollama is unreachable. "
        "Set OPENAI_API_KEY in .env, or run Ollama locally."
    )


def _mark_ask_start() -> None:
    with _ask_debug_lock:
        _ask_debug["active"] = int(_ask_debug.get("active") or 0) + 1
        _ask_debug["total_started"] = int(_ask_debug.get("total_started") or 0) + 1
        _ask_debug["last_started_at"] = time.time()


def _mark_ask_end(*, latency_ms: int, error: str | None = None) -> None:
    with _ask_debug_lock:
        _ask_debug["active"] = max(0, int(_ask_debug.get("active") or 0) - 1)
        _ask_debug["total_finished"] = int(_ask_debug.get("total_finished") or 0) + 1
        _ask_debug["last_finished_at"] = time.time()
        _ask_debug["last_latency_ms"] = int(latency_ms)
        _ask_debug["last_error"] = (str(error).strip() if error else None)


_DISPLAY_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "under", "over",
    "about", "there", "their", "they", "them", "what", "when", "where", "which",
    "who", "whom", "whose", "have", "has", "had", "does", "did", "are", "was",
    "were", "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "your", "ours", "its", "not", "only", "also", "than", "then", "such",
    "all", "any", "each", "either", "both", "between", "within", "across", "into",
    "onto", "upon", "before", "after", "because", "while", "during", "through",
    "agreement", "document", "contract",
}
_DEF_CLAUSE_RE = re.compile(
    r"\b("
    r"means|shall\s+mean|is\s+defined\s+as|are\s+defined\s+as|"
    r"referred\s+to\s+as|collectively\s+referred\s+to|"
    r"each\s+referred\s+to\s+as|is\s+a\b|are\s+the\b"
    r")\b",
    re.IGNORECASE,
)
_ENTITY_REL_CLAUSE_RE = re.compile(
    r"\b("
    r"referred\s+to\s+as|collectively\s+as|collectively\s+referred\s+to\s+as|"
    r"between|means|defined\s+as|each\s+referred\s+to\s+as"
    r")\b",
    re.IGNORECASE,
)
_INDIRECT_ENTITY_RE = re.compile(
    r"\b("
    r"payment|fee|invoice|liability|liable|obligation|obligations|"
    r"responsible|indemnif|breach|terminate|termination|damages|"
    r"warranty|confidential|security|tax|costs?|charges?"
    r")\b",
    re.IGNORECASE,
)


def _display_terms(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9'_-]*", (text or "").lower())
    return {t for t in tokens if len(t) >= 3 and t not in _DISPLAY_STOPWORDS}


def _is_definition_entity_question(result: dict) -> bool:
    q_node = result.get("q_node") if isinstance(result.get("q_node"), dict) else {}
    node_type = str(q_node.get("node_type") or "").upper().strip()
    if node_type == "DEFINITION":
        return True

    slot_slots = result.get("slot_slots") if isinstance(result.get("slot_slots"), list) else []
    slot_names = {
        str(s.get("slot") or "").upper().strip()
        for s in slot_slots
        if isinstance(s, dict)
    }
    if slot_names & {"MEANING", "ACTOR"}:
        return True

    topo = result.get("topo_pred") if isinstance(result.get("topo_pred"), dict) else {}
    meta = topo.get("meta") if isinstance(topo.get("meta"), dict) else {}
    q_intent = meta.get("question_intent") if isinstance(meta.get("question_intent"), dict) else {}
    primary_slot = str(q_intent.get("primary_slot") or "").upper().strip()
    proof_type = str(q_intent.get("proof_type") or "").lower().strip()
    if primary_slot in {"MEANING", "ACTOR"}:
        return True
    if proof_type in {"definition", "entity", "identity"}:
        return True
    return False


def _is_full_sentence_text(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 40:
        return False
    if t.startswith("...") or t.endswith("..."):
        return False
    if not re.search(r"[.!?;:]$", t):
        return False
    if re.search(r"^[a-z]", t):
        return False
    return True


def _extract_answer_entities(answer_text: str) -> list[str]:
    raw = (answer_text or "").strip()
    if not raw:
        return []

    # Prefer proper-name spans from the answer text (general, non-question-specific).
    stop = {
        "agreement", "party", "parties", "services", "terms", "term",
        "section", "order form", "input", "output", "customer content",
    }
    entities: list[str] = []
    for m in re.finditer(r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*\b", raw):
        ent = re.sub(r"\s+", " ", m.group(0)).strip(" ,.;:()[]{}\"'").lower()
        if not ent or ent in stop:
            continue
        if ent in {"in", "this", "that", "the", "and", "or"}:
            continue
        if ent not in entities:
            entities.append(ent)

    return entities[:4]


def _is_entity_definition_direct_clause(text: str, answer_entities: list[str]) -> bool:
    raw_text = (text or "").strip()
    t = raw_text.lower()
    if not t:
        return False
    if not _ENTITY_REL_CLAUSE_RE.search(t):
        return False
    if answer_entities:
        matched = [ent for ent in answer_entities if ent in t]
        required = 1 if len(answer_entities) == 1 else min(2, len(answer_entities))
        if len(matched) < required:
            return False
    else:
        # If answer entities are unavailable, require explicit named entities in clause text.
        named = re.findall(r"\b[A-Z][A-Za-z0-9&.'-]*\b", raw_text)
        named_norm = []
        for n in named:
            nn = n.strip().lower()
            if nn in {"party", "parties", "agreement", "section", "services", "terms"}:
                continue
            if nn not in named_norm:
                named_norm.append(nn)
        if len(named_norm) < 2:
            return False
    # Explicitly reject Party/parties-only references without named entities.
    if re.search(r"\bpart(?:y|ies)\b", t):
        named_like = [ent for ent in answer_entities if ent and ent not in {"party", "parties"}]
        if named_like and not any(ent in t for ent in named_like):
            return False
    return True


def _extract_entity_definition_span(text: str, answer_entities: list[str]) -> str:
    """
    Display-layer helper: for entity/definition questions, pick the shortest
    direct supporting span from potentially aggregated text blobs.
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    candidates: list[str] = []
    # Bundled definition blobs often use pipes; split first to isolate clauses.
    if "|" in raw:
        for part in raw.split("|"):
            p = part.strip()
            if p:
                candidates.append(p)

    # Also consider sentence-like spans from the full text.
    sentence_parts = re.split(r"(?<=[.!?;])\s+|\s+\|\s+", raw)
    for part in sentence_parts:
        p = part.strip()
        if p:
            candidates.append(p)

    # Always include the raw text as fallback.
    candidates.append(raw)

    # Keep unique, preserve order.
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    direct = [c for c in unique if _is_entity_definition_direct_clause(c, answer_entities)]
    if direct:
        # Prefer the shortest direct span that still carries full meaning.
        direct.sort(key=lambda s: (len(s), s.count("|"), s.count(":")))
        best = direct[0]
        # Drop leading definition labels like "Party:" for cleaner display.
        best = re.sub(r"^[A-Za-z][A-Za-z0-9'’ _-]{0,40}:\s*", "", best).strip()
        return best
    return raw


def _score_support_node(
    node: dict,
    *,
    question_terms: set[str],
    answer_terms: set[str],
    answer_text: str,
    is_definition_question: bool,
    answer_entities: list[str],
) -> dict | None:
    text = (node.get("source_text") or "").strip()
    if not text:
        return None

    if is_definition_question:
        narrowed = _extract_entity_definition_span(text, answer_entities)
        if narrowed:
            text = narrowed

    text_terms = _display_terms(text)
    if not text_terms:
        return None

    source = node.get("_section") or f"Chunk {node.get('_chunk_id', '?')}"
    node_type = str(node.get("type") or "").upper()
    text_lower = text.lower()
    has_definition_cue = bool(_DEF_CLAUSE_RE.search(text))
    has_indirect_mention = bool(_INDIRECT_ENTITY_RE.search(text))
    is_full_sentence = _is_full_sentence_text(text)
    is_entity_relation_clause = _is_entity_definition_direct_clause(text, answer_entities)

    q_overlap = len(text_terms & question_terms) / max(1, len(question_terms))
    a_overlap = len(text_terms & answer_terms) / max(1, len(answer_terms)) if answer_terms else 0.0
    phrase_match = bool(answer_text and answer_text.lower() in text_lower)

    type_bonus = {
        "DEFINITION": 1.5,
        "OBLIGATION": 1.1,
        "RIGHT": 1.1,
        "CONDITION": 0.8,
        "NUMERIC": 0.7,
    }.get(node_type, 0.0)

    retrieval_score = float(node.get("_score") or 0.0)
    vote_frac = float(node.get("_vote_frac") or 0.0)

    direct_strength = (3.0 if phrase_match else 0.0) + (2.0 * a_overlap) + q_overlap
    score = (
        direct_strength * 6.0
        + type_bonus
        + (1.2 if has_definition_cue else 0.0)
        + (1.0 if is_full_sentence else -1.6)
        + retrieval_score * 1.4
        + vote_frac * 1.2
        - (0.6 if len(text) < 40 else 0.0)
        - (1.6 if has_indirect_mention and not has_definition_cue else 0.0)
    )
    if is_definition_question:
        # Penalize long aggregated text when shorter direct support exists.
        score -= min(2.0, max(0.0, (len(text) - 180) / 140.0))
        if text.count("|") >= 2:
            score -= 1.8

    is_definition = node_type == "DEFINITION"
    is_direct = bool(phrase_match or a_overlap >= 0.5 or (a_overlap >= 0.3 and q_overlap >= 0.12))
    is_entity_direct = bool(
        is_entity_relation_clause
        and (is_definition or q_overlap >= 0.14)
    )
    is_weak = bool(
        not is_direct
        and q_overlap < 0.08
        and a_overlap < 0.08
        and retrieval_score < 0.55
        and not is_definition
    )
    if not is_full_sentence and not (is_definition_question and is_entity_direct):
        is_weak = True

    # For definition/entity questions, only allow explicit naming/definition text.
    if is_definition_question and not is_entity_direct:
        is_weak = True

    return {
        "key": text.lower(),
        "text": text,
        "source": str(source),
        "score": score,
        "q_overlap": q_overlap,
        "a_overlap": a_overlap,
        "is_definition": is_definition,
        "is_direct": is_direct,
        "phrase_match": phrase_match,
        "has_definition_cue": has_definition_cue,
        "has_indirect_mention": has_indirect_mention,
        "is_full_sentence": is_full_sentence,
        "is_entity_relation_clause": is_entity_relation_clause,
        "is_entity_direct": is_entity_direct,
        "is_weak": is_weak,
    }


def _pick_supporting_clauses(result: dict, limit: int = 6) -> tuple[list[dict[str, str]], dict[str, Any]]:
    question = str(result.get("question") or "").strip()
    answer_text = str(result.get("answer") or "").strip()
    question_terms = _display_terms(question)
    answer_terms = _display_terms(answer_text)
    is_definition_question = _is_definition_entity_question(result)
    answer_entities = _extract_answer_entities(answer_text)

    seen: set[str] = set()
    scored: list[dict[str, Any]] = []
    scan_limit = 200 if is_definition_question else 40
    for node in result.get("surviving_nodes", [])[:scan_limit]:
        ranked = _score_support_node(
            node,
            question_terms=question_terms,
            answer_terms=answer_terms,
            answer_text=answer_text,
            is_definition_question=is_definition_question,
            answer_entities=answer_entities,
        )
        if not ranked:
            continue
        if is_definition_question and not ranked["is_entity_direct"]:
            continue
        if ranked["key"] in seen:
            continue
        seen.add(ranked["key"])
        scored.append(ranked)

    scored.sort(key=lambda item: item["score"], reverse=True)
    # For entity/definition questions, show a single exact proof clause only.
    if is_definition_question:
        strict_candidates = [
            s for s in scored
            if s["is_entity_direct"] and not s["is_weak"]
        ]
        if strict_candidates:
            strict_candidates.sort(key=lambda item: item["score"], reverse=True)
            best = strict_candidates[0]
            clauses = [{"text": best["text"], "source": best["source"]}]
            context = {
                "has_direct_support": True,
                "has_definition_support": True,
                "used_fallback": False,
            }
            return clauses, context
        # No strict direct evidence found: do not show indirect clauses.
        return [], {
            "has_direct_support": False,
            "has_definition_support": False,
            "used_fallback": True,
        }

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    used_fallback = False
    single_clause_answer = False

    # If one strong full sentence fully answers the question, show only that.
    for cand in scored:
        fully_answers = bool(
            cand["is_full_sentence"]
            and (
                cand["phrase_match"]
                or (cand["a_overlap"] >= 0.72 and cand["q_overlap"] >= 0.20)
                or (is_definition_question and cand["has_definition_cue"] and cand["q_overlap"] >= 0.20)
            )
        )
        if fully_answers:
            selected = [cand]
            selected_keys = {cand["key"]}
            single_clause_answer = True
            break

    if single_clause_answer:
        clauses = [{"text": selected[0]["text"], "source": selected[0]["source"]}]
        context = {
            "has_direct_support": True,
            "has_definition_support": bool(selected[0]["is_definition"] or selected[0]["has_definition_cue"]),
            "used_fallback": False,
        }
        return clauses, context

    direct_candidates = [s for s in scored if s["is_direct"] and not s["is_weak"]]
    if direct_candidates:
        top_direct = direct_candidates[0]
        selected.append(top_direct)
        selected_keys.add(top_direct["key"])

    definition_candidates = [
        s for s in scored
        if (
            s["is_definition"]
            and s["q_overlap"] >= (0.12 if is_definition_question else 0.08)
            and s["key"] not in selected_keys
            and not s["is_weak"]
        )
    ]
    if definition_candidates and len(selected) < limit:
        top_definition = definition_candidates[0]
        selected.append(top_definition)
        selected_keys.add(top_definition["key"])

    for item in scored:
        if len(selected) >= limit:
            break
        if item["key"] in selected_keys:
            continue
        if item["is_weak"]:
            continue
        if not item["is_full_sentence"]:
            continue
        selected.append(item)
        selected_keys.add(item["key"])

    if not selected and scored:
        full_sentence_scored = [s for s in scored if s["is_full_sentence"]]
        selected = [full_sentence_scored[0] if full_sentence_scored else scored[0]]
        used_fallback = True

    clauses = [{"text": item["text"], "source": item["source"]} for item in selected[:limit]]
    context = {
        "has_direct_support": any(item["is_direct"] for item in selected),
        "has_definition_support": any(item["is_definition"] for item in selected),
        "used_fallback": used_fallback,
    }
    return clauses, context


def _shape_response(result: dict) -> dict:
    answerable = bool(result.get("answerable", False))
    is_definition_question = _is_definition_entity_question(result)
    clauses: list[dict[str, str]] = []
    evidence_context: dict[str, Any] = {
        "has_direct_support": False,
        "has_definition_support": False,
        "used_fallback": False,
    }
    if answerable:
        clauses, evidence_context = _pick_supporting_clauses(result, limit=6)
        if not clauses and not is_definition_question:
            # Serialization fallback: if answerable and we have surviving node text,
            # attach at least one readable evidence snippet.
            for node in result.get("surviving_nodes", [])[:40]:
                text = _normalize_clause_text(node.get("source_text"))
                if not text:
                    continue
                if not _is_full_sentence_text(text):
                    continue
                source = node.get("_section") or f"Chunk {node.get('_chunk_id', '?')}"
                clauses = [{"text": text, "source": str(source)}]
                evidence_context["used_fallback"] = True
                break
    short_answer: str | None = None
    if answerable:
        raw_answer = str(result.get("answer") or "").strip()
        if raw_answer:
            short_answer = raw_answer
        elif clauses:
            # Display-layer fallback so answerable responses never return null short answers.
            short_answer = str(clauses[0].get("text") or "").strip() or None
        else:
            reasoning_fallback = str(result.get("reasoning") or "").strip()
            short_answer = reasoning_fallback or "Answerable based on supporting text."

    if answerable:
        if evidence_context["has_direct_support"]:
            reason = "The agreement states this directly in the supporting text below."
        elif evidence_context["has_definition_support"]:
            reason = "The document explicitly identifies the relevant party or term in the supporting text below."
        elif evidence_context["used_fallback"]:
            reason = "The answer is inferred from the strongest supporting text retrieved below, but the support is less direct than ideal."
        else:
            reason = "The answer is supported by the document text shown below."
    else:
        reason = (result.get("reasoning") or "").strip() or "No evidence in the uploaded contract supports this answer."

    topo = result.get("topo_pred") or {}
    gate = result.get("answerability_gate") or {}

    metadata: dict[str, Any] = {
        "confidence": result.get("confidence"),
        "topology_score": topo.get("score"),
        "topology_predicted": topo.get("predicted"),
        "gate_failures": gate.get("failures") or [],
        "slot_score": result.get("slot_score"),
        "vote_fraction": result.get("vote_fraction"),
    }
    if gate:
        metadata["gate_mode"] = gate.get("mode")
        metadata["llm_answerable_raw"] = result.get("llm_answerable_raw")
        metadata["llm_confidence_raw"] = result.get("llm_confidence_raw")
    if topo.get("score") is not None:
        metadata["topology_confidence"] = topo.get("confidence")
        metadata["topology_threshold"] = topo.get("threshold")

    return {
        "answerable": answerable,
        "short_answer": short_answer,
        "reason": reason,
        # Canonical evidence field consumed by the UI.
        "supporting_clauses": clauses if answerable else [],
        # Backward-compatible alias for any callers still reading `clauses`.
        "clauses": clauses if answerable else [],
        "metadata": metadata,
    }


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pick_topology_snapshot(raw: dict) -> dict:
    topo = raw.get("topology") or {}
    return {
        "n_nodes": topo.get("n_nodes"),
        "n_edges": topo.get("n_edges"),
        "typed_path_coverage": topo.get("typed_path_coverage", {}).get("score"),
        "answer_path": {
            "slot_type": topo.get("answer_path", {}).get("slot_type"),
            "path_exists": topo.get("answer_path", {}).get("path_exists"),
            "path_bottleneck": topo.get("answer_path", {}).get("path_bottleneck"),
            "path_score": topo.get("answer_path", {}).get("path_score"),
            "n_targets": topo.get("answer_path", {}).get("n_targets"),
            "n_well_connected": topo.get("answer_path", {}).get("n_well_connected"),
        },
        "persistent_homology": {
            "h0_mean_lifetime": topo.get("persistent_homology", {}).get("h0_mean_lifetime"),
        },
        "chain": {
            "chain_strength": topo.get("chain", {}).get("chain_strength"),
            "trigger_edge_weight": topo.get("chain", {}).get("trigger_edge_weight"),
            "conditions_required": topo.get("chain", {}).get("conditions_required"),
            "reachable_conditions": topo.get("chain", {}).get("reachable_conditions"),
        },
        "sheaf": {
            "consistent_frac": topo.get("sheaf", {}).get("consistent_frac"),
        },
        "weighted": {
            "weighted_consistent_frac": topo.get("weighted", {}).get("weighted_consistent_frac"),
            "weighted_mean": topo.get("weighted", {}).get("weighted_mean"),
            "anchor_bridge_score": topo.get("weighted", {}).get("anchor_bridge_score"),
        },
        "type_metrics": {
            "type_entropy": topo.get("type_metrics", {}).get("type_entropy"),
        },
        "slot_mismatch_depth": {
            "exact_frac": topo.get("slot_mismatch_depth", {}).get("exact_frac"),
            "adjacent_frac": topo.get("slot_mismatch_depth", {}).get("adjacent_frac"),
            "distant_frac": topo.get("slot_mismatch_depth", {}).get("distant_frac"),
        },
        "required_sequence": {
            "sequence_frac": topo.get("required_sequence", {}).get("sequence_frac"),
        },
        "direction": {
            "party_mismatch_frac": topo.get("direction", {}).get("party_mismatch_frac"),
            "edge_role_inversion": topo.get("direction", {}).get("edge_role_inversion"),
        },
        "value_tag_match": {
            "has_tags": topo.get("value_tag_match", {}).get("has_tags"),
            "match_score": topo.get("value_tag_match", {}).get("match_score"),
        },
        "placeholder_values": {
            "placeholder_frac": topo.get("placeholder_values", {}).get("placeholder_frac"),
        },
        "keyword_trap": {
            "flagged": topo.get("keyword_trap", {}).get("flagged"),
            "confidence": topo.get("keyword_trap", {}).get("confidence"),
        },
    }


def _append_topology_event(event: dict[str, Any]) -> None:
    line = json.dumps(event, ensure_ascii=False)
    with _telemetry_lock:
        with TOPOLOGY_TELEMETRY_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _record_topology_event(
    *,
    question: str,
    pdf_path: Path,
    doc_token: str | None,
    raw: dict | None,
    shaped: dict | None,
    latency_ms: int,
    error: str | None = None,
) -> None:
    topo_pred = (raw or {}).get("topo_pred") or {}
    event = {
        "ts_utc": _now_iso_utc(),
        "document": {
            "name": pdf_path.name,
            "path": str(pdf_path.resolve()),
            "token": doc_token,
        },
        "question": question,
        "latency_ms": latency_ms,
        "error": error,
        "llm": {
            "answerable": (raw or {}).get("llm_answerable_raw", (raw or {}).get("answerable")),
            "confidence": (raw or {}).get("llm_confidence_raw", (raw or {}).get("confidence")),
            "slot_score": (raw or {}).get("slot_score"),
            "slot_verdict": (raw or {}).get("slot_verdict"),
        },
        "gate": (raw or {}).get("answerability_gate"),
        "retrieval": {
            "variants_count": len((raw or {}).get("variants") or []),
            "all_nodes_seen": (raw or {}).get("all_nodes_seen"),
            "surviving_nodes": len((raw or {}).get("surviving_nodes") or []),
        },
        "topology_pred": {
            "predicted": topo_pred.get("predicted"),
            "score": topo_pred.get("score"),
            "threshold": topo_pred.get("threshold"),
            "confidence": topo_pred.get("confidence"),
            "signals": topo_pred.get("signals"),
            "meta": topo_pred.get("meta"),
        },
        "topology_snapshot": _pick_topology_snapshot(raw or {}),
    }
    if shaped is not None:
        event["api_response"] = {
            "answerable": shaped.get("answerable"),
            "reason": shaped.get("reason"),
            "metadata": shaped.get("metadata"),
        }
    _append_topology_event(event)


def _read_recent_topology_events(limit: int = 100) -> list[dict[str, Any]]:
    if not TOPOLOGY_TELEMETRY_PATH.exists():
        return []
    try:
        lines = TOPOLOGY_TELEMETRY_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _resolve_pdf(
    *,
    sample_name: str | None,
    file: UploadFile | None,
    doc_token: str | None = None,
) -> Path:
    if doc_token:
        p = _doc_tokens.get(doc_token)
        if not p:
            raise HTTPException(status_code=404, detail="Prepared document token not found.")
        return p

    if file is not None and file.filename:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
        filename = f"{int(time.time())}_{_safe_name(file.filename)}"
        out = UPLOAD_DIR / filename
        payload = file.file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        out.write_bytes(payload)
        return out

    if sample_name:
        candidate = SAMPLE_DIR / Path(sample_name).name
        if not candidate.exists():
            raise HTTPException(status_code=404, detail="Sample document not found.")
        return candidate

    docs = _collect_demo_docs()
    if len(docs) == 1:
        return Path(docs[0]["path"])

    raise HTTPException(
        status_code=400,
        detail="Upload a PDF or select a sample document.",
    )


def _run_prepare_job(job_id: str, pdf_path: Path) -> None:
    try:
        with _job_lock:
            _prep_jobs[job_id]["status"] = "running"
            _prep_jobs[job_id]["started_at"] = time.time()
            _prep_jobs[job_id]["metrics"] = {
                "stage": "starting",
                "total_chunks": 0,
                "extracted_chunks": 0,
                "nodes_created": 0,
                "llm_chunks_seen": 0,
                "llm_calls_actual": 0,
                "first_extract_at": None,
                "done_extract_at": None,
                "embedding_started_at": None,
            }

        # Fast path: reuse in-memory prepared pipeline when available.
        if cache.has(pdf_path):
            pipeline = cache.get(pdf_path)
            token = uuid4().hex
            _register_prepared_doc(token, pdf_path)
            with _job_lock:
                _prep_jobs[job_id]["status"] = "done"
                _prep_jobs[job_id]["doc_token"] = token
                _prep_jobs[job_id]["finished_at"] = time.time()
                _prep_jobs[job_id]["metrics"]["stage"] = "done_extract"
                _prep_jobs[job_id]["metrics"]["llm_calls_actual"] = 0
                _prep_jobs[job_id]["metrics"]["done_extract_at"] = _prep_jobs[job_id]["finished_at"]
            return

        reset_llm_call_count()

        def _progress_cb(payload: dict[str, Any]) -> None:
            with _job_lock:
                job = _prep_jobs.get(job_id)
                if not job:
                    return
                metrics = job.setdefault("metrics", {})
                for k in ("stage", "total_chunks", "extracted_chunks", "nodes_created", "llm_chunks_seen"):
                    if k in payload:
                        metrics[k] = payload[k]
                stage = str(metrics.get("stage") or "").strip().lower()
                now = time.time()
                if stage == "extracting" and not metrics.get("first_extract_at"):
                    metrics["first_extract_at"] = now
                if stage == "done_extract" and not metrics.get("done_extract_at"):
                    metrics["done_extract_at"] = now
                if stage == "embedding_nodes" and not metrics.get("embedding_started_at"):
                    metrics["embedding_started_at"] = now
                metrics["llm_calls_actual"] = get_llm_call_count()

        pipeline = load_pdf(pdf_path, progress_cb=_progress_cb)
        cache.set(pdf_path, pipeline)
        token = uuid4().hex
        _register_prepared_doc(token, pdf_path)
        with _job_lock:
            _prep_jobs[job_id]["status"] = "done"
            _prep_jobs[job_id]["doc_token"] = token
            _prep_jobs[job_id]["finished_at"] = time.time()
            _prep_jobs[job_id]["metrics"]["llm_calls_actual"] = get_llm_call_count()
            started_at = float(_prep_jobs[job_id].get("started_at") or 0.0)
            finished_at = float(_prep_jobs[job_id].get("finished_at") or time.time())
            n_pages = int(_prep_jobs[job_id].get("pages") or 0)
            m = _prep_jobs[job_id].get("metrics") or {}
            total_chunks = int(m.get("total_chunks") or 0)
            first_extract_at = m.get("first_extract_at")
            done_extract_at = m.get("done_extract_at")
            nodes_created = int(m.get("nodes_created") or 0)
            llm_calls_actual = int(m.get("llm_calls_actual") or 0)
        if started_at > 0 and finished_at >= started_at:
            _record_prep_runtime(n_pages=n_pages, elapsed_sec=finished_at - started_at)
            _record_phase_runtime(
                total_chunks=total_chunks,
                started_at=started_at,
                first_extract_at=float(first_extract_at) if first_extract_at else None,
                done_extract_at=float(done_extract_at) if done_extract_at else None,
                nodes_created=nodes_created,
                llm_calls_actual=llm_calls_actual,
                finished_at=finished_at,
            )
    except Exception as exc:
        with _job_lock:
            _prep_jobs[job_id]["status"] = "error"
            _prep_jobs[job_id]["error"] = str(exc)
            _prep_jobs[job_id]["finished_at"] = time.time()
            _prep_jobs[job_id].setdefault("metrics", {})["llm_calls_actual"] = get_llm_call_count()


@app.get("/")
def root():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/health")
def health(request: Request) -> dict:
    _enforce_rate_limit(request, "health", _rate_limit_general)
    return {"ok": True}


@app.get("/api/debug/runtime")
def debug_runtime(request: Request) -> dict:
    _enforce_access(request)
    _enforce_rate_limit(request, "debug_runtime", _rate_limit_general)
    llm_preflight = _llm_preflight_error()
    with _ask_debug_lock:
        ask_debug = dict(_ask_debug)
    return {
        "ok": True,
        "uptime_sec": int(max(0.0, time.time() - _server_started_at)),
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
        "llm_preflight_error": llm_preflight,
        "cache": cache.stats(),
        "prepared_docs_count": len(_prepared_index),
        "doc_tokens_count": len(_doc_tokens),
        "active_prepare_jobs": sum(1 for j in _prep_jobs.values() if j.get("status") in {"queued", "running"}),
        "ask_debug": ask_debug,
    }


@app.get("/api/demo-docs")
def demo_docs(request: Request) -> dict:
    _enforce_access(request)
    _enforce_rate_limit(request, "demo_docs", _rate_limit_general)
    docs = _collect_demo_docs()
    # Attach token if the doc is already prepared (enables instant load in UI)
    name_to_token: dict[str, str] = {
        str(item.get("name") or ""): str(item.get("token") or "")
        for item in _prepared_index
        if item.get("token") and item.get("name")
    }
    for doc in docs:
        doc["token"] = name_to_token.get(doc["name"], "")
    return {"documents": docs}


@app.get("/api/saved-docs")
def saved_docs(request: Request) -> dict:
    _enforce_access(request)
    _enforce_rate_limit(request, "saved_docs", _rate_limit_general)
    return {"documents": _list_saved_docs()}


@app.get("/api/prepared-doc/{doc_token}")
def prepared_doc(doc_token: str, request: Request, access_code: str | None = None):
    _enforce_access(request, access_code=access_code)
    _enforce_rate_limit(request, "prepared_doc", _rate_limit_general)
    pdf_path = _resolve_pdf(sample_name=None, file=None, doc_token=doc_token)
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)


@app.get("/api/prepared-nodes/{doc_token}")
def prepared_nodes(doc_token: str, request: Request, access_code: str | None = None):
    _enforce_access(request, access_code=access_code)
    _enforce_rate_limit(request, "prepared_nodes", _rate_limit_general)
    pdf_path = _resolve_pdf(sample_name=None, file=None, doc_token=doc_token)
    nodes_path = pdf_path.parent / f"{pdf_path.stem}_typed_nodes.json"
    if not nodes_path.exists():
        raise HTTPException(status_code=404, detail="Nodes JSON not found for this document.")
    return FileResponse(str(nodes_path), media_type="application/json", filename=nodes_path.name)


@app.get("/api/sample-doc/{sample_name:path}")
def sample_doc(sample_name: str, request: Request, access_code: str | None = None):
    _enforce_access(request, access_code=access_code)
    _enforce_rate_limit(request, "sample_doc", _rate_limit_general)
    pdf_path = _resolve_pdf(sample_name=sample_name, file=None, doc_token=None)
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)


@app.get("/api/sample-nodes/{sample_name:path}")
def sample_nodes(sample_name: str, request: Request, access_code: str | None = None):
    _enforce_access(request, access_code=access_code)
    _enforce_rate_limit(request, "sample_nodes", _rate_limit_general)
    pdf_path = _resolve_pdf(sample_name=sample_name, file=None, doc_token=None)
    nodes_path = pdf_path.parent / f"{pdf_path.stem}_typed_nodes.json"
    if not nodes_path.exists():
        raise HTTPException(status_code=404, detail="Nodes JSON not found for this sample.")
    return FileResponse(str(nodes_path), media_type="application/json", filename=nodes_path.name)


@app.get("/api/topology-telemetry")
def topology_telemetry(request: Request, limit: int = 100) -> dict:
    _enforce_access(request)
    _enforce_rate_limit(request, "topology_telemetry", _rate_limit_general)
    lim = max(1, min(1000, int(limit)))
    events = _read_recent_topology_events(lim)
    return {
        "events": events,
        "count": len(events),
        "path": str(TOPOLOGY_TELEMETRY_PATH),
    }


@app.post("/api/prepare-doc")
def prepare_doc(
    request: Request,
    sample_name: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
) -> dict:
    _enforce_access(request)
    _enforce_rate_limit(request, "prepare_doc", _rate_limit_prepare)
    preflight_err = _llm_preflight_error()
    if preflight_err:
        raise HTTPException(status_code=400, detail=preflight_err)

    pdf_path = _resolve_pdf(sample_name=sample_name, file=file)
    n_pages = _pdf_page_count(pdf_path)
    estimate_sec = _estimate_prep_seconds(pdf_path)
    job_id = uuid4().hex
    with _job_lock:
        _prep_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "estimate_sec": estimate_sec,
            "pages": n_pages,
            "document": pdf_path.name,
            "error": None,
            "doc_token": None,
            "metrics": {
                "stage": "queued",
                "total_chunks": 0,
                "extracted_chunks": 0,
                "nodes_created": 0,
                "llm_chunks_seen": 0,
                "llm_calls_actual": 0,
                "first_extract_at": None,
                "done_extract_at": None,
                "embedding_started_at": None,
            },
        }
    t = Thread(target=_run_prepare_job, args=(job_id, pdf_path), daemon=True)
    t.start()
    return {
        "job_id": job_id,
        "estimate_sec": estimate_sec,
        "document": pdf_path.name,
        "message": "Document preparation started.",
    }


@app.get("/api/prepare-doc/{job_id}")
def prepare_doc_status(job_id: str, request: Request) -> dict:
    _enforce_access(request)
    _enforce_rate_limit(request, "prepare_status", _rate_limit_general)
    with _job_lock:
        job = _prep_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        payload = dict(job)

    status = payload["status"]
    estimate = max(1, int(payload.get("estimate_sec") or 30))
    now_ts = time.time()
    started = payload.get("started_at")
    elapsed = 0.0
    if started:
        elapsed = max(0.0, now_ts - float(started))
    elif payload.get("created_at"):
        elapsed = max(0.0, now_ts - float(payload["created_at"]))

    metrics = payload.get("metrics") or {}
    total_chunks = int(metrics.get("total_chunks") or 0)
    extracted_chunks = int(metrics.get("extracted_chunks") or 0)
    nodes_created = int(metrics.get("nodes_created") or 0)
    llm_calls_actual = int(metrics.get("llm_calls_actual") or 0)
    if total_chunks > 0:
        llm_calls_estimated_total = total_chunks + 3
    else:
        llm_calls_estimated_total = int(payload.get("estimate_sec", 0) / 8) + 3
    llm_calls_estimated_total = max(llm_calls_estimated_total, llm_calls_actual, 1)
    chunk_progress = min(1.0, extracted_chunks / total_chunks) if total_chunks > 0 else 0.0
    llm_progress = min(1.0, llm_calls_actual / llm_calls_estimated_total)
    stage = str(metrics.get("stage") or "").strip().lower()
    first_extract_at = float(metrics.get("first_extract_at")) if metrics.get("first_extract_at") else None
    done_extract_at = float(metrics.get("done_extract_at")) if metrics.get("done_extract_at") else None
    embedding_started_at = float(metrics.get("embedding_started_at")) if metrics.get("embedding_started_at") else None
    with _prep_runtime_lock:
        setup_sec_ema = float(_prep_phase_model.get("setup_sec_ema") or 10.0)
        sec_per_chunk_ema = float(_prep_phase_model.get("sec_per_chunk_ema") or 2.2)
        post_base_sec_ema = float(_prep_phase_model.get("post_base_sec_ema") or 6.0)
        post_sec_per_1k_nodes_ema = float(_prep_phase_model.get("post_sec_per_1k_nodes_ema") or 10.0)
        post_sec_per_llm_call_ema = float(_prep_phase_model.get("post_sec_per_llm_call_ema") or 1.2)

    # Stage-aware floor used to stabilize progress display.
    stage_floor = {
        "queued": 0.01,
        "start_load": 0.03,
        "metadata": 0.08,
        "extracting": 0.14,
        "retrying_zero_chunks": 0.78,
        "metadata_injected": 0.82,
        "gap_audit": 0.84,
        "entity_resolution": 0.87,
        "summary_injected": 0.88,
        "expand_condition_numerics": 0.90,
        "signature_extract": 0.93,
        "graph_build": 0.90,
        "done_extract": 0.93,
        "embedding_nodes": 0.98,
        "ready": 0.995,
        "load_cached_artifacts": 0.90,
    }.get(stage, 0.05)
    observed_progress = max(stage_floor, 0.10 + 0.78 * chunk_progress + 0.12 * llm_progress)

    # Rate-based ETA model: setup + remaining extraction chunks + post-extraction tail.
    remaining_model: int | None = None
    if status == "running":
        rem_setup = 0.0
        rem_extract = 0.0

        if not first_extract_at:
            rem_setup = max(0.0, setup_sec_ema - elapsed)

        if total_chunks > 0 and stage not in {"done_extract", "embedding_nodes", "ready"}:
            rem_chunks = max(0, total_chunks - extracted_chunks)
            observed_sec_per_chunk = None
            if first_extract_at and extracted_chunks > 0:
                extract_elapsed = max(0.0, now_ts - first_extract_at)
                if extract_elapsed >= 0.8:
                    observed_sec_per_chunk = extract_elapsed / max(1, extracted_chunks)
            if observed_sec_per_chunk is None:
                sec_per_chunk_est = sec_per_chunk_ema
            elif extracted_chunks >= 5:
                sec_per_chunk_est = 0.70 * observed_sec_per_chunk + 0.30 * sec_per_chunk_ema
            else:
                sec_per_chunk_est = 0.40 * observed_sec_per_chunk + 0.60 * sec_per_chunk_ema
            sec_per_chunk_est = max(0.3, min(30.0, sec_per_chunk_est))
            rem_extract = rem_chunks * sec_per_chunk_est

        # Estimate total nodes from observed extraction density.
        if total_chunks > 0 and extracted_chunks > 0:
            nodes_per_chunk = nodes_created / max(1, extracted_chunks)
            predicted_total_nodes = int(max(nodes_created, round(nodes_per_chunk * total_chunks)))
        else:
            # Fallback prior when no chunk data is available yet.
            predicted_total_nodes = max(nodes_created, total_chunks * 35)
        llm_calls_est_total = max(llm_calls_estimated_total, llm_calls_actual)
        post_target = (
            post_base_sec_ema
            + max(
                post_sec_per_1k_nodes_ema * (max(0, predicted_total_nodes) / 1000.0),
                post_sec_per_llm_call_ema * max(0, llm_calls_est_total),
            )
        )

        if stage in {"embedding_nodes", "ready"}:
            if embedding_started_at:
                rem_post = max(0.0, post_target - (now_ts - embedding_started_at))
            else:
                rem_post = max(0.0, post_target * 0.8)
        elif done_extract_at:
            rem_post = max(0.0, post_target - (now_ts - done_extract_at))
        else:
            rem_post = post_target

        remaining_model = int(math.ceil(max(0.0, rem_setup + rem_extract + rem_post)))

    if status == "queued":
        progress = 0.02
        remaining_sec: int | None = estimate
        over_eta_sec = 0
    elif status == "running":
        if remaining_model is not None:
            remaining_sec = remaining_model
            if stage not in {"embedding_nodes", "ready"}:
                remaining_sec = max(12, remaining_sec)
            model_progress = elapsed / max(1.0, (elapsed + float(remaining_sec)))
            progress = min(0.99, max(observed_progress, model_progress))
            over_eta_sec = 0
            if elapsed > estimate and remaining_sec > 20:
                over_eta_sec = int(elapsed - estimate)
        else:
            # Final fallback when no useful model signal exists.
            if elapsed <= estimate:
                progress = min(0.90, 0.05 + 0.85 * (elapsed / estimate))
                remaining_sec = max(0, estimate - int(elapsed))
                over_eta_sec = 0
            else:
                overtime = elapsed - estimate
                tau = max(30.0, estimate * 0.6)
                progress = min(0.99, 0.90 + 0.09 * (1.0 - math.exp(-overtime / tau)))
                remaining_sec = None
                over_eta_sec = int(overtime)
    elif status == "done":
        progress = 1.0
        remaining_sec = 0
        over_eta_sec = 0
    else:
        progress = 0.0
        remaining_sec = 0
        over_eta_sec = 0

    payload["progress"] = round(progress, 4)
    payload["elapsed_sec"] = int(elapsed)
    payload["remaining_sec"] = remaining_sec
    payload["over_eta_sec"] = over_eta_sec
    if status == "running" and over_eta_sec > 0:
        payload["note"] = "Over ETA; still processing extraction and graph build."
    warn_after = max(estimate * 3, 900)  # 15 min minimum.
    if status == "running" and elapsed > warn_after:
        payload["warning"] = "Still running far past estimate. Check API key/network model availability."
    if total_chunks > 0:
        payload["chunk_progress"] = round(chunk_progress, 4)
    else:
        payload["chunk_progress"] = 0.0

    payload["llm_progress"] = round(llm_progress, 4)
    payload["llm_calls_estimated_total"] = llm_calls_estimated_total
    return payload


@app.post("/api/ask")
def ask(
    request: Request,
    question: str = Form(...),
    doc_token: str | None = Form(default=None),
) -> dict:
    _enforce_access(request)
    _enforce_rate_limit(request, "ask", _rate_limit_ask)
    t0 = time.time()
    _mark_ask_start()
    q = (question or "").strip()
    if not q:
        _mark_ask_end(latency_ms=int((time.time() - t0) * 1000), error="Question is required.")
        raise HTTPException(status_code=400, detail="Question is required.")
    if not doc_token:
        _mark_ask_end(latency_ms=int((time.time() - t0) * 1000), error="Missing doc token.")
        raise HTTPException(status_code=400, detail="Please prepare a document before asking questions.")

    try:
        pdf_path = _resolve_pdf(sample_name=None, file=None, doc_token=doc_token)
        pipeline = cache.get(pdf_path)
        raw = run_query(pipeline, q)
        shaped = _shape_response(raw)
        shaped["document"] = pdf_path.name
        _mark_ask_end(latency_ms=int((time.time() - t0) * 1000))
        _record_topology_event(
            question=q,
            pdf_path=pdf_path,
            doc_token=doc_token,
            raw=raw,
            shaped=shaped,
            latency_ms=int((time.time() - t0) * 1000),
        )
        return shaped
    except HTTPException:
        _mark_ask_end(latency_ms=int((time.time() - t0) * 1000), error="HTTPException")
        raise
    except Exception as exc:
        _mark_ask_end(latency_ms=int((time.time() - t0) * 1000), error=str(exc))
        _record_topology_event(
            question=q,
            pdf_path=pdf_path,
            doc_token=doc_token,
            raw=None,
            shaped=None,
            latency_ms=int((time.time() - t0) * 1000),
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")


@app.post("/api/ask-batch")
def ask_batch(
    request: Request,
    questions: str = Form(...),
    doc_token: str | None = Form(default=None),
) -> dict:
    _enforce_access(request)
    _enforce_rate_limit(request, "ask_batch", _rate_limit_ask)
    t0 = time.time()
    _mark_ask_start()
    if not doc_token:
        _mark_ask_end(latency_ms=int((time.time() - t0) * 1000), error="Missing doc token.")
        raise HTTPException(status_code=400, detail="Please prepare a document before asking questions.")

    parsed = [q.strip() for q in (questions or "").splitlines() if q.strip()]
    if not parsed:
        _mark_ask_end(latency_ms=int((time.time() - t0) * 1000), error="Empty question batch.")
        raise HTTPException(status_code=400, detail="Provide at least one question (one per line).")
    if len(parsed) > 40:
        _mark_ask_end(latency_ms=int((time.time() - t0) * 1000), error="Batch size > 40.")
        raise HTTPException(status_code=400, detail="Maximum batch size is 40 questions.")

    try:
        pdf_path = _resolve_pdf(sample_name=None, file=None, doc_token=doc_token)
        pipeline = cache.get(pdf_path)
    except HTTPException:
        _mark_ask_end(latency_ms=int((time.time() - t0) * 1000), error="HTTPException while loading pipeline.")
        raise
    except Exception as exc:
        _mark_ask_end(latency_ms=int((time.time() - t0) * 1000), error=str(exc))
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    results: list[dict[str, Any]] = []
    for idx, q in enumerate(parsed, start=1):
        t0 = time.time()
        try:
            raw = run_query(pipeline, q)
            shaped = _shape_response(raw)
            shaped["document"] = pdf_path.name
            shaped["question"] = q
            shaped["index"] = idx
            results.append(shaped)
            _record_topology_event(
                question=q,
                pdf_path=pdf_path,
                doc_token=doc_token,
                raw=raw,
                shaped=shaped,
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as exc:
            failure = {
                "question": q,
                "index": idx,
                "document": pdf_path.name,
                "answerable": False,
                "short_answer": None,
                "reason": f"Pipeline error: {exc}",
                "supporting_clauses": [],
                "metadata": {},
            }
            results.append(failure)
            _record_topology_event(
                question=q,
                pdf_path=pdf_path,
                doc_token=doc_token,
                raw=None,
                shaped=failure,
                latency_ms=int((time.time() - t0) * 1000),
                error=str(exc),
            )

    answerable_count = sum(1 for r in results if r.get("answerable"))
    unanswerable_count = len(results) - answerable_count
    _mark_ask_end(latency_ms=int((time.time() - t0) * 1000))
    return {
        "document": pdf_path.name,
        "total": len(results),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "results": results,
    }


_load_prepared_index()
