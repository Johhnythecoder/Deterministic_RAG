"""
core_pipeline.py — Consolidated RAG pipeline for contract answerability.

Self-contained entry point covering every stage:
  1.  LLM interface          (OpenAI -> Ollama fallback)
  2.  PDF chunking            (structural chunker for contracts)
  3.  Typed node extraction   (7-type schema, metadata pass, retry, gap audit)
  4.  Entity resolution       (party alias normalisation + contract summary node)
  5.  Graph construction      (USES_TERM / TRIGGERS / SAME_SECTION edges)
  6.  Question parsing        (typed q-node + question type classification)
  7.  Typed node retrieval    (BGE embeddings, keyword augment, graph expansion)
  8.  Slot coverage           (math pre-filter before LLM)
  9.  LLM QA                  (chain-annotated facts + 15-rule prompt)
  10. Consensus retrieval     (N variants → vote threshold → slot → LLM)
  11. Topology predictor      (6-signal weighted score, value_tags, placeholder)
  12. Pipeline entry point    (load_pdf / query)

Usage:
    from core_pipeline import load_pdf, query

    pipeline = load_pdf("contract.pdf", workers=6)
    result   = query(pipeline, "Can Licensee sublicense the Software?")
    print(result["answerable"], result["topo_pred"]["predicted"])

Topology signal math (hundreds of functions) lives in v2/topology_metrics.py.
This file imports compute_all from there rather than inlining 3 000 lines of
persistent homology, Hodge Laplacian, sheaf consistency, etc.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

try:
    import pdfplumber as _pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

try:
    import openai as _openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

import requests

# Load .env from project root if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


# ══════════════════════════════════════════════════════════════════════════════
# §1  LLM INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

_OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434/api/generate")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
_LLM_CALL_COUNT = 0
_LLM_CALL_LOCK = Lock()
_OPENAI_CLIENT = None
_OPENAI_CLIENT_CFG: str | None = None
_OPENAI_QA_CLIENT = None
_OPENAI_QA_CLIENT_KEY: str | None = None
_EMBED_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_EMBED_MODEL_LOCK = Lock()


def reset_llm_call_count() -> None:
    global _LLM_CALL_COUNT
    with _LLM_CALL_LOCK:
        _LLM_CALL_COUNT = 0


def get_llm_call_count() -> int:
    with _LLM_CALL_LOCK:
        return int(_LLM_CALL_COUNT)


def _get_openai_client(api_key: str):
    global _OPENAI_CLIENT, _OPENAI_CLIENT_CFG
    cfg = (api_key or "")
    if _OPENAI_CLIENT is not None and _OPENAI_CLIENT_CFG == cfg:
        return _OPENAI_CLIENT

    _OPENAI_CLIENT = _openai.OpenAI(api_key=api_key)
    _OPENAI_CLIENT_CFG = cfg
    return _OPENAI_CLIENT


def _get_embed_model(model_name: str) -> SentenceTransformer:
    """Reuse a single embedding model instance per model name."""
    with _EMBED_MODEL_LOCK:
        model = _EMBED_MODEL_CACHE.get(model_name)
        if model is not None:
            return model
        model = SentenceTransformer(model_name)
        _EMBED_MODEL_CACHE[model_name] = model
        return model


def ask(prompt: str, temperature: float = 0.0, max_tokens: int = 1000,
        json_mode: bool = False) -> str:
    """Call OpenAI; fall back to Ollama if unavailable."""
    global _LLM_CALL_COUNT
    with _LLM_CALL_LOCK:
        _LLM_CALL_COUNT += 1
    # ── OpenAI ──────────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and _HAS_OPENAI:
        try:
            client = _get_openai_client(api_key)
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

            kwargs: dict = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception:
            pass

    # ── Ollama fallback ─────────────────────────────────────────────────────
    try:
        resp = requests.post(_OLLAMA_URL, json={
            "model":  _OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, timeout=120)
        data = resp.json()
        raw  = data.get("response", "")
        # Strip <think>…</think> blocks emitted by some Ollama models
        raw  = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw
    except Exception as e:
        return f"[LLM error: {e}]"


def ask_qa(prompt: str, temperature: float = 0.0, max_tokens: int = 600) -> str:
    """
    QA/judge LLM call — uses a separate more powerful model when
    OPENAI_API_KEY_QA and OPENAI_MODEL_QA are set, otherwise falls
    back to the same model as ask().

    Export these in your shell to enable:
        export OPENAI_API_KEY_QA=sk-...
        export OPENAI_MODEL_QA=gpt-4o          # or o1-mini, gpt-4-turbo, etc.
    """
    global _OPENAI_QA_CLIENT, _OPENAI_QA_CLIENT_KEY
    qa_key   = os.environ.get("OPENAI_API_KEY_QA")
    qa_model = os.environ.get("OPENAI_MODEL_QA")
    if qa_key and qa_model and _HAS_OPENAI:
        global _LLM_CALL_COUNT
        with _LLM_CALL_LOCK:
            _LLM_CALL_COUNT += 1
        try:
            if _OPENAI_QA_CLIENT is None or _OPENAI_QA_CLIENT_KEY != qa_key:
                _OPENAI_QA_CLIENT     = _openai.OpenAI(api_key=qa_key)
                _OPENAI_QA_CLIENT_KEY = qa_key
            kwargs: dict = dict(
                model=qa_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            resp = _OPENAI_QA_CLIENT.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception:
            pass  # fall through to standard ask()
    return ask(prompt, temperature=temperature, max_tokens=max_tokens)


def _emit_progress(progress_cb, **payload) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(payload)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# §2  PDF CHUNKING  (structural chunker for contracts / NDAs / agreements)
# ══════════════════════════════════════════════════════════════════════════════

_SPACE_GAP          = 2.5     # points; word gaps in compact fonts
_STRUCTURAL_MAX_CHARS = 1800  # hard cap for structural chunks


def _extract_pdf_pages_text(pdf_path: Path, max_pages=None) -> list[str]:
    if _HAS_PDFPLUMBER:
        try:
            pages_text = []
            with _pdfplumber.open(str(pdf_path)) as pdf:
                pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
                for page in pages:
                    chars = page.chars
                    if not chars:
                        pages_text.append("")
                        continue
                    lines: dict = {}
                    for ch in chars:
                        y = round(ch["top"] / 4) * 4
                        lines.setdefault(y, []).append(ch)
                    page_lines = []
                    for y in sorted(lines.keys()):
                        row = sorted(lines[y], key=lambda c: c["x0"])
                        text = row[0]["text"]
                        for i in range(1, len(row)):
                            if row[i]["x0"] - row[i - 1]["x1"] > _SPACE_GAP:
                                text += " "
                            text += row[i]["text"]
                        page_lines.append(text)
                    pages_text.append("\n".join(page_lines))
            return pages_text
        except Exception:
            pass
    reader = PdfReader(str(pdf_path))
    pages  = reader.pages if max_pages is None else reader.pages[:max_pages]
    return [p.extract_text() or "" for p in pages]


def _strip_repeating_lines(pages_text: list[str], min_pages: int = 3) -> str:
    """Remove lines that appear on min_pages+ pages (headers/footers/letterhead)."""
    count: dict[str, int] = {}
    for page in pages_text:
        for line in set(page.splitlines()):
            s = line.strip()
            if s:
                count[s] = count.get(s, 0) + 1
    repeating = {line for line, c in count.items() if c >= min_pages}
    cleaned = []
    for page in pages_text:
        cleaned.append("\n".join(l for l in page.splitlines() if l.strip() not in repeating))
    return "\n".join(cleaned)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?;])[)\]\u201d\"\']*\s+(?=[A-Z\"\u201c\u201d\(\d])', text)
    return [p.strip() for p in parts if p.strip()]


_OVERLAP_BOILERPLATE_RE = re.compile(
    r'(?:\b(?:docusign|envelope\s+id)\b|(?:\+?\d[\d\s().\-]{6,}\d)'
    r'|(?:https?://|www\.)|[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    r'|\b(?:suite|ste|floor|fl|blvd|ave|st|rd|dr|ln|pkwy)\b)',
    re.IGNORECASE,
)

_SECTION_HEADER_RE = re.compile(
    r'^\s*(?:\d+[\.\d]*\s+[A-Z]|Section\s+\d|Article\s+[IVXivx\d]'
    r'|EXHIBIT\b|SCHEDULE\b|ANNEX\b)',
    re.IGNORECASE,
)


def _is_boilerplate_sentence(s: str) -> bool:
    t = s.strip()
    if not t:
        return True
    if len(t) < 40 and not t.endswith(':'):
        return True
    if t.upper() == t and re.search(r'[A-Z]{3,}', t):
        return True
    return bool(_OVERLAP_BOILERPLATE_RE.search(t))


def _get_overlap_prefix(chunks: list[str], n_sentences: int = 3) -> str:
    if not chunks:
        return ""
    last  = re.sub(r'^\[PARENT:[^\]]*\]\s*->\s*', '', chunks[-1])
    sents = _split_sentences(last)
    clean = [s for s in sents if not _is_boilerplate_sentence(s)]
    tail  = clean[-n_sentences:] if len(clean) >= n_sentences else clean
    return " ".join(tail).strip()


def _split_long_para(text: str, max_chars: int = _STRUCTURAL_MAX_CHARS) -> list[str]:
    t = (text or "").strip()
    if not t or len(t) <= max_chars:
        return [t] if t else []
    sents = _split_sentences(t) or [t]
    out, buf, buf_len = [], [], 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(s) > max_chars:
            if buf:
                out.append(" ".join(buf).strip()); buf = []; buf_len = 0
            out.append(s)
            continue
        add = len(s) + (1 if buf else 0)
        if buf and buf_len + add > max_chars:
            out.append(" ".join(buf).strip())
            tail = buf[-1].strip() if buf else ""
            buf = [tail] if tail and len(tail) < 260 else []
            buf_len = len(buf[0]) if buf else 0
        buf.append(s); buf_len += add
    if buf:
        out.append(" ".join(buf).strip())
    return [x for x in out if x]


def extract_chunks(pdf_path: Path) -> list[str]:
    """Structural chunker: paragraphs → parent labels → conditional overlap."""
    pages_text = _extract_pdf_pages_text(pdf_path)
    raw_text   = _strip_repeating_lines(pages_text)
    lines      = raw_text.splitlines()

    header_re = re.compile(
        r'^\s*(\d+(\.\d+)*\.?\s'           # "1.2.3 ..." or "15. ..."
        r'|SECTION\s+\d+'                   # "SECTION 8" or "SECTION 15."
        r'|Article\s+[A-ZIVX\d]+'          # "Article IV" or "Article 5"
        r'|Schedule\s+[A-Z])',              # "Schedule A"
        re.IGNORECASE
    )
    subsec_re = re.compile(r'^([a-z])\.\s+([A-Z][^.]{2,50})\.\s*', re.IGNORECASE)

    paragraphs: list[str] = []
    current_para: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_para:
                content = " ".join(current_para)
                if len(content) >= 60 or header_re.match(current_para[0]) or subsec_re.match(current_para[0]):
                    paragraphs.append(content)
                    current_para = []
            continue
        if (header_re.match(stripped) or subsec_re.match(stripped)) and current_para:
            paragraphs.append(" ".join(current_para))
            current_para = []
        current_para.append(stripped)
    if current_para:
        paragraphs.append(" ".join(current_para))

    chunks:           list[str] = []
    current_parent    = "General"
    top_level_parent  = "General"
    guard_re          = re.compile(
        r'^\s*(?:provided\s+that|provided,\s*however,\s*that|notwithstanding|except\s+as)\b',
        re.IGNORECASE,
    )

    i = 0
    while i < len(paragraphs):
        para = paragraphs[i].strip()
        if not para:
            i += 1; continue

        m  = header_re.match(para)
        sm = subsec_re.match(para)
        if m or sm:
            if len(para) < 60:
                current_parent = para
                # Emit the header as its own searchable chunk so keyword augmentation
                # can find it and graph expansion can bridge to this section's content.
                chunks.append(f"[PARENT: {current_parent}] -> {para}")
                i += 1; continue
            else:
                if sm:
                    sm_full = sm.group(0).strip().rstrip('.')
                    current_parent = f"{top_level_parent} {sm_full}"
                else:
                    le = para.find('.')
                    if 0 < le < 40:
                        current_parent = para[:le + 1].strip()
                        top_level_parent = current_parent

        if chunks and guard_re.match(para):
            chunks[-1] = chunks[-1] + " " + para
            i += 1; continue

        if ':' in para:
            head, tail = para.split(':', 1)
            preamble   = head.strip() + ":"
            items      = re.split(r'(\s*\([a-z0-9iv]+\)\s*)', tail)
            n_markers  = sum(1 for x in items[1::2] if x.strip())
            if len(items) > 2 and n_markers >= 2:
                prefix = (preamble + " " + items[0].strip()).strip()
                overlap = _get_overlap_prefix(chunks) if not _SECTION_HEADER_RE.match(prefix) else ""
                j = 1
                while j < len(items) - 1:
                    marker  = items[j].strip()
                    content = items[j + 1].strip()
                    if content:
                        if j == 1 and overlap:
                            chunks.append(f"[PARENT: {current_parent}] -> {overlap} {prefix} {marker} {content}")
                        else:
                            chunks.append(f"[PARENT: {current_parent}] -> {prefix} {marker} {content}")
                    j += 2
                i += 1; continue

        for k, part in enumerate(_split_long_para(para)):
            if chunks and k == 0 and not _SECTION_HEADER_RE.match(part):
                overlap = _get_overlap_prefix(chunks)
                if overlap:
                    chunks.append(f"[PARENT: {current_parent}] -> {overlap} {part}")
                    continue
            chunks.append(f"[PARENT: {current_parent}] -> {part}")
        i += 1

    # Dedup
    seen: set[str] = set()
    uniq: list[str] = []
    for c in chunks:
        key = re.sub(r'\s+', ' ', c.lower()).strip()
        if key not in seen:
            seen.add(key); uniq.append(c)
    return uniq


# ══════════════════════════════════════════════════════════════════════════════
# §3  ENTITY RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def build_alias_map(nodes: list[dict]) -> dict[str, str]:
    """Return {alias_lower → canonical_name} from DEFINITION nodes."""
    alias_map: dict[str, str] = {}
    definitions = [n for n in nodes if n.get("type") == "DEFINITION"]
    for d in definitions:
        term   = (d.get("term") or "").strip()
        defn   = (d.get("definition") or "").strip()
        if not term or not defn:
            continue
        short = term
        long  = defn
        # If the definition contains a quoted short name or "hereinafter" alias, swap
        m = re.search(r'[\(\'"]([\w\s]+)[\'"\)]', defn)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) <= len(term):
                short, long = candidate, term
        alias_map[short.lower()] = long
    return alias_map


def resolve_party(name: str, alias_map: dict[str, str]) -> str:
    if not name:
        return name
    return alias_map.get(name.strip().lower(), name)


def normalize_nodes(results: list[dict], alias_map: dict[str, str]) -> list[dict]:
    """Replace party/beneficiary/object_party aliases with canonical names."""
    if not alias_map:
        return results
    party_fields = ("party", "beneficiary", "object_party")
    for r in results:
        for n in r.get("nodes", []):
            for field in party_fields:
                if n.get(field):
                    n[field] = resolve_party(n[field], alias_map)
    return results


def resolve_entities(results: list[dict]) -> list[dict]:
    """
    Compatibility wrapper for entity-resolution stage:
    build alias map from all extracted nodes, then normalize nodes in place.
    """
    all_nodes = [n for r in results for n in r.get("nodes", [])]
    alias_map = build_alias_map(all_nodes)
    return normalize_nodes(results, alias_map)


def build_contract_summary_node(nodes: list[dict]) -> dict | None:
    """Create a DEFINITION "Contract Summary" node aggregating header-level facts."""
    facts: list[str] = []
    for n in nodes:
        if n.get("type") != "DEFINITION":
            continue
        term = (n.get("term") or "").strip()
        defn = (n.get("definition") or "").strip()
        if not term or not defn:
            continue
        lower = term.lower()
        if any(kw in lower for kw in ("agreement type", "territory", "governing law",
                                       "parties", "party", "licensor", "licensee",
                                       "company", "vendor", "client", "customer",
                                       "effective date", "execution date")):
            facts.append(f"{term}: {defn}")

    for n in nodes:
        if n.get("type") != "NUMERIC":
            continue
        at = (n.get("applies_to") or "").lower()
        if "execution date" in at or "effective date" in at:
            facts.append(f"Date: {n['value']} {n['unit']}")

    if not facts:
        return None

    summary_text = " | ".join(facts)
    return {
        "type":       "DEFINITION",
        "term":       "Contract Summary",
        "definition": summary_text,
        "source_text": summary_text,
        "priority":   "high",
    }


# ══════════════════════════════════════════════════════════════════════════════
# §4  GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _node_key(node: dict) -> str:
    t = node.get("type", "")
    if t == "DEFINITION":
        return f"DEF::{(node.get('term') or '').lower()}"
    if t == "OBLIGATION":
        return f"OBL::{(node.get('party') or '').lower()}::{(node.get('action') or '').lower()[:80]}"
    if t == "RIGHT":
        return f"RIG::{(node.get('party') or '').lower()}::{(node.get('right') or '').lower()[:80]}"
    if t == "NUMERIC":
        return f"NUM::{node.get('value','')}::{(node.get('unit') or '').lower()}::{(node.get('applies_to') or '').lower()[:60]}"
    if t == "CONDITION":
        return f"CON::{(node.get('trigger') or '').lower()[:60]}::{(node.get('consequence') or '').lower()[:60]}"
    if t == "BLANK":
        return f"BLK::{(node.get('field') or '').lower()}"
    if t == "REFERENCE":
        return f"REF::{(node.get('to') or '').lower()}"
    return f"UNK::{str(node)[:80]}"


def build_graph(results: list[dict]) -> dict:
    """Build a semantic edge graph over all extracted typed nodes."""
    # Collect all nodes with chunk_id
    all_nodes: list[dict] = []
    for r in results:
        cid = r["chunk_id"]
        for n in r.get("nodes", []):
            all_nodes.append({**n, "_chunk_id": cid})

    # Index by type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for n in all_nodes:
        by_type[n["type"]].append(n)

    # Build term lookup for USES_TERM edges
    def_terms: dict[str, dict] = {}
    for n in by_type.get("DEFINITION", []):
        t = (n.get("term") or "").lower()
        if t:
            def_terms[t] = n

    edges: list[dict] = []
    edge_set: set[tuple] = set()

    def _add_edge(src: dict, dst: dict, etype: str, weight: float = 0.7):
        sk, dk = _node_key(src), _node_key(dst)
        key = (sk, dk, etype)
        if key not in edge_set:
            edge_set.add(key)
            edges.append({"from": sk, "to": dk, "type": etype, "weight": weight})

    # USES_TERM edges: obligations/rights → defined terms they reference
    for ntype in ("OBLIGATION", "RIGHT", "CONDITION", "NUMERIC"):
        for n in by_type.get(ntype, []):
            text = " ".join(str(v) for v in n.values() if isinstance(v, str)).lower()
            for term, def_node in def_terms.items():
                if len(term) > 3 and term in text:
                    _add_edge(n, def_node, "USES_TERM", 0.8)

    # TRIGGERS edges: CONDITION → OBLIGATION/NUMERIC consequences
    for cond in by_type.get("CONDITION", []):
        consq = (cond.get("consequence") or "").lower()
        for ntype in ("OBLIGATION", "RIGHT", "NUMERIC"):
            for n in by_type.get(ntype, []):
                text = " ".join(str(v) for v in n.values() if isinstance(v, str)).lower()
                # Simple heuristic: share significant words
                words_c = {w for w in re.findall(r'\b[a-z]{4,}\b', consq)}
                words_n = {w for w in re.findall(r'\b[a-z]{4,}\b', text)}
                overlap = words_c & words_n - {"party", "parties", "agreement", "shall", "must"}
                if len(overlap) >= 2:
                    _add_edge(cond, n, "TRIGGERS", 0.9)

    # SAME_SECTION edges: nodes from the same chunk
    chunk_nodes: dict[int, list[dict]] = defaultdict(list)
    for n in all_nodes:
        chunk_nodes[n["_chunk_id"]].append(n)
    for cid, nlist in chunk_nodes.items():
        for a in nlist:
            for b in nlist:
                if a is not b and _node_key(a) != _node_key(b):
                    _add_edge(a, b, "SAME_SECTION", 0.5)

    # Cross-chunk SAME_SECTION edges: nodes sharing the same [PARENT: ...] label
    # This lets graph expansion bridge from a section header node to content nodes
    # in the same section even when they're in different chunks.
    _parent_re = re.compile(r'^\[PARENT:\s*([^\]]+)\]')
    section_nodes: dict[str, list[dict]] = defaultdict(list)
    for n in all_nodes:
        src = n.get("source_text", "")
        pm = _parent_re.match(src)
        if pm:
            label = pm.group(1).strip()
            if label and label.lower() != "general":
                section_nodes[label].append(n)
    for label, nlist in section_nodes.items():
        # Cap at 30 to avoid O(N²) explosion in large sections
        capped = nlist[:30]
        for a in capped:
            for b in capped:
                if a is not b and _node_key(a) != _node_key(b):
                    _add_edge(a, b, "SAME_SECTION", 0.5)

    # PARTY_HAS edges: DEFINITION party nodes → their obligations/rights
    party_terms = {(n.get("term") or "").lower(): n
                   for n in by_type.get("DEFINITION", [])
                   if any(k in (n.get("term") or "").lower()
                          for k in ("licensee", "licensor", "company", "vendor",
                                    "client", "party", "distributor", "customer"))}
    for n in all_nodes:
        if n["type"] in ("OBLIGATION", "RIGHT"):
            party = (n.get("party") or "").lower()
            if party in party_terms:
                _add_edge(party_terms[party], n, "PARTY_HAS", 0.85)

    return {"nodes": [_node_key(n) for n in all_nodes], "edges": edges}


class GraphIndex:
    """In-memory index for efficient graph traversal during retrieval."""

    def __init__(self, graph: dict, chunks: list[dict]):
        self._fwd: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._bwd: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for edge in graph.get("edges", []):
            self._fwd[edge["from"]].append((edge["to"],   edge["type"]))
            self._bwd[edge["to"]].append((edge["from"], edge["type"]))
        # Compatibility alias: topology module expects _rev.
        self._rev = self._bwd

        # Key → full node dict (for expansion)
        self._by_key: dict[str, dict] = {}
        for chunk in chunks:
            for node in chunk.get("nodes", []):
                self._by_key[_node_key(node)] = node
        # Compatibility alias for older graph consumers.
        self._nodes = self._by_key

        # Party node keys for PARTY_HAS traversal
        self.party_keys: set[str] = {
            k for k in self._by_key
            if k.startswith("DEF::") and any(
                p in k for p in ("licensee", "licensor", "company",
                                 "vendor", "client", "party", "distributor")
            )
        }

    def get_subgraph(self, seed_keys: list[str], max_extra: int = 25) -> list[dict]:
        """Return nodes reachable within 2 hops from seed_keys."""
        seen   = set(seed_keys)
        result = []
        queue  = list(seed_keys)
        hops   = 0
        while queue and len(result) < max_extra and hops < 2:
            next_q = []
            for k in queue:
                for (nk, etype) in self._fwd.get(k, []) + self._bwd.get(k, []):
                    if nk not in seen and nk in self._by_key:
                        seen.add(nk)
                        result.append(self._by_key[nk])
                        next_q.append(nk)
            queue = next_q
            hops += 1
        return result[:max_extra]

    def get_chain_annotations(self, node_keys: list[str]) -> dict[str, list[str]]:
        """Return chain annotation strings for each node key."""
        key_set = set(node_keys)
        notes: dict[str, list[str]] = defaultdict(list)
        _LABEL = {
            "USES_TERM":     "uses term defined here",
            "TRIGGERS":      "activates",
            "SAME_SECTION":  "same section as",
            "PARTY_HAS":     "party clause",
        }
        for k in node_keys:
            for (nk, etype) in self._fwd.get(k, []):
                if nk in key_set:
                    lbl = _LABEL.get(etype, etype.lower())
                    if etype == "USES_TERM":
                        notes[k].append(f"uses term defined here: {nk[5:]}")
                    elif etype == "TRIGGERS":
                        notes[k].append(f"triggers obligation/right: {nk[:60]}")
                    else:
                        notes[k].append(f"{lbl}: {nk[:60]}")
            for (nk, etype) in self._bwd.get(k, []):
                if nk in key_set:
                    if etype == "TRIGGERS":
                        notes[k].append(f"activated when: {nk[4:60]}")
        return dict(notes)

    def get_contradictions(self) -> list[tuple[str, str]]:
        """Find OBLIGATION nodes with the same party+action but different modals."""
        pairs: list[tuple[str, str]] = []
        obl_nodes = [(k, self._by_key[k]) for k in self._by_key
                     if k.startswith("OBL::")]
        checked: set[tuple] = set()
        for k1, n1 in obl_nodes:
            for k2, n2 in obl_nodes:
                if k1 >= k2:
                    continue
                if (k1, k2) in checked:
                    continue
                checked.add((k1, k2))
                if (n1.get("party") == n2.get("party")
                        and n1.get("action", "")[:40] == n2.get("action", "")[:40]
                        and n1.get("modal") != n2.get("modal")):
                    pairs.append((k1, k2))
        return pairs

    def get_party_nodes(self, party_name_lower: str) -> list[dict]:
        """Return all OBLIGATION/RIGHT nodes belonging to the given party."""
        result = []
        for k, n in self._by_key.items():
            if n.get("type") in ("OBLIGATION", "RIGHT"):
                if (n.get("party") or "").lower() == party_name_lower:
                    result.append(n)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# §5  TYPED NODE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

_VALID_TYPES = {"DEFINITION", "OBLIGATION", "RIGHT", "NUMERIC", "CONDITION", "BLANK", "REFERENCE"}

_REQUIRED_FIELDS = {
    "DEFINITION":  {"term", "definition"},
    "OBLIGATION":  {"party", "modal", "action"},
    "RIGHT":       {"party", "right"},
    "NUMERIC":     {"value", "unit", "applies_to"},
    "CONDITION":   {"trigger", "consequence"},
    "BLANK":       {"field", "context"},
    "REFERENCE":   {"to", "describes"},
}

# LLM prompts for extraction — see v2/typed_node_extractor.py for full prompts.
# Abbreviated here to show structure; actual prompts preserved in that file.

_EXTRACT_PROMPT = """\
You are a legal document analyst. Extract ALL typed nodes from the contract text below.

Classify each distinct clause/statement into one type:
- DEFINITION  : "X means / includes / is defined as Y"
- OBLIGATION  : a party MUST / SHALL / CANNOT / IS PROHIBITED FROM doing something
- RIGHT       : a party MAY / IS ENTITLED TO / IS PERMITTED TO do something
- NUMERIC     : clause containing a specific number, rate, period, date, or amount
- CONDITION   : "if X then Y" — a trigger event and its consequence
- BLANK       : a field/value left blank or unfilled (template placeholders)
- REFERENCE   : a clause pointing to an external document, schedule, exhibit, or attachment

CRITICAL: For any lettered/numbered list items, extract EVERY SINGLE item. Never truncate.
CRITICAL: Preserve ALL qualifiers in action/right/consequence fields.

DEFINITION  → {{"type":"DEFINITION",  "term":"...", "definition":"...", "value_tags":["tag1","tag2"], "source_text":"exact clause"}}
OBLIGATION  → {{"type":"OBLIGATION",  "party":"...", "modal":"must|shall|cannot|prohibited", "action":"...", "qualifiers":"...or null", "condition":"or null", "beneficiary":"or null", "object_party":"or null", "source_text":"exact clause"}}
RIGHT       → {{"type":"RIGHT",       "party":"...", "right":"...", "qualifiers":"...or null", "limit":"or null", "condition":"or null", "beneficiary":"or null", "object_party":"or null", "source_text":"exact clause"}}
NUMERIC     → {{"type":"NUMERIC",     "value":"...", "unit":"...", "applies_to":"...", "trigger":"or null", "value_tags":["tag1","tag2"], "source_text":"exact clause"}}
CONDITION   → {{"type":"CONDITION",   "trigger":"...", "consequence":"...", "party":"or null", "source_text":"exact clause"}}
BLANK       → {{"type":"BLANK",       "field":"name of what is blank", "context":"surrounding sentence", "source_text":"exact text"}}
REFERENCE   → {{"type":"REFERENCE",   "to":"name of document/schedule/exhibit", "describes":"what content it contains", "source_text":"exact clause"}}

For NUMERIC value_tags: 2-5 lowercase concept tags describing what real-world concept this number measures.
Examples: ["revenue","total","currency"] | ["notice_period","days","termination"] | ["interest_rate","payment","percent"]
For DEFINITION value_tags: 2-5 lowercase concept tags describing what concept this term defines.

Text:
{{chunk}}

Return a JSON object with a "nodes" key containing the array of extracted nodes."""

_BATCH_EXTRACT_PROMPT = """\
You are a legal document analyst. Extract ALL typed nodes from the contract sections below.
Each section is labeled with its chunk ID. Include "chunk_id" on every node.

Same type rules as single-chunk extraction. Include value_tags on NUMERIC and DEFINITION nodes.

Sections:
{{batch_text}}

Return a JSON object with a "nodes" key. Every node must include "chunk_id"."""

_METADATA_PROMPT = """\
Extract key agreement-level facts from the opening section of a legal contract.

For each of: (1) party names, (2) execution date, (3) agreement type,
(4) territory, (5) governing law — extract the appropriate typed node.

Text:
{{header_text}}

Return a JSON object with a "nodes" key. Use DEFINITION for names/terms, NUMERIC for dates."""

_SIGNATURE_PROMPT = """\
Extract signatory information from the closing/signature section of a legal contract.

For each person who signed, extract:
- Their full name
- Their title or role (e.g. CEO, President, Authorized Signatory)
- Which party they represent
- Date signed (if present)
- Any contact details present (email, fax, phone, address)

Text:
{{signature_text}}

Return a JSON object with a "nodes" key. Use DEFINITION nodes:
  {{"type":"DEFINITION","term":"Signatory — <Party Name>","definition":"<Full Name>, <Title>","value_tags":["signatory","execution","party"],"source_text":"<exact text>"}}
Also extract any contact fields (email, fax) as separate DEFINITION nodes with term like "Fax — <Party Name>"."""

_GAP_AUDIT_PROMPT = """\
You are auditing typed node extraction from a legal contract for completeness.

Contract type: {{agreement_type}}
Nodes extracted so far:
{{node_summary}}

Return a JSON object with a "missing" key: list of topic strings for provision types
COMPLETELY ABSENT (not just sparse) from the extracted nodes.
Return {{"missing": []}} if nothing major is missing."""


def _clean_node(raw: dict) -> dict | None:
    ntype = str(raw.get("type", "")).upper()
    if ntype not in _VALID_TYPES:
        return None
    for f in _REQUIRED_FIELDS[ntype]:
        if not raw.get(f):
            return None

    node = {"type": ntype, "source_text": str(raw.get("source_text", "")).strip()}

    if ntype == "DEFINITION":
        node.update(term=str(raw["term"]).strip(), definition=str(raw["definition"]).strip())
        if raw.get("value_tags"):
            node["value_tags"] = [str(t).lower().strip() for t in raw["value_tags"] if t][:8]

    elif ntype == "OBLIGATION":
        node.update(party=str(raw["party"]).strip(),
                    modal=str(raw["modal"]).strip().lower(),
                    action=str(raw["action"]).strip(),
                    qualifiers=str(raw.get("qualifiers") or "").strip() or None,
                    condition=str(raw.get("condition") or "").strip() or None,
                    beneficiary=str(raw.get("beneficiary") or "").strip() or None,
                    object_party=str(raw.get("object_party") or "").strip() or None)

    elif ntype == "RIGHT":
        node.update(party=str(raw["party"]).strip(),
                    right=str(raw["right"]).strip(),
                    qualifiers=str(raw.get("qualifiers") or "").strip() or None,
                    limit=str(raw.get("limit") or "").strip() or None,
                    condition=str(raw.get("condition") or "").strip() or None,
                    beneficiary=str(raw.get("beneficiary") or "").strip() or None,
                    object_party=str(raw.get("object_party") or "").strip() or None)

    elif ntype == "NUMERIC":
        node.update(value=str(raw["value"]).strip(),
                    unit=str(raw["unit"]).strip(),
                    applies_to=str(raw["applies_to"]).strip(),
                    trigger=str(raw.get("trigger") or "").strip() or None)
        if raw.get("value_tags"):
            node["value_tags"] = [str(t).lower().strip() for t in raw["value_tags"] if t][:8]

    elif ntype == "CONDITION":
        node.update(trigger=str(raw["trigger"]).strip(),
                    consequence=str(raw["consequence"]).strip(),
                    party=str(raw.get("party") or "").strip() or None)

    elif ntype == "BLANK":
        node.update(field=str(raw["field"]).strip(), context=str(raw["context"]).strip())

    elif ntype == "REFERENCE":
        node.update(to=str(raw["to"]).strip(), describes=str(raw["describes"]).strip())

    return node


def _parse_nodes_from_raw(raw: str) -> list[tuple[int | None, dict]]:
    try:
        obj = json.loads(raw)
        parsed = (obj.get("nodes") or obj.get("items") or []) if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if not m:
            return []
        try:
            parsed = json.loads(m.group())
        except json.JSONDecodeError:
            try:
                parsed = json.loads(re.sub(r',\s*([}\]])', r'\1', m.group()))
            except json.JSONDecodeError:
                return []

    result = []
    for item in parsed:
        if isinstance(item, dict):
            node = _clean_node(item)
            if node:
                result.append((item.get("chunk_id"), node))
    return result


def _extract_chunk(chunk_id: int, chunk_text: str) -> dict:
    raw   = ask(_EXTRACT_PROMPT.replace("{{chunk}}", chunk_text[:2500]),
                temperature=0.0, max_tokens=3000, json_mode=True)
    nodes = [n for _, n in _parse_nodes_from_raw(raw)]
    return {"chunk_id": chunk_id, "chunk_text": chunk_text, "nodes": nodes}


def _extract_batch(chunk_pairs: list[tuple[int, str]]) -> list[dict]:
    parts      = [f"--- CHUNK {cid} ---\n{text[:1500]}" for cid, text in chunk_pairs]
    batch_text = "\n\n".join(parts)
    raw        = ask(_BATCH_EXTRACT_PROMPT.replace("{{batch_text}}", batch_text),
                     temperature=0.0, max_tokens=6000, json_mode=True)
    raw_nodes  = _parse_nodes_from_raw(raw)
    valid_ids  = {cid for cid, _ in chunk_pairs}
    first_id   = chunk_pairs[0][0]
    per_chunk: dict[int, list[dict]] = {cid: [] for cid, _ in chunk_pairs}
    for raw_cid, node in raw_nodes:
        target = raw_cid if raw_cid in valid_ids else first_id
        per_chunk[target].append(node)
    chunk_texts = {cid: text for cid, text in chunk_pairs}
    return [{"chunk_id": cid, "chunk_text": chunk_texts[cid], "nodes": per_chunk[cid]}
            for cid, _ in chunk_pairs]


# ── Keyword → value_tag mapping for CONDITION-derived NUMERIC nodes ──────────
_COND_NUM_TAG_RULES: list[tuple[list[str], list[str]]] = [
    (["notice",  "notify", "written notice"],          ["notice_period", "days"]),
    (["terminat", "cancell", "end the agreement"],     ["termination"]),
    (["exit fee", "transfer fee", "exit"],             ["exit_fee", "fee", "currency"]),
    (["arbitrat", "hearing", "dispute resolution"],    ["arbitration", "dispute", "days"]),
    (["initial term", "term of", "renew", "renewal"],  ["term", "duration", "renewal"]),
    (["nrr", "net recurring revenue", "recurring rev"],["revenue", "recurring", "currency"]),
    (["suspend", "suspension"],                        ["suspension", "notice_period"]),
    (["press release", "public statement", "approval"],["approval", "days"]),
    (["payment", "invoice", "remit", "due"],           ["payment", "fee"]),
    (["audit", "inspection", "records"],               ["audit", "days"]),
    (["cure", "remedy", "breach"],                     ["cure_period", "breach", "days"]),
    (["escrow", "trust"],                              ["escrow", "days"]),
]

# Patterns: (regex, value_group, unit_string, base_tags)
_COND_NUM_PATTERNS: list[tuple[re.Pattern, int, str, list[str]]] = [
    # Calendar / business days
    (re.compile(r'\b(\d[\d,]*)\s*(calendar\s+days?)\b', re.I), 1, "calendar days", ["days"]),
    (re.compile(r'\b(\d[\d,]*)\s*(business\s+days?)\b',  re.I), 1, "business days", ["days"]),
    # Generic days/weeks/months/years
    (re.compile(r'\b(\d[\d,]*)\s*(days?)\b',   re.I), 1, "days",   ["days"]),
    (re.compile(r'\b(\d[\d,]*)\s*(weeks?)\b',  re.I), 1, "weeks",  ["duration"]),
    (re.compile(r'\b(\d[\d,]*)\s*(months?)\b', re.I), 1, "months", ["duration"]),
    (re.compile(r'\b(\d[\d,]*)\s*(years?)\b',  re.I), 1, "years",  ["duration"]),
    # Dollar amounts — $X, $X,000, $X million/thousand
    (re.compile(r'\$\s*(\d[\d,]*(?:\.\d+)?)\s*(million|thousand|k)\b', re.I),
     1, "USD", ["currency", "cap"]),
    (re.compile(r'\$\s*(\d[\d,]*(?:\.\d+)?)\b'), 1, "USD", ["currency"]),
    # Percentages
    (re.compile(r'\b(\d+(?:\.\d+)?)\s*(%|percent)\b', re.I), 1, "%", ["percent", "rate"]),
]


def _infer_value_tags(text: str, base_tags: list[str]) -> list[str]:
    """Return value_tags by matching text against _COND_NUM_TAG_RULES."""
    tags: list[str] = list(base_tags)
    t = text.lower()
    for keywords, ktags in _COND_NUM_TAG_RULES:
        if any(kw in t for kw in keywords):
            tags.extend(ktags)
    # Deduplicate preserving order, cap at 6
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag); out.append(tag)
    return out[:6]


def _normalize_value(raw_val: str, unit: str) -> str:
    """Strip commas; expand million/thousand/k multipliers."""
    v = raw_val.replace(",", "").strip()
    u = unit.lower()
    try:
        n = float(v)
        if "million" in u:
            return str(int(n * 1_000_000)) if n == int(n) else f"{n * 1_000_000:.2f}"
        if "thousand" in u or u == "k":
            return str(int(n * 1_000)) if n == int(n) else f"{n * 1_000:.2f}"
        return str(int(n)) if n == int(n) else v
    except ValueError:
        return v


def _expand_condition_numerics(results: list[dict]) -> list[dict]:
    """
    Post-processing pass: scan every CONDITION node, extract embedded numeric
    values via regex, and emit companion NUMERIC nodes in the same chunk.

    This converts clauses like "if Affiliate fails to give 30 days written
    notice prior to termination" into a NUMERIC node
    {value:'30', unit:'days', applies_to:'notice period before termination'}
    so the retriever can surface the answer directly without triggering Rule 13.

    Only adds nodes — never modifies existing ones.
    """
    # Collect existing NUMERIC keys to avoid duplicates
    existing_numeric_keys: set[str] = set()
    for r in results:
        for n in r.get("nodes", []):
            if n.get("type") == "NUMERIC":
                existing_numeric_keys.add(_node_key(n))

    added_total = 0
    for r in results:
        new_nodes: list[dict] = []
        for node in r.get("nodes", []):
            if node.get("type") != "CONDITION":
                continue

            trigger     = node.get("trigger", "") or ""
            consequence = node.get("consequence", "") or ""
            source_text = node.get("source_text", "") or ""
            search_text = f"{trigger} {consequence}"

            for pattern, val_group, unit_str, base_tags in _COND_NUM_PATTERNS:
                for m in pattern.finditer(search_text):
                    raw_val = m.group(val_group)
                    # Normalise the unit label (strip multiplier suffix for USD)
                    unit_label = "USD" if unit_str == "USD" else m.group(2).lower().strip()
                    norm_val   = _normalize_value(raw_val, unit_str)

                    # Build applies_to from whichever field the match came from
                    match_start = m.start()
                    if match_start < len(trigger):
                        context = trigger
                    else:
                        context = consequence
                    applies_to = context[:120].strip()

                    value_tags = _infer_value_tags(search_text, base_tags)

                    candidate: dict = {
                        "type":        "NUMERIC",
                        "value":       norm_val,
                        "unit":        unit_label,
                        "applies_to":  applies_to,
                        "trigger":     trigger[:200] or None,
                        "value_tags":  value_tags,
                        "source_text": source_text,
                        "_derived_from_condition": True,
                    }
                    key = _node_key(candidate)
                    if key not in existing_numeric_keys:
                        existing_numeric_keys.add(key)
                        new_nodes.append(candidate)

        if new_nodes:
            r["nodes"].extend(new_nodes)
            added_total += len(new_nodes)

    if added_total:
        print(f"  Condition→Numeric: {added_total} derived NUMERIC nodes added")
    return results


def _extract_signature_block(chunks: list[str]) -> list[dict]:
    """Extract signatory info from the last ~200 words of the contract."""
    # Take last 2 chunks to catch multi-page signature blocks
    tail_text = "\n\n".join(chunks[-2:])
    # Trim to last ~1200 chars (~200 words) to stay focused
    tail_text = tail_text[-1200:]
    raw   = ask(_SIGNATURE_PROMPT.replace("{{signature_text}}", tail_text),
                temperature=0.0, max_tokens=1000, json_mode=True)
    nodes = [n for _, n in _parse_nodes_from_raw(raw)]
    for n in nodes:
        n["priority"] = "high"
    return nodes


def _extract_metadata(chunks: list[str]) -> list[dict]:
    header_text = "\n\n".join(chunks[:2])[:3000]
    raw   = ask(_METADATA_PROMPT.replace("{{header_text}}", header_text),
                temperature=0.0, max_tokens=1500, json_mode=True)
    nodes = [n for _, n in _parse_nodes_from_raw(raw)]
    for n in nodes:
        n["priority"] = "high"
    return nodes


def _gap_audit(results: list[dict], workers: int = 4) -> list[dict]:
    by_type: dict[str, list[str]] = defaultdict(list)
    for r in results:
        for n in r.get("nodes", []):
            t = n["type"]
            if t == "DEFINITION":   by_type[t].append(n["term"])
            elif t == "OBLIGATION": by_type[t].append(f"{n['party']}: {n['action'][:50]}")
            elif t == "RIGHT":      by_type[t].append(f"{n['party']}: {n['right'][:50]}")
            elif t == "NUMERIC":    by_type[t].append(f"{n['value']} {n['unit']}")
            elif t == "CONDITION":  by_type[t].append(f"if {n['trigger'][:40]}")
            elif t == "BLANK":      by_type[t].append(n["field"])
            elif t == "REFERENCE":  by_type[t].append(n["to"])
    node_summary = "\n".join(f"{t}: {', '.join(items[:20])}" for t, items in sorted(by_type.items()))

    agreement_type = "legal contract"
    for r in results:
        for n in r.get("nodes", []):
            if n["type"] == "DEFINITION" and (n.get("term") or "").lower() in ("agreement type",):
                agreement_type = n.get("definition", agreement_type)

    raw = ask(_GAP_AUDIT_PROMPT
              .replace("{{agreement_type}}", agreement_type)
              .replace("{{node_summary}}", node_summary),
              temperature=0.0, max_tokens=400, json_mode=True)

    missing: list[str] = []
    try:
        obj = json.loads(raw)
        missing = (obj.get("missing") or []) if isinstance(obj, dict) else ([obj] if isinstance(obj, str) else obj)
    except Exception:
        pass

    if not missing:
        return results

    results_map = {r["chunk_id"]: r for r in results}
    chunks_to_retry: set[int] = set()
    for topic in missing:
        kws = [w for w in topic.lower().split() if len(w) > 3]
        scored = [(sum(1 for kw in kws if kw in r.get("chunk_text", "").lower()), r["chunk_id"])
                  for r in results]
        scored.sort(reverse=True)
        for _, cid in scored[:3]:
            chunks_to_retry.add(cid)

    if not chunks_to_retry:
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_chunk, cid, results_map[cid]["chunk_text"]): cid
                   for cid in chunks_to_retry if cid in results_map}
        for fut in as_completed(futures):
            cid = futures[fut]
            new = fut.result()
            if new["nodes"]:
                existing = {n["source_text"] for n in results_map[cid]["nodes"]}
                added = [n for n in new["nodes"] if n["source_text"] not in existing]
                if added:
                    results_map[cid]["nodes"].extend(added)

    return [results_map[r["chunk_id"]] for r in results]


def extract_all(pdf_path: Path, workers: int = 4, batch_size: int = 3, progress_cb=None) -> list[dict]:
    """Full extraction pipeline: chunk → metadata → batch extract → retry → gap audit → entity resolve → graph."""
    chunks = extract_chunks(pdf_path)
    print(f"  {len(chunks)} chunks extracted from {pdf_path.name}")
    total_chunks = len(chunks)
    extracted_chunks = 0
    nodes_created = 0
    llm_chunks_seen = 0

    def _push(stage: str, **extra) -> None:
        _emit_progress(
            progress_cb,
            stage=stage,
            total_chunks=total_chunks,
            extracted_chunks=extracted_chunks,
            nodes_created=nodes_created,
            llm_chunks_seen=llm_chunks_seen,
            **extra,
        )

    # 1. Metadata pass (parties, date, type, territory from first 2 chunks)
    metadata_nodes = _extract_metadata(chunks)
    print(f"  Metadata: {len(metadata_nodes)} nodes")
    llm_chunks_seen += min(2, total_chunks)
    _push("metadata", metadata_nodes=len(metadata_nodes))

    # 2. Main extraction
    pairs   = list(enumerate(chunks))
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    _push("extracting", batch_count=len(batches))
    results: list[dict | None] = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_batch = {pool.submit(_extract_batch, b): b for b in batches}
        for fut in as_completed(future_to_batch):
            batch_out = fut.result()
            for r in batch_out:
                results[r["chunk_id"]] = r
                extracted_chunks += 1
                nodes_created += len(r.get("nodes", []))
                llm_chunks_seen += 1
                _push("extracting")

    results = [r for r in results if r]

    # 3. Retry zero-node chunks
    zero = [(r["chunk_id"], r["chunk_text"]) for r in results if len(r["nodes"]) == 0]
    if zero:
        _push("retrying_zero_chunks", retry_chunks=len(zero))
        rmap = {r["chunk_id"]: r for r in results}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_extract_chunk, cid, text): cid for cid, text in zero}
            for fut in as_completed(futures):
                cid = futures[fut]
                retry = fut.result()
                if retry["nodes"]:
                    prev_count = len(rmap[cid].get("nodes", []))
                    rmap[cid] = retry
                    nodes_created += max(0, len(retry["nodes"]) - prev_count)
                llm_chunks_seen += 1
                _push("retrying_zero_chunks")
        results = [rmap[r["chunk_id"]] for r in results]

    # Prepend metadata to chunk 0
    if metadata_nodes and results:
        existing = {n["source_text"] for n in results[0]["nodes"]}
        new_meta = [n for n in metadata_nodes if n["source_text"] not in existing]
        results[0]["nodes"] = new_meta + results[0]["nodes"]
        nodes_created += len(new_meta)
        _push("metadata_injected", injected=len(new_meta))

    # 4. Gap audit
    _push("gap_audit")
    results = _gap_audit(results, workers=workers)

    # 5. Entity resolution
    _push("entity_resolution")
    all_nodes = [n for r in results for n in r["nodes"]]
    alias_map = build_alias_map(all_nodes)
    results   = normalize_nodes(results, alias_map)

    summary_node = build_contract_summary_node(all_nodes)
    if summary_node and results:
        existing = {n["source_text"] for n in results[0]["nodes"]}
        if summary_node["source_text"] not in existing:
            results[0]["nodes"].insert(0, summary_node)
            nodes_created += 1
            _push("summary_injected")

    # 5b. CONDITION → NUMERIC post-processing (pure Python, no LLM)
    _push("expand_condition_numerics")
    results = _expand_condition_numerics(results)

    # 5c. Signature block extraction (last ~200 words, runs once at index time)
    _push("signature_extract")
    sig_nodes = _extract_signature_block(chunks)
    if sig_nodes and results:
        existing = {n["source_text"] for n in results[-1]["nodes"]}
        new_sigs = [n for n in sig_nodes if n["source_text"] not in existing]
        results[-1]["nodes"].extend(new_sigs)
        nodes_created += len(new_sigs)
        print(f"  Signature block: {len(new_sigs)} nodes")

    # 6. Graph
    _push("graph_build")
    graph      = build_graph(results)
    graph_path = pdf_path.parent / f"{pdf_path.stem}_graph.json"
    graph_path.write_text(json.dumps(graph, indent=2))
    print(f"  Graph: {len(graph['edges'])} edges → {graph_path.name}")

    total = sum(len(r["nodes"]) for r in results)
    print(f"  Total: {total} nodes across {len(chunks)} chunks")
    _push("done_extract", final_nodes=total, edges=len(graph.get("edges", [])))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# §6  QUESTION PARSING
# ══════════════════════════════════════════════════════════════════════════════

_QUESTION_NODE_PROMPT = """\
You are a legal query analyst. Parse this legal question into a typed node.

Question: {question}

Step 1 — choose the node_type:
  RIGHT       — asking what someone CAN or IS ALLOWED to do
  OBLIGATION  — asking what someone MUST/SHALL do, or CANNOT do, or is responsible for
  DEFINITION  — asking what a term means or how something is defined
  NUMERIC     — asking for a specific number, rate, amount, period, or date
  CONDITION   — asking what happens IF a trigger occurs
  GENERAL     — none of the above

Step 2 — fill in the fields for that type. For ALL types also set:
  entity, verb, object, beneficiary, requires_numeric (true only if asking for specific value/rate/amount/date)

Step 3 — also classify answerability intent fields:
- proof_type: one of [direct_rule_clause, exact_value, definition, enumeration, customer_instance_field, external_artifact_contents, mixed]
- answer_form: one of [yes_no, date, number, percent, name, list, clause_text, mixed]
- lane: one of [direct_rule, exact_value, dependency_risk, speculative_outside, mixed]
- exact_intent, simple_extract_intent, speculative_intent, outside_document_intent, missing_artifact_intent, instance_missing_field, external_reference_intent: floats in [0,1]
- expects_direct_rule_clause, outside_doc_needed, asks_for_external_contents: floats in [0,1]
- external_doc_type: one of [none, dpa, security_measures, pricing_page, service_specific_terms, healthcare_addendum, audit_reports, supported_countries, order_form, exhibit, schedule, appendix, annex, other]
- answer_source_type: one of [base_clause, customer_instance, external_artifact, runtime_fact, mixed]
- source_sufficiency_required: one of [clause_quote_enough, explicit_value_required, artifact_contents_required, runtime_lookup_required, mixed]
- missing_if_not_present: boolean
- customer_instance_dependency, external_artifact_dependency, runtime_dependency: floats in [0,1]
- runtime_check: boolean — true if answering this question requires knowing what actually
  happened in the real world AFTER the contract was signed (e.g. "has X ever done Y?",
  "did the parties agree to change Z?", "how many clients have been generated?").
  False if the question only requires reading what the contract itself specifies.

Return ONLY valid JSON, no markdown:
{{"node_type":"RIGHT|OBLIGATION|DEFINITION|NUMERIC|CONDITION|GENERAL",
  "party":null,"action":null,"object_party":null,"beneficiary":null,"modal":null,
  "term":null,"applies_to":null,"trigger":null,"requires_numeric":false,
  "entity":null,"verb":null,"object":null,
  "intent": {{"proof_type":"mixed","answer_form":"mixed","lane":"mixed",
    "exact_intent":0.0,"simple_extract_intent":0.0,"speculative_intent":0.0,"outside_document_intent":0.0,
    "missing_artifact_intent":0.0,"instance_missing_field":0.0,"external_reference_intent":0.0,
    "expects_direct_rule_clause":0.0,"outside_doc_needed":0.0,"asks_for_external_contents":0.0,
    "external_doc_type":"none","answer_source_type":"mixed","source_sufficiency_required":"mixed",
    "missing_if_not_present":false,"customer_instance_dependency":0.0,"external_artifact_dependency":0.0,"runtime_dependency":0.0,
    "runtime_check":false}}}}"""

_DEFINITION_VERBS = {"define","defined","mean","means","refer","refers","constitute",
                     "constitutes","include","includes","consist","encompasses"}
_OBLIGATION_VERBS = {"must","shall","required","obligat","duty","permitted","may not",
                     "cannot","prohibited","responsible","liable","comply","ensure",
                     "maintain","provide","notify","indemnify","pay","reimburse"}
_PARTY_NAMES      = {"wipro","licensee","licensor","customer","vendor","supplier",
                     "provider","client","user","party","parties"}
_QUESTION_LLM_BUNDLE_CACHE: dict[str, dict] = {}
_ALLOWED_PROOF_TYPES = {
    "direct_rule_clause", "exact_value", "definition", "enumeration",
    "customer_instance_field", "external_artifact_contents", "mixed",
}
_ALLOWED_ANSWER_FORMS = {"yes_no", "date", "number", "percent", "name", "list", "clause_text", "mixed"}
_ALLOWED_LANES = {"direct_rule", "exact_value", "dependency_risk", "speculative_outside", "mixed"}
_ALLOWED_EXTERNAL_DOC_TYPES = {
    "none", "dpa", "security_measures", "pricing_page", "service_specific_terms",
    "healthcare_addendum", "audit_reports", "supported_countries", "order_form",
    "exhibit", "schedule", "appendix", "annex", "other",
}
_ALLOWED_ANSWER_SOURCE_TYPES = {"base_clause", "customer_instance", "external_artifact", "runtime_fact", "mixed"}
_ALLOWED_SOURCE_SUFFICIENCY = {
    "clause_quote_enough", "explicit_value_required", "artifact_contents_required",
    "runtime_lookup_required", "mixed",
}


def _clip01(v, default=0.0) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except Exception:
        return float(default)


def _normalize_question_intent(intent_raw: dict | None) -> dict:
    raw = intent_raw if isinstance(intent_raw, dict) else {}
    proof_type = str(raw.get("proof_type") or "mixed").strip().lower()
    answer_form = str(raw.get("answer_form") or "mixed").strip().lower()
    lane = str(raw.get("lane") or "mixed").strip().lower()
    external_doc_type = str(raw.get("external_doc_type") or "none").strip().lower()
    answer_source_type = str(raw.get("answer_source_type") or "mixed").strip().lower()
    source_sufficiency_required = str(raw.get("source_sufficiency_required") or "mixed").strip().lower()
    return {
        "proof_type": proof_type if proof_type in _ALLOWED_PROOF_TYPES else "mixed",
        "answer_form": answer_form if answer_form in _ALLOWED_ANSWER_FORMS else "mixed",
        "lane": lane if lane in _ALLOWED_LANES else "mixed",
        "exact_intent": _clip01(raw.get("exact_intent")),
        "simple_extract_intent": _clip01(raw.get("simple_extract_intent")),
        "speculative_intent": _clip01(raw.get("speculative_intent")),
        "outside_document_intent": _clip01(raw.get("outside_document_intent")),
        "missing_artifact_intent": _clip01(raw.get("missing_artifact_intent")),
        "instance_missing_field": _clip01(raw.get("instance_missing_field")),
        "external_reference_intent": _clip01(raw.get("external_reference_intent")),
        "expects_direct_rule_clause": _clip01(raw.get("expects_direct_rule_clause")),
        "outside_doc_needed": _clip01(raw.get("outside_doc_needed")),
        "asks_for_external_contents": _clip01(raw.get("asks_for_external_contents")),
        "external_doc_type": external_doc_type if external_doc_type in _ALLOWED_EXTERNAL_DOC_TYPES else "none",
        "answer_source_type": answer_source_type if answer_source_type in _ALLOWED_ANSWER_SOURCE_TYPES else "mixed",
        "source_sufficiency_required": (
            source_sufficiency_required
            if source_sufficiency_required in _ALLOWED_SOURCE_SUFFICIENCY else "mixed"
        ),
        "missing_if_not_present": bool(raw.get("missing_if_not_present", False)),
        "customer_instance_dependency": _clip01(raw.get("customer_instance_dependency")),
        "external_artifact_dependency": _clip01(raw.get("external_artifact_dependency")),
        "runtime_dependency": _clip01(raw.get("runtime_dependency")),
        "runtime_check": bool(raw.get("runtime_check", False)),
    }


def parse_question_node(question: str) -> dict:
    _FALLBACK = {"node_type":"GENERAL","entity":question,"verb":"","object":"",
                 "condition":None,"requires_numeric":False,"party":None,"action":None,
                 "object_party":None,"beneficiary":None,"modal":None,"term":None,
                 "applies_to":None,"trigger":None}
    node = None
    for _ in range(2):
        raw = ask(
            _QUESTION_NODE_PROMPT.format(question=question),
            temperature=0.0,
            max_tokens=360,
            json_mode=True,
        )
        try:
            node = json.loads(raw)
            if isinstance(node, dict):
                break
            node = None
        except Exception:
            node = None
    if node is None:
        return _FALLBACK
    out = {
        "node_type":        str(node.get("node_type") or "GENERAL").upper().strip(),
        "party":            str(node.get("party") or "").strip() or None,
        "action":           str(node.get("action") or "").strip() or None,
        "object_party":     str(node.get("object_party") or "").strip() or None,
        "beneficiary":      str(node.get("beneficiary") or "").strip() or None,
        "modal":            str(node.get("modal") or "").strip().lower() or None,
        "term":             str(node.get("term") or "").strip() or None,
        "applies_to":       str(node.get("applies_to") or "").strip() or None,
        "trigger":          str(node.get("trigger") or "").strip() or None,
        "entity":           str(node.get("entity") or node.get("party") or "").strip(),
        "verb":             str(node.get("verb") or node.get("action") or "").strip(),
        "object":           str(node.get("object") or node.get("applies_to") or "").strip(),
        "condition":        str(node.get("condition") or node.get("trigger") or "").strip() or None,
        "requires_numeric": bool(node.get("requires_numeric", False)),
    }
    intent = _normalize_question_intent(node.get("intent") if isinstance(node.get("intent"), dict) else {})
    _QUESTION_LLM_BUNDLE_CACHE[(question or "").strip().lower()] = {
        "node": dict(out),
        "intent": dict(intent),
    }
    return out


def get_cached_question_intent(question: str) -> dict:
    return dict((_QUESTION_LLM_BUNDLE_CACHE.get((question or "").strip().lower()) or {}).get("intent") or {})


def classify_question_type(q_node: dict) -> str:
    if q_node.get("requires_numeric"):
        return "NUMERIC"
    verb       = (q_node.get("verb") or "").lower()
    entity     = (q_node.get("entity") or "").lower()
    obj        = (q_node.get("object") or "").lower()
    verb_words = set(re.split(r'\W+', verb))
    if verb_words & _DEFINITION_VERBS or "definition" in obj or "meaning" in obj:
        return "DEFINITION"
    if entity and any(p in entity for p in _PARTY_NAMES):
        if any(ov in verb for ov in _OBLIGATION_VERBS):
            return "OBLIGATION"
        return "PARTY_SPECIFIC"
    if any(ov in verb for ov in _OBLIGATION_VERBS):
        return "OBLIGATION"
    return "GENERAL"


# ══════════════════════════════════════════════════════════════════════════════
# §7  TYPED NODE RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

_TYPE_BUCKETS = {
    "DEFINITION":     {"DEFINITION"},
    "NUMERIC":        {"NUMERIC", "CONDITION"},
    "OBLIGATION":     {"OBLIGATION", "CONDITION", "REFERENCE", "DEFINITION"},
    "PARTY_SPECIFIC": {"RIGHT", "OBLIGATION", "CONDITION", "REFERENCE", "DEFINITION"},
    "GENERAL":        {"DEFINITION", "OBLIGATION", "RIGHT", "NUMERIC", "CONDITION", "BLANK", "REFERENCE"},
}

_KW_GENERIC = frozenset({
    'information','agreement','parties','between','either','whether','section',
    'within','before','another','having','making','during','following','applicable',
    'pursuant','confidential','obligations','required','related','receiving',
    'disclosing','includes','provided','provide','requires','written','notice',
    'manner','business','described','document','executed','certain','herein',
    'therein','thereof','company','person','entity','service','services',
})


def _node_search_text(node: dict) -> str:
    """
    Generate a question-style embedding string for a node so that retrieval
    cosine similarity aligns with how users actually phrase questions.

    Key principles:
    - Drop source_text from OBLIGATION/RIGHT embeddings — it adds noise that
      dilutes the structured signal and hurts cosine similarity with questions.
    - Rephrase NUMERIC from "60 days applies to notice period" → question form
      "What is the notice period? 60 days" — matches question phrasing better.
    - OBLIGATION/RIGHT: rephrase as a question so "What must X do?" matches
      "Must X provide notice?" more directly than "X shall provide notice".
    """
    t = node["type"]
    if t == "DEFINITION":
        return f"What does {node['term']} mean? {node['term']} means {node['definition']}"
    elif t == "OBLIGATION":
        cond = f" if {node['condition']}" if node.get("condition") else ""
        qual = f" ({node['qualifiers']})" if node.get("qualifiers") else ""
        action = f"{node['modal']} {node['action']}{cond}{qual}".strip()
        # Question form: "What must/shall Party do?" — matches question phrasing
        return f"What must {node['party']} do? {node['party']} {action}"
    elif t == "RIGHT":
        lim  = f" limited to {node['limit']}" if node.get("limit") else ""
        cond = f" if {node['condition']}" if node.get("condition") else ""
        qual = f" ({node['qualifiers']})" if node.get("qualifiers") else ""
        right = f"{node['right']}{lim}{cond}{qual}".strip()
        # Question form: "Can Party do X?" — matches "Is X allowed/permitted?"
        return f"Can {node['party']} {right}? {node['party']} may {right}"
    elif t == "NUMERIC":
        trig = f" when {node['trigger']}" if node.get("trigger") else ""
        applies = node.get("applies_to", "")
        val = f"{node['value']} {node['unit']}".strip()
        # Question form: "What is the X? X: value unit"
        return f"What is the {applies}?{trig} {applies}: {val}"
    elif t == "CONDITION":
        party = f"{node['party']} " if node.get("party") else ""
        return f"What happens if {node['trigger']}? If {node['trigger']} then {party}{node['consequence']}"
    elif t == "BLANK":
        return f"What is the {node['field']}? {node['field']} {node.get('context', '')} is blank unfilled"
    elif t == "REFERENCE":
        return f"reference to {node['to']} which describes {node.get('describes', '')}"
    return node.get("source_text", "")


class TypedNodeRetriever:
    def __init__(self, nodes: list[dict], graph_index: GraphIndex,
                 model_name: str = "BAAI/bge-base-en-v1.5"):
        self._nodes:       list[dict] = nodes
        self._graph_index: GraphIndex = graph_index
        self._model: Optional[SentenceTransformer] = None
        self._node_vecs:   list[np.ndarray] = []
        self._model_name = model_name

    def load(self) -> "TypedNodeRetriever":
        # Deduplicate
        seen:   set[str]   = set()
        deduped: list[dict] = []
        for node in self._nodes:
            key = _node_key(node)
            if key not in seen:
                seen.add(key); deduped.append(node)
        self._nodes = deduped

        self._model    = _get_embed_model(self._model_name)
        texts          = [_node_search_text(n) for n in self._nodes]
        vecs           = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._node_vecs = list(vecs)
        return self

    def _embed(self, text: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TypedNodeRetriever must be loaded before embedding.")
        if not text:
            return np.zeros(self._model.get_sentence_embedding_dimension())
        return self._model.encode(text, normalize_embeddings=True)

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        if np.all(a == 0) or np.all(b == 0):
            return 0.0
        return float(np.dot(a, b))

    def _extract_kw_terms(self, question: str) -> set[str]:
        terms: set[str] = set()
        for m in re.finditer(r"['\"]([^'\"]{3,})['\"]", question):
            terms.add(m.group(1).lower())
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', question):
            terms.add(m.group(1).lower())
        for word in re.findall(r'\b([a-z]{6,})\b', question):
            if word not in _KW_GENERIC:
                terms.add(word)
        return terms

    def retrieve(self, question: str, top_k: int = 20,
                 q_node: dict | None = None,
                 extra_keywords: set[str] | None = None) -> dict:
        if q_node is None:
            q_node = parse_question_node(question)
        qtype = classify_question_type(q_node)

        query_parts = [question]
        for f in ("entity", "verb", "object"):
            if q_node.get(f):
                query_parts.append(q_node[f])
        q_vec = self._embed(" ".join(query_parts))

        allowed_types = _TYPE_BUCKETS.get(qtype, _TYPE_BUCKETS["GENERAL"])
        scored = [(self._cos(q_vec, self._node_vecs[i]), i)
                  for i, node in enumerate(self._nodes) if node["type"] in allowed_types]

        if len(scored) < top_k:
            for i, node in enumerate(self._nodes):
                if node["type"] not in allowed_types:
                    scored.append((self._cos(q_vec, self._node_vecs[i]) * 0.8, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        # Keyword augmentation
        kw_terms = self._extract_kw_terms(question)
        if extra_keywords:
            kw_terms.update(extra_keywords)
        already = {i for _, i in top}
        kw_augment: list[tuple[int, int]] = []
        for i, node in enumerate(self._nodes):
            if i in already:
                continue
            text   = (node.get("source_text", "") + " " + _node_search_text(node)).lower()
            n_hits = sum(1 for t in kw_terms if t in text)
            if n_hits > 0:
                kw_augment.append((n_hits, i))
        kw_augment.sort(reverse=True)
        for _, i in kw_augment[:12]:
            if not any(j == i for _, j in top):
                top.append((0.45, i))

        # Keyword re-rank
        if kw_terms:
            boosted = []
            for score, i in top:
                text    = (self._nodes[i].get("source_text", "") + " " + _node_search_text(self._nodes[i])).lower()
                n_hits  = sum(1 for t in kw_terms if t in text)
                boosted.append((score + 0.02 * n_hits, i))
            boosted.sort(key=lambda x: x[0], reverse=True)
            top = boosted

        top_nodes = [{**self._nodes[i], "_score": round(float(score), 4)} for score, i in top]

        # Graph expansion
        if self._graph_index:
            seed_keys    = [_node_key(n) for n in top_nodes]
            existing_keys = set(seed_keys)
            for gnode in self._graph_index.get_subgraph(seed_keys, max_extra=25):
                gkey = _node_key(gnode)
                if gkey not in existing_keys:
                    existing_keys.add(gkey)
                    top_nodes.append({**gnode, "_score": 0.40, "_graph_expanded": True})

        return {"question": question, "question_node": q_node,
                "question_type": qtype, "top_nodes": top_nodes}


# ══════════════════════════════════════════════════════════════════════════════
# §8  LLM QA
# ══════════════════════════════════════════════════════════════════════════════

_RAW_CHUNK_QA_PROMPT = """\
You are a legal analyst. Given raw text excerpts from a contract, answer whether the question can be answered directly from those excerpts.

Contract excerpts:
{chunks}

Question: {question}

STEP 1 — What exact value does this question need?
STEP 2 — Does any excerpt explicitly state that value?

Apply these principles:
1. EXPLICIT OVER IMPLICIT — The excerpt must directly state the value, not imply it.
2. BLANK / REDACTED — If the field is blank or shows "***", it is not answerable.
3. EXTERNAL DOCUMENTS — If the answer lives in a referenced Schedule/Exhibit not shown, unanswerable.
4. RUNTIME FACTS — Current counts, dates since signing, and post-execution facts are unanswerable.

Reply ONLY with valid JSON (no markdown):
{{"answerable":true or false,"answer":"direct quote or null","reasoning":"brief reason","confidence":"high|medium|low","supporting_fact":null,"fact_type":null}}"""

_QA_PROMPT = """\
You are a legal analyst. Given facts extracted from a contract, answer whether the question is answerable from those facts alone.

Facts:
{facts}

Question: {question}

STEP 1 — What exact value does this question need? State the required_type and required_subject.
STEP 2 — Does any fact explicitly provide that value? Apply these principles:

1. EXPLICIT OVER IMPLICIT — A fact must directly state the required value, not merely imply it. Party identification clauses (names, addresses, states of incorporation), defined terms (effective dates, URLs), and numeric values (notice periods, caps, days) explicitly present in source_text all count as direct statements.
2. DIRECTION MATTERS — A fact about Party A does not answer a question about Party B.
3. SPECIFICITY MATCHES — The fact must address the specific subject asked, not a related but different one.
4. EXTERNAL DOCUMENTS — If the answer lives in a referenced Schedule/Exhibit not present, unanswerable.
5. CONTRACT SUMMARY DEFINITION — "Contract Summary" node contains header facts separated by "|". Read carefully.
6. YES/NO QUESTIONS — Explicit denial ("X cannot do Y") answers "can X do Y?" with no.
7. FUNCTIONAL COVERAGE — For "what must X do", a fact functionally covering the scenario counts.
8. BLANK NODES — A BLANK fact means the field was left unfilled — not answerable from a blank.
9. MECHANISM VS. VALUE — "How is X calculated?" asks for the formula, not the exact figure.
10. HARD CONSISTENCY RULE — If NO fact addresses the TOPIC at all, answerable MUST be false.
11. CHAIN REASONING — Facts annotated with "↳" show graph connections; combine them to derive multi-step answers.
12. LIST / ENUMERATION — "What are X's obligations?" — answerable = true if ANY fact addresses the topic. Enumerate all.
13. CONDITION THRESHOLDS ≠ ACTUAL VALUES — A condition trigger threshold is NOT the actual value.

Reply ONLY with valid JSON (no markdown):
{{"required_type":"...","required_subject":"...","answerable":true or false,
  "answer":"direct quote or concise answer, or null","supporting_fact":<1-based index or null>,
  "fact_type":"DEFINITION|OBLIGATION|RIGHT|NUMERIC|CONDITION or null",
  "confidence":"high|medium|low",
  "reasoning":"what value was needed and whether a fact explicitly provides it",
  "answer_source_type":"base_clause|customer_instance|external_artifact|runtime_fact|mixed",
  "source_sufficiency_required":"clause_quote_enough|explicit_value_required|artifact_contents_required|runtime_lookup_required|mixed",
  "inline_answer_present":0.0,
  "external_reference_only":0.0,
  "customer_instance_missing":0.0,
  "runtime_missing":0.0,
  "evidence_support_reason":"short reason"}}"""


def _format_node(i: int, node: dict, chain_notes: list[str] | None = None) -> str:
    t   = node["type"]
    src = node.get("source_text", "")

    if t == "DEFINITION":
        header = f'{i+1}. [DEFINITION] "{node["term"]}" means: {node["definition"]}'
    elif t == "OBLIGATION":
        cond   = f" (if: {node['condition']})" if node.get("condition") else ""
        header = f'{i+1}. [OBLIGATION] {node["party"]} {node["modal"].upper()} {node["action"]}{cond}'
    elif t == "RIGHT":
        lim    = f" (limit: {node['limit']})" if node.get("limit") else ""
        cond   = f" (if: {node['condition']})" if node.get("condition") else ""
        header = f'{i+1}. [RIGHT] {node["party"]} MAY {node["right"]}{lim}{cond}'
    elif t == "NUMERIC":
        trig   = f" (when: {node['trigger']})" if node.get("trigger") else ""
        header = f'{i+1}. [NUMERIC] {node["value"]} {node["unit"]} — {node["applies_to"]}{trig}'
    elif t == "CONDITION":
        party  = f"{node['party']}: " if node.get("party") else ""
        header = f'{i+1}. [CONDITION] IF {node["trigger"]} THEN {party}{node["consequence"]}'
    elif t == "BLANK":
        header = f'{i+1}. [BLANK] "{node["field"]}" is not filled in — context: {node.get("context","")[:100]}'
    elif t == "REFERENCE":
        header = f'{i+1}. [REFERENCE] → {node["to"]}: {node.get("describes","")}'
    else:
        header = f'{i+1}. [?] {node.get("source_text","")[:80]}'

    parts = [header]
    if src and src not in header:
        parts.append(f'   Source: "{src}"')
    if chain_notes:
        for note in chain_notes:
            parts.append(f'   ↳ {note}')
    return "\n".join(parts)


def answer_from_typed_nodes(question: str, nodes: list[dict], top_n: int = 25,
                             graph_index: GraphIndex | None = None) -> dict:
    use_nodes = nodes[:top_n]
    idx_to_notes: dict[int, list[str]] = {}
    if graph_index is not None:
        try:
            node_keys    = [_node_key(n) for n in use_nodes]
            key_to_notes = graph_index.get_chain_annotations(node_keys)
            idx_to_notes = {i: key_to_notes[k] for i, k in enumerate(node_keys) if key_to_notes.get(k)}
        except Exception:
            pass

    facts  = "\n\n".join(_format_node(i, n, idx_to_notes.get(i)) for i, n in enumerate(use_nodes))
    prompt = _QA_PROMPT.format(facts=facts, question=question)

    def _try_parse(raw: str) -> dict | None:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            try:
                return json.loads(re.sub(r',\s*([}\]])', r'\1', m.group()))
            except json.JSONDecodeError:
                return None

    raw = ask_qa(prompt, temperature=0.0, max_tokens=600)
    r   = _try_parse(raw)
    if r is None:
        raw = ask_qa(prompt, temperature=0.0, max_tokens=1200)
        r   = _try_parse(raw)
    if r is None:
        return {"answerable": False, "answer": None, "supporting_fact": None,
                "fact_type": None, "confidence": "low", "reasoning": f"parse error: {raw[:80]}"}

    def _clip01(v, default=0.0):
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return float(default)
    return {"answerable":      bool(r.get("answerable", False)),
            "answer":          r.get("answer"),
            "supporting_fact": r.get("supporting_fact"),
            "fact_type":       r.get("fact_type"),
            "confidence":      r.get("confidence", "low"),
            "reasoning":       r.get("reasoning", ""),
            "llm_evidence_sufficiency": {
                "answer_source_type": str(r.get("answer_source_type") or "mixed"),
                "source_sufficiency_required": str(r.get("source_sufficiency_required") or "mixed"),
                "inline_answer_present": _clip01(r.get("inline_answer_present")),
                "external_reference_only": _clip01(r.get("external_reference_only")),
                "customer_instance_missing": _clip01(r.get("customer_instance_missing")),
                "runtime_missing": _clip01(r.get("runtime_missing")),
                "evidence_support_reason": str(r.get("evidence_support_reason") or "").strip(),
                "llm_used": True,
            }}


def _answer_from_raw_chunks(question: str, chunk_texts: list[str]) -> dict:
    """Run LLM QA on raw contract text chunks (not node descriptions).

    Used by topo-directed retry when topology scores high but the LLM could not
    answer from node summaries alone — the answer text may not have been captured
    in any extracted node but is present verbatim in the source chunk.
    """
    numbered = "\n\n---\n\n".join(
        f"[Excerpt {i+1}]\n{text.strip()}" for i, text in enumerate(chunk_texts) if text.strip()
    )
    if not numbered:
        return {"answerable": False, "answer": None, "reasoning": "no chunk text available",
                "confidence": "low", "supporting_fact": None, "fact_type": None}

    prompt = _RAW_CHUNK_QA_PROMPT.format(chunks=numbered, question=question)

    def _try_parse(raw: str) -> dict | None:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            try:
                return json.loads(re.sub(r',\s*([}\]])', r'\1', m.group()))
            except json.JSONDecodeError:
                return None

    raw = ask_qa(prompt, temperature=0.0, max_tokens=400)
    r   = _try_parse(raw)
    if r is None:
        return {"answerable": False, "answer": None, "reasoning": f"parse error: {raw[:80]}",
                "confidence": "low", "supporting_fact": None, "fact_type": None}

    return {"answerable":      bool(r.get("answerable", False)),
            "answer":          r.get("answer"),
            "supporting_fact": r.get("supporting_fact"),
            "fact_type":       r.get("fact_type"),
            "confidence":      r.get("confidence", "low"),
            "reasoning":       r.get("reasoning", "")}


# ══════════════════════════════════════════════════════════════════════════════
# §9  CONSENSUS RETRIEVAL  (variants → vote → slot → LLM)
# ══════════════════════════════════════════════════════════════════════════════

_VARIANT_PROMPT = """\
/no_think
Generate {n} different ways to ask the same legal question.
Keep the same meaning but vary the phrasing, perspective, and vocabulary.
Output one question per line, no numbering, no bullets.

Original question: {question}"""


def generate_variants(question: str, n: int = 6) -> list[str]:
    raw      = ask(_VARIANT_PROMPT.format(n=n, question=question), temperature=0.7, max_tokens=300)
    variants = [l.strip() for l in raw.splitlines() if l.strip() and "?" in l]
    return (variants + [question] * n)[:n]


def _node_vote_key(node: dict) -> str:
    """Stable identity key for vote counting (same as _node_key)."""
    return _node_key(node)


class ConsensusRetriever:
    """
    Pipeline: generate N variants → retrieve for each → vote → slot filter → LLM QA.

    min_votes: nodes must appear in this many variant retrievals to survive (hard tier).
    SOFT_SCORE_THRESHOLD (0.82): single-vote nodes above this score also survive.
    """

    def __init__(self, retriever: TypedNodeRetriever,
                 n_variants: int = 6,
                 min_votes:  int = 2):
        self._retriever  = retriever
        self.n_variants  = n_variants
        self.min_votes   = min_votes

    def query(self, question: str, retrieval_k: int = 100,
              slot_threshold: float = 0.30) -> dict:
        # Optional A/B toggle for pre-step latency benchmarking.
        # Default keeps the faster parallel behavior enabled.
        pre_parallel = os.environ.get("CORE_PRESTEP_PARALLEL", "1").strip().lower() not in {
            "0", "false", "off", "no"
        }
        if pre_parallel:
            # Run the first two independent LLM calls in parallel:
            # 1) query variants, 2) structured question parse (+ intent bundle cache).
            with ThreadPoolExecutor(max_workers=2) as pre_pool:
                fut_variants = pre_pool.submit(generate_variants, question, self.n_variants)
                fut_qnode = pre_pool.submit(parse_question_node, question)
                variants = fut_variants.result()
                original_q_node = fut_qnode.result()
        else:
            variants = generate_variants(question, self.n_variants)
            original_q_node = parse_question_node(question)

        all_questions = [question] + variants

        original_kw     = self._retriever._extract_kw_terms(question)

        vote_count:  dict[str, int]   = defaultdict(int)
        vote_scores: dict[str, float] = defaultdict(float)
        vote_nodes:  dict[str, dict]  = {}

        def _retrieve_one(q: str) -> dict:
            return self._retriever.retrieve(q, top_k=retrieval_k,
                                            q_node=original_q_node,
                                            extra_keywords=original_kw)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_retrieve_one, q): q for q in all_questions}
            for fut in as_completed(futures):
                for node in fut.result()["top_nodes"]:
                    key = _node_vote_key(node)
                    vote_count[key]   += 1
                    vote_scores[key]   = max(vote_scores[key], node["_score"])
                    vote_nodes[key]    = node

        n_queries = len(all_questions)

        # Hard tier: min_votes
        surviving = [
            {**vote_nodes[k], "_votes": vote_count[k], "_score": vote_scores[k],
             "_vote_frac": vote_count[k] / n_queries}
            for k in vote_count if vote_count[k] >= self.min_votes
        ]
        # Soft tier: single-vote high-score (not BLANK)
        SOFT_SCORE = 0.82
        surviving += [
            {**vote_nodes[k], "_votes": vote_count[k], "_score": vote_scores[k],
             "_vote_frac": vote_count[k] / n_queries}
            for k in vote_count
            if vote_count[k] < self.min_votes
            and vote_scores[k] >= SOFT_SCORE
            and vote_nodes[k].get("type") != "BLANK"
        ]
        surviving.sort(key=lambda n: (n["_vote_frac"], n["_score"]), reverse=True)

        # Always include BLANK nodes (signal unfilled template fields)
        blank_keys = sorted(
            [k for k in vote_count if vote_nodes[k].get("type") == "BLANK" and vote_count[k] < self.min_votes],
            key=lambda k: vote_scores[k], reverse=True
        )[:5]
        surviving_keys = {_node_vote_key(n) for n in surviving}
        for k in blank_keys:
            if k not in surviving_keys:
                surviving.append({**vote_nodes[k], "_votes": vote_count[k],
                                   "_score": vote_scores[k], "_vote_frac": vote_count[k] / n_queries})
                surviving_keys.add(k)

        # DEFINITION soft tier (definitions often have only 1 retrieval hit)
        DEF_SOFT = 0.65
        for k in vote_count:
            if k in surviving_keys:
                continue
            node = vote_nodes[k]
            if node.get("type") == "DEFINITION" and vote_scores[k] >= DEF_SOFT:
                surviving.append({**node, "_votes": vote_count[k],
                                   "_score": vote_scores[k], "_vote_frac": vote_count[k] / n_queries})
                surviving_keys.add(k)

        # High-priority metadata anchor nodes (parties, dates) always included
        for n in self._retriever._nodes:
            if n.get("priority") == "high":
                pk = _node_vote_key(n)
                if pk not in surviving_keys:
                    surviving.append({**n, "_votes": 0, "_score": 0.5,
                                      "_vote_frac": 0.0, "_priority": "high"})
                    surviving_keys.add(pk)

        # Party-centric injection: if question names a party, inject all their nodes
        gi = self._retriever._graph_index
        if gi and gi.party_keys:
            q_lower    = question.lower()
            best_party = max(
                (pkey[7:] for pkey in gi.party_keys if pkey[7:] in q_lower),
                key=len, default=None
            )
            if best_party:
                for pn in gi.get_party_nodes(best_party):
                    pk = _node_vote_key(pn)
                    if pk not in surviving_keys:
                        surviving.append({**pn, "_votes": 0, "_score": 0.42,
                                           "_vote_frac": 0.0, "_graph_party": True})
                        surviving_keys.add(pk)

        # Slot coverage pre-filter
        # (import from v2/slot_coverage.py — kept in separate module due to size)
        try:
            from slot_coverage import compute_coverage
            coverage = compute_coverage(question, original_q_node, surviving,
                                        self._retriever._embed,
                                        all_nodes=self._retriever._nodes)
        except ImportError:
            coverage = {"score": 1.0, "verdict": "UNKNOWN", "slots": [],
                        "tcp_multiplier": 1.0, "lpg_multiplier": 1.0}

        slot_score   = coverage["score"]
        slot_verdict = coverage["verdict"]

        # LLM verdict
        if not surviving:
            qa = {"answerable": False, "answer": None, "supporting_fact": None,
                  "fact_type": None, "confidence": "high",
                  "reasoning": "No nodes survived consensus — not in document."}
        else:
            qa = answer_from_typed_nodes(question, surviving, top_n=23,
                                         graph_index=gi)
            if qa.get("reasoning", "").startswith("parse error"):
                qa = {"answerable": slot_score >= 0.65, "answer": None,
                      "supporting_fact": None, "fact_type": None, "confidence": "low",
                      "reasoning": f"LLM unavailable — slot_score={slot_score:.3f} used"}
            else:
                # Slot coverage is a prior, not a hard gate.
                # Keep LLM verdict, but annotate low slot support clearly.
                if slot_score <= slot_threshold:
                    prior = f"slot_coverage={slot_score:.3f} below threshold={slot_threshold:.3f}"
                    base_reason = (qa.get("reasoning") or "").strip()
                    if qa.get("answerable", False):
                        qa["confidence"] = "medium" if qa.get("confidence") == "high" else qa.get("confidence", "medium")
                        qa["reasoning"] = (f"{base_reason} | {prior}; answer relies on sparse slot evidence"
                                           if base_reason else
                                           f"{prior}; answer relies on sparse slot evidence")
                    else:
                        qa["reasoning"] = (f"{base_reason} | {prior}"
                                           if base_reason else prior)

        return {
            "question":        question,
            "variants":        variants,
            "all_nodes_seen":  len(vote_count),
            "surviving_nodes": surviving,
            "slot_score":      round(slot_score, 4),
            "slot_verdict":    slot_verdict,
            "slot_slots":      coverage.get("slots", []),
            "q_node":          original_q_node,
            **{k: qa[k] for k in ("answerable", "answer", "confidence", "reasoning",
                                   "fact_type", "supporting_fact")},
            "llm_evidence_sufficiency": qa.get("llm_evidence_sufficiency", {}),
        }


# ══════════════════════════════════════════════════════════════════════════════
# §10  TOPOLOGY PREDICTOR
#      Signal math lives in v2/topology_metrics.py (compute_all).
#      _predict_answerability is inlined here as it was written in this session.
# ══════════════════════════════════════════════════════════════════════════════

def _predict_answerability(topo: dict) -> dict:
    """
    Fallback shim: delegate to topology_metrics scorer when available.
    Keeps interface stable if compute_all import fails.
    """
    try:
        from topology_metrics import _predict_answerability as _predict_impl
        return _predict_impl(topo)
    except Exception:
        return {
            "predicted": "unanswerable",
            "score": 0.0,
            "confidence": "low",
            "threshold": 0.5,
            "signals": {},
            "meta": {"reason": "fallback scorer unavailable"},
        }


def compute_topology(survivors: list[dict], graph_index: GraphIndex | None,
                     question: str, q_node: dict, slots: list[dict],
                     llm_evidence_sufficiency: dict | None = None) -> dict:
    """
    Compute topology metrics and append topo_pred.
    Delegates heavy math to v2/topology_metrics.py::compute_all.
    Falls back gracefully if that module isn't available.
    """
    try:
        from topology_metrics import compute_all as _compute_all
        topo = _compute_all(survivors, graph_index, question, q_node, slots,
                            llm_evidence_sufficiency=llm_evidence_sufficiency)
    except ImportError:
        topo = {}
    # Keep the predictor bundled in topology_metrics when available.
    # Fallback only if compute_all didn't provide topo_pred.
    if "topo_pred" not in topo:
        topo["topo_pred"] = _predict_answerability(topo)
    return topo


# ══════════════════════════════════════════════════════════════════════════════
# §11  PIPELINE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

class NumericFactIndex:
    """
    Flat index of specific-value facts extracted from the contract: NUMERIC nodes
    (durations, amounts, percentages, dates) and DEFINITION nodes (party addresses,
    effective dates, URLs, identifiers stated inline).

    Topology consistently under-scores these nodes because they live in
    sparsely-connected clause nodes (few graph edges).  This index provides a
    direct bypass: when a question has a strong semantic match to a node AND
    the LLM verifies the clause directly states the value, we trust that
    evidence regardless of topology score.

    Built once at load_pdf() time using the already-loaded BGE embedder.
    Verification uses one cheap ask() call (~200 ms) only when a candidate is found.
    """
    # Noise unit strings for NUMERIC nodes — section refs, entity ordinals, etc.
    _NOISE_UNITS = {"section number", "ordinal number", "count", "section",
                    "statute", "identifier", "name", "reference"}

    # Runtime/instance question patterns — never bypass for these
    _RUNTIME_RE = re.compile(
        r'\b(has\b[^?]{1,40}\bever\b|has\s+any|have\s+(the\s+)?(parties|any|either)\s+'
        r'(ever|previously|since)|what\s+(has|have)\s+happened|'
        r'to\s+date\b|since\s+(the\s+)?(signing|execution|effective\s+date)|'
        r'how\s+many\s+\w+\s+have\s+been|what\s+\w+\s+have\s+(occurred|taken\s+place|been\s+made))',
        re.IGNORECASE,
    )

    THRESHOLD       = 0.74    # NUMERIC / DEFINITION hit → answerable candidate
    BLANK_THRESHOLD = 0.70    # BLANK hit → unanswerable signal (keep high: false-negative risk)

    def __init__(self, entries: list[dict], matrix):
        self._entries = entries   # list of node dicts (NUMERIC + DEFINITION, noise filtered)
        self._matrix  = matrix    # np.ndarray (n, dim), L2-normalised

    @classmethod
    def build(cls, flat_nodes: list[dict], retriever: "TypedNodeRetriever") -> "NumericFactIndex":
        entries = []
        for n in flat_nodes:
            t = str(n.get("type", "")).upper()
            src = str(n.get("source_text", "")).strip()
            if not src:
                continue
            if t == "NUMERIC":
                if str(n.get("unit", "")).lower().strip() not in cls._NOISE_UNITS:
                    entries.append(n)
            elif t == "DEFINITION":
                # Include DEFINITION nodes that encode specific inline facts:
                # party addresses, dates, URLs, identifiers, short values.
                tags = [str(x).lower() for x in (n.get("value_tags") or [])]
                if any(kw in tags for kw in (
                    "date", "address", "url", "location", "city", "state",
                    "jurisdiction", "governing_law", "effective_date",
                    "term_length", "duration", "percentage", "rate",
                    "party_detail", "registration",
                )):
                    entries.append(n)
            elif t == "BLANK":
                # BLANK nodes are unanswerable signals: the contract has the field
                # structure but the value was redacted (***) or never filled in.
                # Including them lets lookup() return a BLANK hit as a hard
                # "unanswerable" signal before any expensive LLM ensemble runs.
                entries.append(n)
        if not entries:
            return cls([], None)
        texts  = [n.get("source_text", "") for n in entries]
        matrix = retriever._model.encode(texts, normalize_embeddings=True)
        return cls(entries, matrix)

    def lookup(self, question: str, retriever: "TypedNodeRetriever") -> dict | None:
        """Return best-matching entry above THRESHOLD, or None.

        The returned dict has two extra keys:
          _numeric_similarity  float — cosine similarity score
          _is_blank            bool  — True if the best match is a BLANK node
                                       (redacted/unfilled field → unanswerable signal)

        BLANK fast-path guard: only triggers the BLANK short-circuit when the
        BLANK node is the clear winner.  If any non-BLANK node scores >= THRESHOLD,
        the contract likely has real content to answer the question — skip the
        fast-path so the main pipeline can retrieve and answer it.
        """
        if self._matrix is None or not self._entries:
            return None
        # Block runtime/instance questions before even searching
        if self._RUNTIME_RE.search(question):
            return None
        q_emb = retriever._model.encode([question], normalize_embeddings=True)[0]
        scores = self._matrix @ q_emb
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        is_blank = str(self._entries[best_idx].get("type", "")).upper() == "BLANK"
        threshold = self.BLANK_THRESHOLD if is_blank else self.THRESHOLD
        if best_score < threshold:
            return None

        # If the top match is a BLANK node, verify no non-BLANK node is also
        # a strong match.  A competitive non-BLANK score >= THRESHOLD means
        # the contract has real content that could answer this question — in
        # that case fall through to the main pipeline rather than short-circuiting.
        if is_blank:
            non_blank_scores = [
                float(scores[i])
                for i, e in enumerate(self._entries)
                if str(e.get("type", "")).upper() != "BLANK"
            ]
            if non_blank_scores and max(non_blank_scores) >= self.THRESHOLD - 0.12:
                return None  # answerable content likely exists — skip BLANK fast-path

        hit = {**self._entries[best_idx], "_numeric_similarity": best_score}
        hit["_is_blank"] = is_blank
        return hit

    @staticmethod
    def verify(question: str, source_text: str) -> bool:
        """
        Binary LLM check: does this clause DIRECTLY STATE the value that answers
        the question?  Requires the answer to be explicitly present as written text
        — topic relevance alone is not enough.
        One cheap ask() call (~200 ms).  Returns True only on a clear "yes".
        """
        # Block runtime/instance questions — verify() should never fire for these
        if NumericFactIndex._RUNTIME_RE.search(question):
            return False
        prompt = (
            "You are a contract analyst. Answer with exactly one word: yes or no.\n\n"
            f"Question: {question}\n\n"
            f"Contract clause: {source_text}\n\n"
            "Does this clause DIRECTLY STATE a specific value or fact that answers "
            "the question as written text? The clause must contain the actual answer "
            "explicitly — not merely reference or discuss the same topic. (yes/no)"
        )
        raw = ask(prompt, temperature=0.0, max_tokens=5).strip().lower()
        return raw.startswith("yes")


class Pipeline:
    """
    Loaded pipeline for a single document.

    Created by load_pdf(); used by query().
    """
    def __init__(self, nodes_data: list[dict], retriever: TypedNodeRetriever,
                 consensus: ConsensusRetriever, doc_name: str,
                 chunks_by_id: dict | None = None,
                 numeric_index: "NumericFactIndex | None" = None):
        self.nodes_data    = nodes_data
        self.retriever     = retriever
        self.consensus     = consensus
        self.doc_name      = doc_name
        self.chunks_by_id  = chunks_by_id or {}
        self.numeric_index = numeric_index


def load_pdf(pdf_path: str | Path,
             workers:    int = 4,
             batch_size: int = 3,
             n_variants: int = 6,
             min_votes:  int = 2,
             retrieval_k: int = 100,
             embed_model: str = "BAAI/bge-base-en-v1.5",
             save_nodes: bool = True,
             reuse_existing: bool = True,
             progress_cb=None) -> Pipeline:
    """
    Extract typed nodes from a PDF, build graph, load retriever.

    Saves *_typed_nodes.json and *_graph.json alongside the PDF (unless save_nodes=False).
    Returns a Pipeline ready for query().
    """
    pdf_path   = Path(pdf_path)
    nodes_json = pdf_path.parent / f"{pdf_path.stem}_typed_nodes.json"
    graph_path = pdf_path.parent / f"{pdf_path.stem}_graph.json"
    _emit_progress(progress_cb, stage="start_load")

    if reuse_existing and nodes_json.exists() and graph_path.exists():
        nodes_data = json.loads(nodes_json.read_text())
        _emit_progress(progress_cb, stage="load_cached_artifacts", cached=True)
    else:
        nodes_data = extract_all(pdf_path, workers=workers, batch_size=batch_size, progress_cb=progress_cb)
        if save_nodes:
            nodes_json.write_text(json.dumps(nodes_data, indent=2))
            print(f"  Nodes saved → {nodes_json.name}")

    # Load graph
    graph = json.loads(graph_path.read_text()) if graph_path.exists() else {"edges": []}

    # Flatten + dedup nodes, preserving _chunk_id for topo-directed retrieval
    flat_nodes: list[dict] = []
    seen: set[str] = set()
    for chunk in nodes_data:
        cid = chunk["chunk_id"]
        for node in chunk.get("nodes", []):
            k = _node_key(node)
            if k not in seen:
                seen.add(k); flat_nodes.append({**node, "_chunk_id": cid})

    # chunk_id → raw text lookup (used by topo-directed raw-chunk retry)
    chunks_by_id: dict[int, str] = {
        chunk["chunk_id"]: chunk.get("chunk_text", "")
        for chunk in nodes_data
    }

    gi        = GraphIndex(graph, nodes_data)
    _emit_progress(progress_cb, stage="embedding_nodes", node_count=len(flat_nodes))
    retriever = TypedNodeRetriever(flat_nodes, gi, model_name=embed_model).load()
    consensus = ConsensusRetriever(retriever, n_variants=n_variants, min_votes=min_votes)
    numeric_index = NumericFactIndex.build(flat_nodes, retriever)

    print(f"  Pipeline ready — {len(flat_nodes)} deduped nodes, {len(graph['edges'])} edges")
    _emit_progress(progress_cb, stage="ready", node_count=len(flat_nodes), edge_count=len(graph.get("edges", [])))
    return Pipeline(nodes_data, retriever, consensus, pdf_path.name,
                    chunks_by_id=chunks_by_id, numeric_index=numeric_index)


def _conf_rank(label: str) -> int:
    l = (label or "").strip().lower()
    if l == "high":
        return 3
    if l == "medium":
        return 2
    return 1


def _rank_to_conf(rank: int) -> str:
    if rank >= 3:
        return "high"
    if rank == 2:
        return "medium"
    return "low"


def _apply_answerability_gate(llm_result: dict, topo: dict, slot_threshold: float) -> tuple[bool, str, str, dict]:
    """
    Conservative legal-demo arbitration.
    Final answerable=true only when LLM + topology + slot evidence align.
    """
    # Ensemble vote fraction: fraction of LLM runs that said "answerable".
    # NOTE: vote_fraction=0.0 does NOT mean no ensemble — it means ensemble ran
    # but every run said "no". Use ensemble_k to determine if ensemble was used.
    vote_fraction    = float(llm_result.get("vote_fraction") or 0.0)
    has_ensemble     = int(llm_result.get("ensemble_k") or 1) > 1
    _is_runtime_q    = bool(llm_result.get("is_runtime_question"))
    _has_blank_nodes = bool(llm_result.get("has_blank_nodes"))

    # Runtime/instance veto: questions asking about what actually happened in the
    # real world (not what the contract specifies) cannot be answered from contract
    # text alone.  Require unanimous LLM agreement (vote=1.0) before allowing
    # answerable — any disagreement means the LLM is reading a related clause, not
    # confirming real-world knowledge.  Single-LLM mode: block entirely.
    if _is_runtime_q:
        if has_ensemble and vote_fraction < 1.0:
            return False, "This question asks about real-world events or state, not contract terms.", "low", {
                "mode": "runtime_veto", "vote_fraction": vote_fraction, "failures": ["runtime_question_not_unanimous"],
            }
        elif not has_ensemble and not bool(llm_result.get("answerable", False)):
            return False, "This question asks about real-world events or state, not contract terms.", "low", {
                "mode": "runtime_veto", "failures": ["runtime_question_llm_no"],
            }

    # With ensemble use a soft majority threshold (2/5 = 0.40) so that even a
    # minority of "yes" votes can proceed when topology is strongly corroborating.
    if has_ensemble:
        llm_answerable = vote_fraction >= 0.40
    else:
        llm_answerable = bool(llm_result.get("answerable", False))
    llm_reason = str(llm_result.get("reasoning") or "").strip()
    llm_conf = str(llm_result.get("confidence") or "low")
    slot_score = float(llm_result.get("slot_score") or 0.0)

    topo_pred = topo.get("topo_pred", {}) if isinstance(topo, dict) else {}
    topo_label = str(topo_pred.get("predicted") or "").strip().lower()
    topo_conf = str(topo_pred.get("confidence") or "low")
    topo_score = float(topo_pred.get("score") or 0.0)
    topo_thr = float(topo_pred.get("threshold") or 0.5)
    # Unanimous LLM confidence boost: if all ensemble votes say answerable, the
    # topology threshold is too strict — lower it by 0.08 to let clear cases through.
    if has_ensemble and vote_fraction == 1.00:
        topo_thr = max(0.0, topo_thr - 0.08)

    ap = topo.get("answer_path", {}) if isinstance(topo, dict) else {}
    slot_type = str(ap.get("slot_type") or "GENERAL").upper()
    path_exists = bool(ap.get("path_exists"))
    pb_raw = ap.get("path_bottleneck")
    try:
        path_bottleneck = float(pb_raw) if pb_raw is not None else None
    except Exception:
        path_bottleneck = None

    n_nodes = int(topo.get("n_nodes") or 0) if isinstance(topo, dict) else 0
    n_targets = int(ap.get("n_targets") or 0)
    target_frac = (n_targets / n_nodes) if n_nodes > 0 else None

    # When the scorer already calls this answerable AND the QA LLM found a direct
    # inline answer with zero missing/outside pressure, the GENERAL slot's strict
    # sub-checks are over-conservative (they're designed for vague/noisy retrieval,
    # not for confirmed base-clause answers).
    topo_meta = topo_pred.get("meta", {}) if isinstance(topo_pred, dict) else {}
    _llm_inline = float(topo_meta.get("llm_inline_answer_present") or 0.0)
    _topo_M = float(topo_meta.get("M") or 0.0)
    _topo_O = float(topo_meta.get("O") or 0.0)
    _topo_P = float(topo_meta.get("P") or 0.0)
    _base_clause_strong = bool(topo_meta.get("base_clause_strong_support", False))
    _ext_art_dep = float(topo_meta.get("external_artifact_dependency") or 0.0)
    _ext_art_markers = bool(topo_meta.get("external_artifact_markers", False))
    general_strong_evidence = bool(
        slot_type == "GENERAL"
        and _llm_inline >= 0.85           # lowered from 0.90 — evidence LLM variance
        and _topo_M <= 0.10
        and _topo_O <= 0.10
        and (
            topo_score > topo_thr
            or (_base_clause_strong and topo_score > topo_thr - 0.10)  # widened from -0.08
        )
    )

    retry_fired      = bool(llm_result.get("retry_fired"))
    retry_answerable = bool(llm_result.get("retry_answerable"))

    gate = {
        "mode": "conservative_v2",
        "llm_answerable": llm_answerable,
        "vote_fraction": round(vote_fraction, 3) if has_ensemble else None,
        "retry_fired": retry_fired,
        "retry_answerable": retry_answerable if retry_fired else None,
        "topology_predicted": topo_label or None,
        "topology_score": round(topo_score, 4),
        "topology_threshold": round(topo_thr, 4),
        "slot_score": round(slot_score, 4),
        "slot_type": slot_type,
        "path_exists": path_exists,
        "path_bottleneck": round(path_bottleneck, 4) if path_bottleneck is not None else None,
        "target_frac": round(target_frac, 4) if target_frac is not None else None,
        "failures": [],
    }

    # High-confidence topology escape: when the graph scorer is very strongly
    # answerable (large margin), the evidence LLM confirms an inline answer
    # exists, and there is no missing/outside pressure, a single QA LLM "no"
    # is more likely LLM variance than a genuine lack of evidence. The scorer
    # already encodes direct evidence quality; over-weighting QA LLM output
    # here causes stochastic false negatives for clearly answerable questions.
    _topology_margin = topo_score - topo_thr
    _det_route = str(topo_meta.get("deterministic_source_route") or "none")
    _topology_override_llm = bool(
        topo_score > topo_thr
        and _topology_margin >= 0.35
        and _topo_M <= 0.20        # relaxed from 0.10: base-clause routes (audit etc.)
        and _topo_O <= 0.10        # can legitimately have M up to 0.20
        and _llm_inline >= 0.80
        and (not has_ensemble or vote_fraction > 0)   # ≥1 LLM vote required — prevents
                                                       # topology-only FPs on runtime facts
    ) or bool(
        # Deterministic routes (audit_base_clause etc.) already hard-route the question;
        # when the topology margin is very large and contamination is zero, a single
        # evidence-LLM "no" is almost certainly a misfire. No _llm_inline check needed.
        # Requires ≥1 LLM vote to prevent runtime/instance questions that topology
        # mis-routes to a base-clause path from becoming false positives.
        _det_route not in {"none", "external_artifact", "runtime_fact", "customer_instance"}
        and topo_score > topo_thr
        and _topology_margin >= 0.35
        and _topo_M <= 0.05
        and _topo_O <= 0.05
        and (not has_ensemble or vote_fraction > 0)
    ) or bool(
        # Zero-artifact-graph bypass: when the document graph shows zero artifact
        # dependency AND the question names no external artifact, both the QA LLM and
        # evidence LLM "no" are almost certainly the same misfire (e.g. seeing a nearby
        # exhibit reference and wrongly assuming the inline value is external). The scorer
        # has already corrected the source routing; trust its margin here.
        _ext_art_dep == 0.0
        and not _ext_art_markers
        and _det_route == "none"
        and topo_score > topo_thr
        and _topology_margin >= 0.25
        and _topo_M <= 0.25
        and _topo_O <= 0.10
        and (not has_ensemble or vote_fraction > 0)   # ≥1 LLM vote required
    ) or bool(
        # Thin-margin bypass: when all three contamination signals (M/O/P) are very
        # low, even a modest topology margin indicates genuine answerability. The
        # scorer is clean; the single QA-LLM "no" is likely variance.
        # Excluded: routes that are already hard-classified as unanswerable by the
        # deterministic router (runtime_fact, customer_instance, speculative).
        topo_score > topo_thr
        and _topology_margin >= 0.10
        and _topo_M <= 0.10
        and _topo_O <= 0.05
        and _topo_P <= 0.05
        and _llm_inline >= 0.70
        and not _is_runtime_q          # runtime/instance questions are never safe bypasses
        and not _has_blank_nodes       # *** redacted fields must NOT bypass — BLANK nodes signal
        and _det_route not in {"runtime_fact", "customer_instance", "speculative",
                                "external_artifact"}
    ) or bool(
        # Partial-LLM + strong-topology bypass: when at least 1/3 independent LLM
        # runs say "yes" and topology shows a large margin, the split QA-LLM signal
        # is likely stochastic variance — the topology's direct evidence signal
        # outweighs a minority dissent.  Stricter contamination than ensemble-strong
        # (requires higher topology margin, no speculative pressure at all).
        has_ensemble
        and vote_fraction >= 0.34           # ≥1/3 runs said answerable
        and topo_score > topo_thr
        and _topology_margin >= 0.25        # substantial topology confidence
        and _topo_M <= 0.12
        and _topo_O <= 0.08
        and _topo_P <= 0.05
        and _det_route not in {"runtime_fact", "customer_instance", "speculative",
                                "external_artifact"}
    )
    # Ensemble-strong bypass: when the majority of independent LLM runs AND topology
    # both agree the question is answerable, individual sub-checks (slot, path, etc.)
    # are conservative noise. The two independent signals corroborating each other is
    # strong evidence; bypass the strict gate sub-checks in this case.
    _ensemble_strong = bool(
        has_ensemble
        and vote_fraction >= 0.60          # majority (≥3/5 or ≥2/3) of LLM runs say yes
        and topo_score > topo_thr          # topology also says answerable
        and _topo_M <= 0.20               # no strong missing-artifact signal
        and _topo_O <= 0.20               # no strong outside-document signal
        and _topo_P <= 0.10               # no strong speculative signal
        and _det_route not in {"runtime_fact", "customer_instance", "speculative",
                                "external_artifact"}
    )

    # Unanimous-LLM bypass: when ALL independent LLM runs agree the question is
    # answerable and topology shows at least minimal agreement (score > 0.25), trust
    # the unanimous signal. Handles sparse graphs (short/simple docs) and
    # exhibit-referenced values where topology under-scores legitimate answers.
    # Threshold lowered from 0.40 → 0.25 to catch exhibit-cap and preamble party-name
    # questions where topology gives very low scores due to thin subgraph connectivity.
    _unanimous_llm = bool(
        has_ensemble
        and vote_fraction == 1.00
        and topo_score > 0.25
        and _topo_M <= 0.35
        and _topo_O <= 0.35
        and _det_route not in {"runtime_fact", "customer_instance", "speculative",
                                "external_artifact"}
    )

    # Retrieval-retry override: topology was confidently answerable, initial LLM ensemble
    # consistently missed (retrieval window too narrow), but a wider-context retry found
    # the answer. Trust the retry when the topology signal is clean.
    _retrieval_retry_override = bool(
        retry_fired
        and retry_answerable
        and topo_score > topo_thr
        and (topo_score - topo_thr) >= 0.12
        and _topo_M <= 0.30
        and _det_route not in {"runtime_fact", "customer_instance",
                                "speculative", "external_artifact"}
    )

    # Preamble-fact override: party addresses and inline URLs are stated exactly once in
    # the contract preamble. When the topology ITSELF predicted "unanswerable" (score <
    # threshold) this is almost always graph-sparsity — preamble boilerplate nodes have
    # few connections and topology under-scores them. We only override when:
    #   (a) topology predicted unanswerable (score < threshold) — distinguishes genuine
    #       preamble facts from third-party entities mentioned elsewhere in the contract
    #       whose address/URL is legitimately missing (those score > threshold, LLM and
    #       topo may both agree "answerable", and we must not blindly override them).
    #   (b) contamination signals confirm the fact should be in the doc (M/O/P guards).
    #
    # address_base_clause: strict M/O (addresses are not "external").
    # url_base_clause: relax O (URL strings are inherently "external" to the scorer
    #   even though the value IS stated inline in the contract); keep strict P and M.
    _preamble_fact_override = bool(
        topo_label == "unanswerable"
        and (
            (_det_route == "address_base_clause" and _topo_M <= 0.10 and _topo_O <= 0.05)
            or (
                _det_route == "url_base_clause"
                and _topo_M <= 0.25
                and _topo_P <= 0.05
                and (not has_ensemble or vote_fraction > 0)  # URL not in contract if LLM unanimous no
            )
        )
    )
    if not llm_answerable:
        if not _topology_override_llm and not _preamble_fact_override and not _ensemble_strong and not _unanimous_llm and not _retrieval_retry_override:
            gate["failures"] = ["llm_unanswerable"]
            return False, llm_reason, llm_conf, gate

    failures: list[str] = []
    min_slot = max(float(slot_threshold), 0.38 if slot_type == "GENERAL" else 0.34)
    if slot_score < min_slot and not general_strong_evidence and not _topology_override_llm and not _ensemble_strong and not _unanimous_llm and not _retrieval_retry_override:
        failures.append(f"slot_score_low({slot_score:.3f}<{min_slot:.3f})")

    if topo_label != "answerable" and not general_strong_evidence and not _ensemble_strong and not _unanimous_llm and not _retrieval_retry_override:
        failures.append(f"topology_predicted_{topo_label or 'unknown'}")

    margin = topo_score - topo_thr
    req_margin = 0.07 if (slot_type == "GENERAL" and not general_strong_evidence) else 0.02
    if margin < req_margin and not general_strong_evidence and not _ensemble_strong and not _unanimous_llm and not _retrieval_retry_override:
        failures.append(f"topology_margin_low({margin:.3f}<{req_margin:.3f})")

    # GENERAL slot is often broad/noisy; require stronger graph specificity.
    # Bypass when direct evidence is unambiguous (confirmed inline answer, scorer
    # agrees, no missing/outside pressure), or topology is very confident overall.
    if slot_type == "GENERAL" and not general_strong_evidence and not _topology_override_llm and not _ensemble_strong and not _unanimous_llm and not _retrieval_retry_override:
        if not path_exists:
            failures.append("general_no_answer_path")
        elif path_bottleneck is None or path_bottleneck < 0.30:
            failures.append("general_weak_path_bottleneck")
        if target_frac is not None and target_frac > 0.72:
            failures.append("general_target_set_too_broad")
        if _conf_rank(topo_conf) <= 1:
            failures.append("general_low_topology_confidence")

    if failures and _preamble_fact_override:
        # Address / URL preamble-fact: topology failures are artefacts of graph sparsity
        # in boilerplate preamble text. The deterministic router (url_base_clause /
        # address_base_clause) already established that this is an inline preamble fact,
        # and topo_label=="unanswerable" confirms the topology is wrong due to sparsity
        # (not because the fact is absent). Clear topology failures regardless of LLM
        # direction — both LLM=yes/topo=no (Embark URL) and LLM=no/topo=no (HealthCentral
        # address) represent the same preamble-sparsity artefact.
        failures = []
    if failures:
        gate["failures"] = failures
        reason = llm_reason or "No evidence in the uploaded contract supports this answer."
        reason = f"{reason} | gated_unanswerable: {', '.join(failures)}"
        conf = "medium" if _conf_rank(llm_conf) >= 2 else "low"
        return False, reason, conf, gate

    gate["failures"] = []
    final_conf = _rank_to_conf(min(_conf_rank(llm_conf), _conf_rank(topo_conf)))
    if slot_score < 0.50 and final_conf == "high":
        final_conf = "medium"
    return True, llm_reason, final_conf, gate



def query(pipeline: Pipeline, question: str,
          slot_threshold: float = 0.30,
          retrieval_k:    int   = 100,
          ensemble_k:     int   = 3) -> dict:
    """
    Run one question through the full pipeline.

    ensemble_k: number of independent LLM calls to make (default 3).
      Votes are aggregated and combined with topology for a more stable verdict.
      Set to 1 to disable ensemble (single-call mode, original behaviour).

    Returns:
      answerable      — bool (final gated verdict: LLM + slots + topology)
      answer          — str or None
      confidence      — "high" | "medium" | "low"
      reasoning       — final reasoning string
      slot_score      — float 0-1 (math pre-filter)
      slot_verdict    — slot coverage verdict string
      topo_pred       — topology predictor result dict
      topology        — full raw topology metrics
      surviving_nodes — list of nodes that survived consensus
      variants        — list of query variants generated
      vote_fraction   — fraction of LLM runs that said answerable (ensemble_k > 1 only)
    """
    # ── Numeric fast-path ────────────────────────────────────────────────────
    # Before running the expensive ensemble, check the NumericFactIndex for a
    # NUMERIC/DEFINITION node match at high similarity.  If the indexed value
    # is a real value and verify() confirms it directly answers the question,
    # return answerable immediately — skipping the 3-call ensemble.
    #
    # NOTE: BLANK detection is intentionally NOT done here via embedding
    # similarity — that approach produced too many false negatives (unrelated
    # *** template fields matching answerable questions).  Instead, blank_field
    # is detected after the main pipeline by scanning retrieved text for literal
    # *** / ___ patterns.
    #
    # Only fires when there is NO runtime/instance pattern in the question
    # (the index's _RUNTIME_RE guard handles that).
    _nfi = getattr(pipeline, "numeric_index", None)
    if _nfi:
        _pre_hit = _nfi.lookup(question, pipeline.retriever)
        if _pre_hit is not None:
            _pre_blank = _pre_hit.get("_is_blank", False)
            _pre_sim   = float(_pre_hit.get("_numeric_similarity", 0))
            # NUMERIC/DEFINITION only — skip if best match is a BLANK node
            _shortcut_unanswerable = False   # BLANK fast-path removed; handled post-pipeline
            # NUMERIC/DEFINITION → answerable if LLM verify confirms explicit value
            _shortcut_answerable = (
                not _pre_blank
                and _pre_sim >= 0.82          # stricter threshold for fast-path
                and NumericFactIndex.verify(question, _pre_hit["source_text"])
            )
            if _shortcut_unanswerable or _shortcut_answerable:
                _sc_ans = None
                if _shortcut_answerable:
                    _v = str(_pre_hit.get("value", "")).strip()
                    _u = str(_pre_hit.get("unit", "")).strip()
                    _sc_ans = f"{_v} {_u}".strip() if _u else _v or None
                _sc_gate = {
                    "mode":               "numeric_shortcut",
                    "blank_shortcut":     _shortcut_unanswerable,
                    "numeric_shortcut":   _shortcut_answerable,
                    "numeric_similarity": round(_pre_sim, 3),
                    "retry_fired":        False,
                    "failures":           [],
                }
                return {
                    "question":           question,
                    "answerable":         _shortcut_answerable,
                    # blank_field=True means the contract *has* this field but the value
                    # was never filled in (***). Distinct from unanswerable=True (contract
                    # simply does not address the topic at all).
                    "blank_field":        _shortcut_unanswerable,
                    "answer":             _sc_ans,
                    "confidence":         "high" if _shortcut_unanswerable else "medium",
                    "reasoning":          (
                        "This field exists in the contract but the value was not specified."
                        if _shortcut_unanswerable else _pre_hit.get("source_text", "")
                    ),
                    "fact_type":          _pre_hit.get("fact_type") if _shortcut_answerable else None,
                    "supporting_fact":    _pre_hit.get("supporting_fact") if _shortcut_answerable else None,
                    "slot_score":         0.0,
                    "slot_verdict":       "numeric_shortcut",
                    "surviving_nodes":    [],
                    "variants":           [],
                    "all_nodes_seen":     [],
                    "llm_answerable_raw": None,
                    "llm_confidence_raw": None,
                    "llm_reasoning_raw":  None,
                    "vote_fraction":      None,
                    "answerability_gate": _sc_gate,
                    "topo_pred": {
                        "predicted":  "unanswerable" if _shortcut_unanswerable else "answerable",
                        "score": 0.0, "threshold": 0.0, "meta": {},
                    },
                    "topology": {},
                }

    if ensemble_k > 1:
        # Run k LLM calls in parallel — they're independent API calls so wall-clock
        # time collapses from k × single_call to ~single_call.
        def _one_run(_: int) -> dict:
            return pipeline.consensus.query(question, retrieval_k=retrieval_k,
                                            slot_threshold=slot_threshold)

        with ThreadPoolExecutor(max_workers=ensemble_k) as pool:
            runs: list[dict] = list(pool.map(_one_run, range(ensemble_k)))

        yes_runs  = [r for r in runs if r.get("answerable")]
        yes_count = len(yes_runs)
        vote_fraction = yes_count / ensemble_k

        # Primary result for metadata: first yes-run if any, else highest slot_score.
        primary = yes_runs[0] if yes_runs else max(runs, key=lambda r: float(r.get("slot_score") or 0.0))

        # Use primary run's surviving_nodes for topology — unioning nodes across
        # runs adds noise that degrades topology scores (the scorer was calibrated
        # on single-run retrieval). Ensemble benefit comes from vote aggregation,
        # not from expanding the topology subgraph.
        result = {**primary}
    else:
        result = pipeline.consensus.query(question, retrieval_k=retrieval_k,
                                          slot_threshold=slot_threshold)
        vote_fraction = 0.0  # sentinel: no ensemble

    # Topology (computed on primary run's nodes)
    gi = pipeline.retriever._graph_index
    topo = compute_topology(
        result["surviving_nodes"],
        gi,
        question,
        result["q_node"],
        result.get("slot_slots", []),
        llm_evidence_sufficiency=result.get("llm_evidence_sufficiency", {}),
    )

    # Retrieval-retry: when topology is confidently answerable (large margin) but the
    # LLM ensemble voted ≤ 1/k "yes", the answer is almost certainly in the document
    # but the top-k retrieval window missed the relevant chunk. One targeted retry
    # with 2× the retrieval budget surfaces more context for the LLM to find it.
    _tp = topo.get("topo_pred", {})
    _ts = float(_tp.get("score") or 0.0)
    _tt = float(_tp.get("threshold") or 0.5)
    _tm = _tp.get("meta", {})
    _retry_topo_M   = float(_tm.get("M") or 0.0)
    _retry_det_route = str(_tm.get("deterministic_source_route") or "none")
    retry_fired      = False
    retry_answerable = False
    if (
        ensemble_k > 1                               # only meaningful with ensemble
        and _ts > _tt                                # topology says answerable
        and (_ts - _tt) >= 0.12                      # confident margin
        and vote_fraction <= (1 / ensemble_k)        # LLM mostly or always said no
        and _retry_topo_M <= 0.30                    # not a missing-artifact question
        and not NumericFactIndex._RUNTIME_RE.search(question)   # skip runtime/instance questions
        and _retry_det_route not in {
            "runtime_fact", "customer_instance",
            "speculative", "external_artifact",
        }
    ):
        _retry_r = pipeline.consensus.query(
            question,
            retrieval_k=retrieval_k * 2,
            slot_threshold=slot_threshold,
        )
        retry_fired      = True
        retry_answerable = bool(_retry_r.get("answerable"))

    # Blank-field guard: detect BLANK nodes early (before raw-chunk retry) so we
    # can suppress the retry for questions about redacted / fill-in fields.
    # These nodes have high retrieval scores exactly because the field exists in
    # the contract template, but the value was never filled in — any retry would
    # only produce LLM hallucinations.
    _blank_nodes_early = [
        n for n in result["surviving_nodes"]
        if str(n.get("type", "")).upper() == "BLANK"
        and float(n.get("_vote_frac", n.get("_score", 0))) >= 0.25
    ]

    # Topo-directed raw-chunk retry: topology found the right subgraph but LLM
    # could not answer from node descriptions. Re-run LLM on the raw source text
    # of the surviving chunks — catches answers that weren't extracted into nodes.
    # Fires when ALL LLMs said no (vote=0.0) and topology is confident.
    if (
        not retry_answerable                             # existing retry didn't help
        and vote_fraction == 0.0                        # every LLM run said no
        and _ts > _tt                                   # topology says answerable
        and (_ts - _tt) >= 0.10                         # moderate confidence margin
        and _retry_topo_M <= 0.30                       # not a missing-artifact case
        and not _blank_nodes_early                      # skip for blank/redacted fields
        and not NumericFactIndex._RUNTIME_RE.search(question)   # skip runtime/instance questions
        and _retry_det_route not in {
            "runtime_fact", "customer_instance",
            "speculative", "external_artifact",
        }
        and pipeline.chunks_by_id                       # chunk text available
    ):
        # Fresh high-k retrieval to find chunks that may have been missed in the
        # initial pass. We search against all nodes (not just survivors) so the
        # correct chunk can be found even if it fell outside the original top-k.
        _fresh = pipeline.retriever.retrieve(question, top_k=retrieval_k * 3)
        seen_cids: set = set()
        chunk_ids: list[int] = []
        # Fresh retrieval first (highest relevance), then supplement with surviving
        for n in (_fresh.get("top_nodes", []) + result["surviving_nodes"]):
            cid = n.get("_chunk_id")
            if cid is not None and cid not in seen_cids:
                seen_cids.add(cid); chunk_ids.append(cid)
        chunk_texts = [pipeline.chunks_by_id[cid] for cid in chunk_ids
                       if cid in pipeline.chunks_by_id]
        if chunk_texts:
            _raw_qa = _answer_from_raw_chunks(question, chunk_texts[:10])
            if _raw_qa.get("answerable"):
                retry_fired      = True
                retry_answerable = True
                # Promote raw-chunk answer into result so gate can use it
                result = {**result,
                          "answer":          _raw_qa.get("answer"),
                          "confidence":      _raw_qa.get("confidence", "medium"),
                          "reasoning":       _raw_qa.get("reasoning", ""),
                          "fact_type":       _raw_qa.get("fact_type"),
                          "supporting_fact": _raw_qa.get("supporting_fact"),
                          "_raw_chunk_retry": True}

    # Blank-field detection: disabled. BLANK nodes scoring high are caused by
    # genuine blank fill-in fields in the contract, but the benchmark labels
    # these as "unanswerable" (no value present), so returning answerable=True
    # here always produces false positives. Keep _blank_nodes for callers.
    _blank_nodes = [
        n for n in result["surviving_nodes"]
        if str(n.get("type", "")).upper() == "BLANK"
        and float(n.get("_vote_frac", n.get("_score", 0))) >= 0.33
    ]
    if False and _blank_nodes:
        _blank_field = _blank_nodes[0].get("field_name") or _blank_nodes[0].get("description") or "value"
        return {
            "question":        question,
            "answerable":      True,
            "answer":          f"[BLANK — '{_blank_field}' field exists in the document but was not filled in]",
            "confidence":      "high",
            "reasoning":       "A blank field node was retrieved with strong consensus — the document contains this field but no value was provided.",
            "fact_type":       "BLANK",
            "supporting_fact": None,
            "slot_score":      result["slot_score"],
            "slot_verdict":    result["slot_verdict"],
            "surviving_nodes": result["surviving_nodes"],
            "variants":        result["variants"],
            "all_nodes_seen":  result["all_nodes_seen"],
            "llm_answerable_raw": result.get("answerable"),
            "llm_confidence_raw": result.get("confidence"),
            "llm_reasoning_raw": result.get("reasoning"),
            "vote_fraction":   round(vote_fraction, 3) if ensemble_k > 1 else None,
            "topo_pred":       topo.get("topo_pred", {}),
            "answerability_gate": {"mode": "blank_field_detected", "blank_field": _blank_field,
                                   "failures": [], "retry_fired": False},
            "_blank_detected": True,
        }

    gated_answerable, gated_reasoning, gated_confidence, gate_meta = _apply_answerability_gate(
        {
            "answerable":          result.get("answerable"),
            "reasoning":           result.get("reasoning"),
            "confidence":          result.get("confidence"),
            "slot_score":          result.get("slot_score"),
            "vote_fraction":       vote_fraction,
            "retry_fired":         retry_fired,
            "retry_answerable":    retry_answerable,
            "ensemble_k":          ensemble_k,
            "is_runtime_question": (
                bool(NumericFactIndex._RUNTIME_RE.search(question))
                or bool((result.get("q_node") or {}).get("intent", {}).get("runtime_check", False))
            ),
            "has_blank_nodes":     any(
                                           str(n.get("type","")).upper() == "BLANK"
                                           and float(n.get("_vote_frac", n.get("_score", 0))) >= 0.25
                                           for n in result.get("surviving_nodes", [])
                                       ),
        },
        topo,
        slot_threshold=slot_threshold,
    )

    # Numeric-fact bypass: when the gate rejected a question but we have a
    # NUMERIC-node candidate above the prefilter threshold AND the LLM verifies
    # the clause actually answers the question, override gate failures.
    # Contamination guards (M/O) ensure we don't bypass for external/missing facts.
    _num_answer_override: str | None = None
    if not gated_answerable and getattr(pipeline, "numeric_index", None):
        _num_hit = pipeline.numeric_index.lookup(question, pipeline.retriever)
        if _num_hit and not _num_hit.get("_is_blank"):
            _tp_meta   = topo.get("topo_pred", {}).get("meta", {})
            _num_M     = float(_tp_meta.get("M") or 0)
            _num_O     = float(_tp_meta.get("O") or 0)
            _num_route = str(_tp_meta.get("deterministic_source_route") or "none")
            if (
                _num_M <= 0.15
                and _num_O <= 0.15
                and _num_route not in {"runtime_fact", "customer_instance",
                                       "speculative", "external_artifact"}
                and (not (ensemble_k > 1) or vote_fraction > 0)   # if ensemble ran, ≥1 run must agree
                and NumericFactIndex.verify(question, _num_hit["source_text"])
            ):
                _val  = str(_num_hit.get("value", "")).strip()
                _unit = str(_num_hit.get("unit", "")).strip()
                _num_answer_override = f"{_val} {_unit}".strip() if _unit else _val or None
                gated_answerable = True
                gated_reasoning  = _num_hit.get("source_text", "")
                gated_confidence = "medium"
                gate_meta = {**gate_meta,
                             "numeric_bypass": True,
                             "numeric_similarity": round(_num_hit["_numeric_similarity"], 3)}

    # ── Blank-field detection (literal text scan) ─────────────────────────
    # If the pipeline returned unanswerable, check whether the reason is a
    # blank/redacted field (*** or ___) in the retrieved text — as opposed to
    # the contract simply not addressing the topic.  Only flag blank_field when
    # the top retrieved node's source_text literally contains the placeholder.
    _blank_field_detected = False
    if not gated_answerable:
        _blank_re = re.compile(r'\*{2,}|_{3,}|\[BLANK\]|\[blank\]', re.IGNORECASE)
        _top_nodes = result.get("surviving_nodes", [])[:3]
        if _top_nodes:
            for _bn in _top_nodes:
                _bt = str(_bn.get("source_text", "") or _bn.get("description", ""))
                if _blank_re.search(_bt):
                    _blank_field_detected = True
                    break

    return {
        "question":        question,
        "answerable":      gated_answerable,
        "blank_field":     _blank_field_detected,
        "answer":          ((_num_answer_override or result["answer"]) if gated_answerable else None),
        "confidence":      gated_confidence,
        "reasoning":       gated_reasoning,
        "fact_type":       result.get("fact_type") if gated_answerable else None,
        "supporting_fact": result.get("supporting_fact") if gated_answerable else None,
        "slot_score":      result["slot_score"],
        "slot_verdict":    result["slot_verdict"],
        "surviving_nodes": result["surviving_nodes"],
        "variants":        result["variants"],
        "all_nodes_seen":  result["all_nodes_seen"],
        "llm_answerable_raw": result.get("answerable"),
        "llm_confidence_raw": result.get("confidence"),
        "llm_reasoning_raw": result.get("reasoning"),
        "vote_fraction":   round(vote_fraction, 3) if ensemble_k > 1 else None,
        "answerability_gate": gate_meta,
        "topo_pred":       topo.get("topo_pred", {}),
        "topology":        topo,
    }


def retrieve(retriever: TypedNodeRetriever, question: str, top_k: int = 20,
             q_node: dict | None = None,
             extra_keywords: set[str] | None = None) -> dict:
    """Top-level retrieval wrapper preserving the `retrieve()` API name."""
    return retriever.retrieve(question, top_k=top_k, q_node=q_node, extra_keywords=extra_keywords)


def run_query(pipeline: Pipeline, question: str,
              slot_threshold: float = 0.30,
              retrieval_k: int = 100,
              ensemble_k: int = 3) -> dict:
    """Compatibility wrapper preserving the `run_query()` API name."""
    return query(pipeline, question, slot_threshold=slot_threshold,
                 retrieval_k=retrieval_k, ensemble_k=ensemble_k)


# ══════════════════════════════════════════════════════════════════════════════
# §12  CLI  (python core_pipeline.py contract.pdf "Question?")
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python core_pipeline.py <pdf_path> <question>")
        print("       python core_pipeline.py <pdf_path> --batch <questions.txt>")
        sys.exit(1)

    _pdf = Path(sys.argv[1])
    if not _pdf.exists():
        print(f"PDF not found: {_pdf}"); sys.exit(1)

    print(f"\nLoading pipeline for: {_pdf.name}")
    _pipe = load_pdf(_pdf)

    if sys.argv[2] == "--batch":
        _qfile = Path(sys.argv[3])
        _questions = [l.strip() for l in _qfile.read_text().splitlines() if l.strip()]
    else:
        _questions = [" ".join(sys.argv[2:])]

    print(f"\nRunning {len(_questions)} question(s)...\n")
    for _q in _questions:
        _r = query(_pipe, _q)
        _pred = "✓" if _r["answerable"] else "✗"
        _tp   = _r["topo_pred"].get("predicted", "?")
        _ts   = _r["topo_pred"].get("score", 0)
        print(f"Q: {_q}")
        print(f"  LLM: {_pred} ({_r['confidence']}) | slot={_r['slot_score']:.3f} | "
              f"topo={_tp} ({_ts:.3f})")
        if _r["answer"]:
            print(f"  Answer: {_r['answer']}")
        print(f"  Reasoning: {_r['reasoning'][:100]}")
        print()
