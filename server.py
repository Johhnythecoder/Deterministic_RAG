from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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


cache = PipelineCache()
app = FastAPI(title="Deterministic RAG Demo", version="1.0.0")

app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

_prep_jobs: dict[str, dict[str, Any]] = {}
_job_lock = Lock()
_doc_tokens: dict[str, Path] = {}
_prepared_index: list[dict[str, Any]] = []
_telemetry_lock = Lock()


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", name).strip("._")
    return cleaned or "upload.pdf"


def _collect_demo_docs() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for p in sorted(SAMPLE_DIR.glob("*.pdf")):
        docs.append({"name": p.name, "path": str(p)})
    return docs


def _load_prepared_index() -> None:
    global _prepared_index
    if not PREPARED_INDEX_PATH.exists():
        _prepared_index = []
        return
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


def _estimate_prep_seconds(pdf_path: Path) -> int:
    """Heuristic ETA for first pipeline load; cached docs are much faster."""
    if cache.has(pdf_path):
        return 5
    try:
        n_pages = len(PdfReader(str(pdf_path)).pages)
    except Exception:
        n_pages = 12
    # Baseline + per-page cost for extraction, embeddings, graph build.
    return max(12, min(240, int(10 + 2.6 * max(1, n_pages))))


def _llm_preflight_error() -> str | None:
    """Return error string when no viable LLM backend is configured."""
    has_openai = bool(os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"))
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
        "No OpenAI/Azure key found and Ollama is unreachable. "
        "Set OPENAI_API_KEY (or Azure vars) in .env, or run Ollama locally."
    )


def _pick_supporting_clauses(result: dict, limit: int = 5) -> list[dict[str, str]]:
    clauses: list[dict[str, str]] = []
    seen: set[str] = set()

    for node in result.get("surviving_nodes", [])[:30]:
        text = (node.get("source_text") or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        source = node.get("_section") or f"Chunk {node.get('_chunk_id', '?')}"
        clauses.append({"text": text, "source": str(source)})
        if len(clauses) >= limit:
            break

    return clauses


def _shape_response(result: dict) -> dict:
    answerable = bool(result.get("answerable", False))
    short_answer = result.get("answer") if answerable else None
    reason = (result.get("reasoning") or "").strip()

    if not reason:
        if answerable:
            reason = "The contract includes evidence supporting this answer."
        else:
            reason = "No evidence in the uploaded contract supports this answer."

    topo = result.get("topo_pred") or {}
    metadata: dict[str, Any] = {
        "confidence": result.get("confidence"),
    }
    gate = result.get("answerability_gate") or {}
    if gate:
        metadata["gate_mode"] = gate.get("mode")
        metadata["gate_failures"] = gate.get("failures")
        metadata["llm_answerable_raw"] = result.get("llm_answerable_raw")
        metadata["llm_confidence_raw"] = result.get("llm_confidence_raw")
    if topo.get("score") is not None:
        metadata["topology_score"] = topo.get("score")
        metadata["topology_predicted"] = topo.get("predicted")
        metadata["topology_confidence"] = topo.get("confidence")
        metadata["topology_threshold"] = topo.get("threshold")

    return {
        "answerable": answerable,
        "short_answer": short_answer,
        "reason": reason,
        "supporting_clauses": _pick_supporting_clauses(result),
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
            }
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
    except Exception as exc:
        with _job_lock:
            _prep_jobs[job_id]["status"] = "error"
            _prep_jobs[job_id]["error"] = str(exc)
            _prep_jobs[job_id]["finished_at"] = time.time()
            _prep_jobs[job_id].setdefault("metrics", {})["llm_calls_actual"] = get_llm_call_count()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/demo-docs")
def demo_docs() -> dict:
    return {"documents": _collect_demo_docs()}


@app.get("/api/saved-docs")
def saved_docs() -> dict:
    return {"documents": _list_saved_docs()}


@app.get("/api/topology-telemetry")
def topology_telemetry(limit: int = 100) -> dict:
    lim = max(1, min(1000, int(limit)))
    events = _read_recent_topology_events(lim)
    return {
        "events": events,
        "count": len(events),
        "path": str(TOPOLOGY_TELEMETRY_PATH),
    }


@app.post("/api/prepare-doc")
def prepare_doc(
    sample_name: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
) -> dict:
    preflight_err = _llm_preflight_error()
    if preflight_err:
        raise HTTPException(status_code=400, detail=preflight_err)

    pdf_path = _resolve_pdf(sample_name=sample_name, file=file)
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
def prepare_doc_status(job_id: str) -> dict:
    with _job_lock:
        job = _prep_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        payload = dict(job)

    status = payload["status"]
    estimate = max(1, int(payload.get("estimate_sec") or 30))
    started = payload.get("started_at")
    elapsed = 0.0
    if started:
        elapsed = max(0.0, time.time() - float(started))
    elif payload.get("created_at"):
        elapsed = max(0.0, time.time() - float(payload["created_at"]))

    if status == "queued":
        progress = 0.02
        remaining_sec: int | None = estimate
        over_eta_sec = 0
    elif status == "running":
        if elapsed <= estimate:
            # Move up to 90% during estimated window.
            progress = min(0.90, 0.05 + 0.85 * (elapsed / estimate))
            remaining_sec = max(0, estimate - int(elapsed))
            over_eta_sec = 0
        else:
            # Past ETA: keep moving slowly toward 99% to show liveness.
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

    metrics = payload.get("metrics") or {}
    total_chunks = int(metrics.get("total_chunks") or 0)
    extracted_chunks = int(metrics.get("extracted_chunks") or 0)
    if total_chunks > 0:
        payload["chunk_progress"] = round(min(1.0, extracted_chunks / total_chunks), 4)
    else:
        payload["chunk_progress"] = 0.0

    llm_calls_actual = int(metrics.get("llm_calls_actual") or 0)
    llm_calls_estimated_total = int(payload.get("estimate_sec", 0) / 8) + 3
    llm_calls_estimated_total = max(llm_calls_estimated_total, llm_calls_actual, 1)
    payload["llm_progress"] = round(min(1.0, llm_calls_actual / llm_calls_estimated_total), 4)
    payload["llm_calls_estimated_total"] = llm_calls_estimated_total
    return payload


@app.post("/api/ask")
def ask(
    question: str = Form(...),
    doc_token: str | None = Form(default=None),
) -> dict:
    t0 = time.time()
    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is required.")
    if not doc_token:
        raise HTTPException(status_code=400, detail="Please prepare a document before asking questions.")
    pdf_path = _resolve_pdf(sample_name=None, file=None, doc_token=doc_token)

    try:
        pipeline = cache.get(pdf_path)
        raw = run_query(pipeline, q)
        shaped = _shape_response(raw)
        shaped["document"] = pdf_path.name
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
        raise
    except Exception as exc:
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
    questions: str = Form(...),
    doc_token: str | None = Form(default=None),
) -> dict:
    if not doc_token:
        raise HTTPException(status_code=400, detail="Please prepare a document before asking questions.")

    parsed = [q.strip() for q in (questions or "").splitlines() if q.strip()]
    if not parsed:
        raise HTTPException(status_code=400, detail="Provide at least one question (one per line).")
    if len(parsed) > 40:
        raise HTTPException(status_code=400, detail="Maximum batch size is 40 questions.")

    pdf_path = _resolve_pdf(sample_name=None, file=None, doc_token=doc_token)

    try:
        pipeline = cache.get(pdf_path)
    except HTTPException:
        raise
    except Exception as exc:
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
    return {
        "document": pdf_path.name,
        "total": len(results),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "results": results,
    }


_load_prepared_index()
