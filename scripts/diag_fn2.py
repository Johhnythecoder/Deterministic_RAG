#!/usr/bin/env python3
"""
Diagnostic: run idx=2 and idx=16 through the live pipeline and dump full meta.

Usage:
    TOPO_LLM_INTENT_MODE=llm_only CORE_PRESTEP_PARALLEL=1 \
        python scripts/diag_fn2.py
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core_pipeline as cp

PDF = ROOT / "uploads" / "1774488921_openai-services-agreement.pdf"
OUT = ROOT / "scripts" / "diag_fn2_result.json"

CASES = [
    {"idx": 2,  "question": "Where does the agreement say the Services Term is listed?",           "gold": "answerable"},
    {"idx": 16, "question": "Which arbitration provider is named for commencing binding arbitration?", "gold": "answerable"},
]

META_KEYS = [
    # Intent routing
    "proof_type_initial", "proof_type_routed", "lane", "answer_form",
    "answer_source_type", "source_sufficiency_required", "missing_if_not_present",
    # External / missing dependency
    "external_reference_intent", "external_doc_dependency_prior",
    "external_artifact_dependency", "external_doc_dependency_refined",
    "asks_for_external_contents", "missing_artifact_intent",
    "external_ref_only_ratio", "external_inline_substance_ratio",
    "external_doc_match_ratio",
    # Customer instance
    "customer_instance_dependency", "customer_specific_dependency_prior",
    "customer_specific_dependency_refined", "instance_missing_field",
    # Speculative / outside
    "speculative_intent", "outside_document_intent", "outside_doc_needed",
    # Q2 compound penalty
    "Q2", "Q2_extra" if "Q2_extra" in dir() else "Q2",  # logged in meta? check below
    # Primary features
    "T", "A", "D", "S", "S_eff", "P", "O", "Q", "M", "X", "G", "B",
    "score", "threshold",
    # Gate/escape
    "gate_reason", "hard_gate_reasons", "hard_gate_triggered",
    "escape_hatch_triggered", "escape_hatch_eligible",
    "direct_rule_clause_override",
    # Slot/path
    "slot_type", "path_exists", "path_bottleneck", "n_targets",
    "slot_fill_ratio", "direct_fill_ratio", "exact_intent", "simple_extract_intent",
    # Evidence
    "llm_inline_answer_present", "llm_external_reference_only",
    "llm_customer_instance_missing", "llm_runtime_missing",
    "evidence_score", "risk_score", "disagreement",
    # LLM evidence sufficiency raw
    "llm_evidence_sufficiency",
    # Question intent raw (full dict)
    "question_intent",
    # Variant consensus
    "mean_vote_frac", "p75_vote_frac", "high_vote_frac", "n_vote_ge2", "n_vote_ge3",
    # Slot fill flags
    "exact_slot_heavy", "weak_slot_channels", "weak_variant_consensus", "slot_variant_reject",
    # Relief flags
    "soft_exactness_escape", "no_answer_path_escape",
    # Dependency
    "expects_direct_rule_clause", "direct_support_source",
]

def extract_meta(topo_pred: dict) -> dict:
    meta = topo_pred.get("meta", {})
    out = {}
    # Grab everything in meta (full dict)
    out["_meta_full"] = meta
    # Also grab top-level topo_pred fields
    for k in ("predicted", "score", "threshold", "confidence"):
        out[k] = topo_pred.get(k)
    out["signals"] = topo_pred.get("signals", {})
    return out

def main():
    print(f"\nLoading pipeline from: {PDF.name}")
    t0 = time.time()
    pipe = cp.load_pdf(str(PDF))
    print(f"Pipeline ready in {time.time()-t0:.1f}s\n")

    results = []
    for case in CASES:
        idx = case["idx"]
        q   = case["question"]
        gold = case["gold"]
        print(f"Running idx={idx}: {q}")
        t1 = time.time()
        r = cp.run_query(pipe, q)
        elapsed = time.time() - t1

        topo_pred = r.get("topo_pred", {})
        meta = topo_pred.get("meta", {})

        # Pull Q2_extra from meta if present (it may be logged there), else compute
        q2 = meta.get("Q2")
        # Q2_extra may not be separately logged; derive from signals if needed
        p_val = meta.get("P", 0.0) or 0.0
        o_val = meta.get("O", 0.0) or 0.0
        inst  = meta.get("instance_missing_field", 0.0) or 0.0
        extri = meta.get("external_reference_intent", 0.0) or 0.0
        q2_computed = max(p_val, o_val, inst, extri)
        q2_extra_computed = max(0.0, q2_computed - max(p_val, o_val))

        entry = {
            "idx":       idx,
            "question":  q,
            "gold":      gold,
            "predicted": r.get("answerable"),
            "elapsed_s": round(elapsed, 2),
            "topo_score":     topo_pred.get("score"),
            "topo_threshold": topo_pred.get("threshold"),
            "topo_predicted": topo_pred.get("predicted"),
            "topo_confidence": topo_pred.get("confidence"),
            "gate": r.get("answerability_gate", {}),
            "signals": topo_pred.get("signals", {}),
            "meta": meta,
            # Derived helpers
            "_derived": {
                "Q2_computed":       round(q2_computed, 4),
                "Q2_extra_computed": round(q2_extra_computed, 4),
                "margin":            round((topo_pred.get("score") or 0.0) - (topo_pred.get("threshold") or 0.5), 4),
                "correct":           (r.get("answerable") == (gold == "answerable")),
            },
        }
        results.append(entry)

        # Print summary to console
        margin = entry["_derived"]["margin"]
        correct_str = "CORRECT" if entry["_derived"]["correct"] else "WRONG"
        print(f"  [{correct_str}] predicted={r.get('answerable')} | score={topo_pred.get('score')} threshold={topo_pred.get('threshold')} margin={margin:+.4f}")
        print(f"  Q2={q2_computed:.4f} Q2_extra={q2_extra_computed:.4f}")
        print(f"  P={p_val:.4f} O={o_val:.4f} instance_missing={inst:.4f} external_ref={extri:.4f}")
        print(f"  proof_type_routed={meta.get('proof_type_routed')} lane={meta.get('lane')} answer_form={meta.get('answer_form')}")
        print(f"  llm_inline_answer_present={meta.get('llm_inline_answer_present')} sf={meta.get('slot_fill_ratio')} Q={meta.get('Q')}")
        print(f"  gate_reason={meta.get('gate_reason')} hard_gates={meta.get('hard_gate_reasons')}")
        print()

    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"Full results → {OUT}")

if __name__ == "__main__":
    main()
