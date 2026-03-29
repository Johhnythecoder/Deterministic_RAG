#!/usr/bin/env python3
"""Time every internal step for a single question through the pipeline."""
from __future__ import annotations
import time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core_pipeline as cp

PDF      = Path("/Users/jonathang/Downloads/EmbarkComInc_19991008_S-1A_EX-10.10_6487661_EX-10.10_Co-Branding Agreement.pdf")
QUESTION = "How are the costs and expenses of the arbitrator shared between the parties?"

# ── 1. Pipeline load ────────────────────────────────────────────────────────
print("\n── Timing breakdown ──────────────────────────────────────────────────")
t0 = time.perf_counter()
pipe = cp.load_pdf(str(PDF))
print(f"  pipeline load      : {time.perf_counter()-t0:6.2f}s")

# ── 2. Patch consensus.query to print internal step times ──────────────────
original_consensus_query = pipe.consensus.query

def timed_consensus_query(question, retrieval_k=15, slot_threshold=0.30):
    t = time.perf_counter()

    import os
    from concurrent.futures import ThreadPoolExecutor
    from core_pipeline import (generate_variants, parse_question_node,
                                answer_from_typed_nodes)

    # Step A: variants + q_node (parallel)
    ta = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fv = pool.submit(generate_variants, question, pipe.consensus.n_variants)
        fq = pool.submit(parse_question_node, question)
        variants   = fv.result()
        q_node     = fq.result()
    print(f"    [consensus] variants+qnode (parallel) : {time.perf_counter()-ta:6.2f}s")

    # Step B: retrieval for all variants (parallel)
    tb = time.perf_counter()
    result_full = original_consensus_query(question, retrieval_k=retrieval_k,
                                           slot_threshold=slot_threshold)
    t_full = time.perf_counter() - tb
    # We already have the full result; just show what retrieval+LLM took together
    # (split is hard without deeper patching — estimate below)
    print(f"    [consensus] full call (retrieval+slot+LLM) : {t_full:6.2f}s  (includes above)")
    return result_full

pipe.consensus.query = timed_consensus_query

# ── 3. Run with ensemble_k=1 (single call, original mode) ──────────────────
print(f"\n  question: {QUESTION[:80]}")
print()

print("  ── ensemble_k=1 (single call) ──")
t1 = time.perf_counter()
r1 = cp.run_query(pipe, QUESTION, ensemble_k=1)
print(f"  TOTAL single call  : {time.perf_counter()-t1:6.2f}s")
print(f"  answerable={r1['answerable']}  score={r1['topo_pred'].get('score',0):.3f}  thr={r1['topo_pred'].get('threshold',0):.3f}")

# ── 4. Now time ensemble_k=3 (parallel) ────────────────────────────────────
# Reset patch to bare original for accurate parallel timing
pipe.consensus.query = original_consensus_query

print()
print("  ── ensemble_k=3 (parallel LLM calls) ──")
t2 = time.perf_counter()
r3 = cp.run_query(pipe, QUESTION, ensemble_k=3)
t_ensemble = time.perf_counter() - t2
print(f"  TOTAL ensemble_k=3 : {t_ensemble:6.2f}s")
print(f"  answerable={r3['answerable']}  vote={r3.get('vote_fraction'):.2f}  score={r3['topo_pred'].get('score',0):.3f}  thr={r3['topo_pred'].get('threshold',0):.3f}")

gate = r3.get("answerability_gate", {})
print(f"  retry_fired={gate.get('retry_fired')}  retry_answerable={gate.get('retry_answerable')}")
