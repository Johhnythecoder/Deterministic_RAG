#!/usr/bin/env python3
"""Run 40-question benchmark against the Affiliate Agreement PDF.

Usage:
    TOPO_LLM_INTENT_MODE=llm_only CORE_PRESTEP_PARALLEL=1 \
        python scripts/run_eval_affiliate.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core_pipeline as cp

PDF = Path("/Users/jonathang/Downloads/UsioInc_20040428_SB-2_EX-10.11_1723988_EX-10.11_Affiliate Agreement 2.pdf")
OUT = ROOT / "scripts" / "affiliate_eval_result.json"

QUESTIONS = [
    # ── ANSWERABLE (gold = "answerable") ─────────────────────────────────────
    # Parties & identity
    (1,  "What state is Network 1 Financial, Inc. incorporated in?",                            "answerable"),
    (2,  "What is Network 1's principal place of business address?",                            "answerable"),
    (3,  "What state is the Affiliate, Payment Data Systems, Inc., incorporated in?",           "answerable"),
    (4,  "What payment card networks does Network 1 process for?",                              "answerable"),
    # Term & termination
    (5,  "What is the initial term of this agreement in days?",                                 "answerable"),
    (6,  "How many days written notice must a party give to prevent automatic renewal?",        "answerable"),
    (7,  "Under what circumstance does the initial term become 3 years instead of 180 days?",   "answerable"),
    (8,  "Can this agreement be terminated for default, and how many days must the defaulting party be given to cure?", "answerable"),
    # Exit fee
    (9,  "What is the maximum dollar cap on the Exit Fee?",                                     "answerable"),
    (10, "How many months of net recurring revenue is used to calculate the Exit Fee per merchant?", "answerable"),
    (11, "When must the Exit Fee be paid relative to the transfer of merchants?",               "answerable"),
    # Compensation
    (12, "What does Affiliate receive as compensation for its services?",                       "answerable"),
    (13, "Who pays the Affiliate's Fee — Network 1 or an affiliate of Network 1?",             "answerable"),
    # Merchant agreements
    (14, "Does Network 1 have the right to approve or decline Merchant Agreements submitted by the Affiliate?", "answerable"),
    (15, "What must the Affiliate do with all Merchant Agreements it procures?",                "answerable"),
    # Assignment
    (16, "Can Network 1 assign this agreement without the Affiliate's prior written consent?",  "answerable"),
    (17, "Can the Affiliate assign this agreement without Network 1's prior written consent?",  "answerable"),
    # Independent contractor
    (18, "Does this agreement create an employer-employee relationship between Affiliate and its contractors?", "answerable"),
    (19, "Who is responsible for paying the Affiliate's business overhead, taxes, and self-employment expenses?", "answerable"),
    # Dispute resolution & misc
    (20, "Where are disputes brought by Payment Data Systems resolved under this agreement?",   "answerable"),
    # ── UNANSWERABLE (gold = "unanswerable") ─────────────────────────────────
    # External exhibits / schedules not included
    (21, "What is the exact revenue share percentage stated in Exhibit A?",                     "unanswerable"),
    (22, "What are the specific buy rates for merchants as listed in Exhibit A?",               "unanswerable"),
    (23, "What equipment prices appear on Network 1's price list provided to the Affiliate?",   "unanswerable"),
    # Customer-instance / runtime facts
    (24, "What is the Affiliate's current monthly revenue earned under this agreement?",        "unanswerable"),
    (25, "How many merchants has the Affiliate enrolled as of today?",                         "unanswerable"),
    (26, "What is the current outstanding balance owed by the Affiliate to Network 1 for equipment?", "unanswerable"),
    (27, "On what specific date was this agreement executed?",                                  "unanswerable"),
    (28, "Who are the authorized representatives who signed this agreement on behalf of each party?", "unanswerable"),
    # Outside-document / speculative
    (29, "What is Network 1 Financial Corporation's total annual revenue?",                     "unanswerable"),
    (30, "Who is the current CEO of Network 1 Financial Corporation?",                         "unanswerable"),
    (31, "How many other affiliate offices has Network 1 entered into similar agreements with?","unanswerable"),
    (32, "Has this agreement ever been subject to an arbitration proceeding?",                  "unanswerable"),
    (33, "What is Harris Bank's principal place of business address?",                         "unanswerable"),
    (34, "What specific software products does Network 1 currently offer for sale to affiliates?", "unanswerable"),
    (35, "What was Network 1's stock price on the date this agreement was signed?",             "unanswerable"),
    (36, "What is the current processing fee percentage charged to the merchants enrolled under this agreement?", "unanswerable"),
    (37, "Which specific merchants by name have been approved by Network 1 under this agreement?", "unanswerable"),
    (38, "What amendments to this agreement have been executed since it was first signed?",     "unanswerable"),
    (39, "What is the internal credit risk score Network 1 assigns to the Affiliate?",         "unanswerable"),
    (40, "What will happen to this agreement if Network 1 is acquired by another company next year?", "unanswerable"),
]


def main():
    print(f"\nLoading pipeline from: {PDF.name}")
    t0 = time.time()
    pipe = cp.load_pdf(str(PDF))
    print(f"Pipeline ready in {time.time()-t0:.1f}s — running {len(QUESTIONS)} questions\n")

    tp = tn = fp = fn = 0
    rows, errors = [], []
    t_start = time.time()

    for idx, question, gold in QUESTIONS:
        t1 = time.time()
        r = cp.run_query(pipe, question)
        elapsed = round(time.time() - t1, 2)

        pred_label = "answerable" if r["answerable"] else "unanswerable"
        gold_bool  = (gold == "answerable")
        pred_bool  = r["answerable"]
        correct    = (gold_bool == pred_bool)

        topo = r.get("topo_pred", {})
        meta = topo.get("meta", {})

        row = {
            "idx": idx,
            "question": question,
            "gold_label": gold,
            "predicted_label": pred_label,
            "score": topo.get("score"),
            "threshold": topo.get("threshold"),
            "gate_reason": meta.get("gate_reason", "none"),
            "escape_hatch_triggered": meta.get("escape_hatch_triggered", False),
            "q_seconds": elapsed,
        }
        rows.append(row)
        if not correct:
            errors.append(row)

        if gold_bool and pred_bool:        tp += 1
        elif not gold_bool and not pred_bool: tn += 1
        elif not gold_bool and pred_bool:     fp += 1
        else:                                  fn += 1

        mark = "✓" if correct else "✗"
        print(f"  {mark} idx={idx:2d} [{gold[0].upper()}→{pred_label[0].upper()}] "
              f"score={topo.get('score', 0):.3f} thr={topo.get('threshold', 0):.3f}  vote={r.get('vote_fraction') or 0:.2f}{'R' if r.get('answerability_gate',{}).get('retry_fired') else ''}"
              f"({elapsed}s)  {question[:65]}")

    total = len(QUESTIONS)
    accuracy = round((tp + tn) / total, 4)
    print(f"\n── Results ────────────────────────────────────────")
    print(f"  Accuracy:  {accuracy:.1%}  ({tp+tn}/{total})")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    idx={e['idx']}  gold={e['gold_label']}  pred={e['predicted_label']}  "
                  f"score={e['score']:.3f}  thr={e['threshold']:.3f}  {e['question'][:70]}")
    else:
        print("\n  No errors — perfect run!")

    result = {
        "summary": {
            "total": total,
            "accuracy": accuracy,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "total_seconds": round(time.time() - t_start, 1),
        },
        "rows": rows,
        "errors": errors,
    }
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nFull results → {OUT.name}")


if __name__ == "__main__":
    main()
