#!/usr/bin/env python3
"""Run 40-question benchmark against the Co-Branding Agreement PDF.

Usage:
    TOPO_LLM_INTENT_MODE=llm_only CORE_PRESTEP_PARALLEL=1 \
        python scripts/run_eval_cobranding.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core_pipeline as cp

PDF = Path("/Users/jonathang/Downloads/LeadersonlineInc_20000427_S-1A_EX-10.8_4991089_EX-10.8_Co-Branding Agreement.pdf")
OUT = ROOT / "scripts" / "cobranding_eval_result.json"

QUESTIONS = [
    # ── ANSWERABLE ────────────────────────────────────────────────────────────
    (1,  "What is the Effective Date of this agreement?",                                        "answerable"),
    (2,  "What state is VerticalNet, Inc. incorporated in?",                                     "answerable"),
    (3,  "What is LeadersOnline's principal place of business address?",                         "answerable"),
    (4,  "What is the end date of the initial Term of this agreement?",                          "answerable"),
    (5,  "On which specific pages of the LeadersOnline Site must the VerticalNet Branded Link appear?", "answerable"),
    (6,  "Is the Mark License granted to LeadersOnline exclusive or non-exclusive?",             "answerable"),
    (7,  "To which URL does the Link License direct users?",                                     "answerable"),
    (8,  "For how long must VerticalNet place a purchased Banner on the HR Site Home Page?",     "answerable"),
    (9,  "How many times per calendar quarter may LeadersOnline contact a Resume Bank candidate?", "answerable"),
    (10, "Within how many days must LeadersOnline pay an underpayment identified by an audit?",  "answerable"),
    (11, "Within how many days must VerticalNet return an overpayment identified by an audit?",  "answerable"),
    (12, "What interest rate applies to payments not paid by their due date?",                   "answerable"),
    (13, "How many days does a breaching party have to cure a material breach before the other party may terminate?", "answerable"),
    (14, "How many days before the end of the initial Term must LeadersOnline notify VerticalNet to exercise the renewal option?", "answerable"),
    (15, "In what city will arbitration be held under this agreement?",                          "answerable"),
    (16, "Who appoints the arbitrator, and how long do the parties have to agree on one?",       "answerable"),
    (17, "What types of damages is the arbitrator prohibited from awarding?",                    "answerable"),
    (18, "What is the limitations period for bringing a claim under this agreement?",            "answerable"),
    (19, "How are the costs and expenses of the arbitrator shared between the parties?",         "answerable"),
    (20, "For how many months after termination is LeadersOnline still obligated to pay placement fees under Section 6.3.2?", "answerable"),
    # ── UNANSWERABLE ──────────────────────────────────────────────────────────
    # Redacted (***) values
    (21, "What is the total dollar amount of the Banner and Newsletter Ad purchase commitment?", "unanswerable"),
    (22, "What percentage discount does LeadersOnline receive on Banner and Newsletter Ad purchases?", "unanswerable"),
    (23, "What is the revenue share percentage LeadersOnline must pay VerticalNet under Section 6.3.1?", "unanswerable"),
    (24, "What is the slotting fee for Employer Spotlights under Section 6.1.1?",               "unanswerable"),
    (25, "What is the license fee for access to the Resume Bank under Section 6.1.3?",          "unanswerable"),
    (26, "What is the first installment payment amount due on the Effective Date?",              "unanswerable"),
    (27, "What is the Guaranteed Revenue amount VerticalNet promises under Section 6.5?",       "unanswerable"),
    (28, "What are the specific placement fee amounts for candidates with salaries below the first salary tier?", "unanswerable"),
    # Runtime / customer-instance facts
    (29, "What is the total LeadersOnline-VerticalNet Revenue earned to date under this agreement?", "unanswerable"),
    (30, "How many VerticalNet-LeadersOnline Clients have been generated since the Effective Date?", "unanswerable"),
    (31, "Who are the authorized signatories who executed this agreement on behalf of each party?", "unanswerable"),
    (32, "Has this agreement been renewed for a Renewal Term?",                                  "unanswerable"),
    # Outside document / speculative
    (33, "What is VerticalNet's current stock price?",                                          "unanswerable"),
    (34, "How many total Online Communities does VerticalNet operate?",                         "unanswerable"),
    (35, "What specific industries does each VerticalNet Online Community cover?",              "unanswerable"),
    (36, "What is VerticalNet's current Privacy Policy for the Resume Bank?",                   "unanswerable"),
    (37, "Has any arbitration proceeding been commenced under this agreement?",                  "unanswerable"),
    (38, "What amendments to this agreement have been signed since the Effective Date?",        "unanswerable"),
    (39, "What is LeadersOnline's current number of registered users?",                        "unanswerable"),
    (40, "What will happen to VerticalNet's Online Communities if the internet advertising market collapses?", "unanswerable"),
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
            "idx": idx, "question": question, "gold_label": gold,
            "predicted_label": pred_label,
            "score": topo.get("score"), "threshold": topo.get("threshold"),
            "gate_reason": meta.get("gate_reason", "none"),
            "vote_fraction": r.get("vote_fraction"),
            "q_seconds": elapsed,
        }
        rows.append(row)
        if not correct:
            errors.append(row)

        if gold_bool and pred_bool:            tp += 1
        elif not gold_bool and not pred_bool:  tn += 1
        elif not gold_bool and pred_bool:      fp += 1
        else:                                  fn += 1

        mark = "✓" if correct else "✗"
        print(f"  {mark} idx={idx:2d} [{gold[0].upper()}→{pred_label[0].upper()}] "
              f"score={topo.get('score',0):.3f} thr={topo.get('threshold',0):.3f}  vote={r.get('vote_fraction') or 0:.2f}{'R' if r.get('answerability_gate',{}).get('retry_fired') else ''}"
              f"({elapsed}s)  {question[:65]}")

    total = len(QUESTIONS)
    accuracy = round((tp + tn) / total, 4)
    print(f"\n── Results ────────────────────────────────────────")
    print(f"  Accuracy:  {accuracy:.1%}  ({tp+tn}/{total})")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Answerable:   {tp}/20 = {tp/20:.0%}")
    print(f"  Unanswerable: {tn}/20 = {tn/20:.0%}")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    idx={e['idx']}  gold={e['gold_label']}  pred={e['predicted_label']}  "
                  f"score={e['score']:.3f}  thr={e['threshold']:.3f}  {e['question'][:70]}")
    else:
        print("\n  No errors — perfect run!")

    result = {
        "summary": {"total": total, "accuracy": accuracy,
                    "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                    "total_seconds": round(time.time() - t_start, 1)},
        "rows": rows, "errors": errors,
    }
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nFull results → {OUT.name}")


if __name__ == "__main__":
    main()
