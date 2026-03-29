#!/usr/bin/env python3
"""Run 40-question benchmark against the Snap/United Airlines (Embark) Co-Branding Agreement PDF."""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core_pipeline as cp

PDF = Path("/Users/jonathang/Downloads/EmbarkComInc_19991008_S-1A_EX-10.10_6487661_EX-10.10_Co-Branding Agreement.pdf")
OUT = ROOT / "scripts" / "embark_eval_result.json"

QUESTIONS = [
    # ── ANSWERABLE ────────────────────────────────────────────────────────────
    (1,  "What is the date of this agreement?",                                                                     "answerable"),
    (2,  "What is Snap's principal place of business address?",                                                     "answerable"),
    (3,  "What state is Snap Technologies, Inc. incorporated in?",                                                  "answerable"),
    (4,  "What state is United Airlines incorporated in?",                                                          "answerable"),
    (5,  "What URL is the primary home page of the Snap Web Site?",                                                 "answerable"),
    (6,  "What URL is the primary home page of the Sponsor Web Site?",                                              "answerable"),
    (7,  "What is the end date of the Term of this agreement?",                                                     "answerable"),
    (8,  "How many days prior written notice is required to terminate for breach?",                                  "answerable"),
    (9,  "How many days prior to expiration must Sponsor deliver notice to exercise the right of first refusal?",   "answerable"),
    (10, "How many days does Snap have to negotiate with Sponsor after Sponsor delivers the right-of-first-refusal notice before Snap is free to deal with third parties?", "answerable"),
    (11, "Is the license Sponsor grants Snap to use Sponsor Marks revocable?",                                      "answerable"),
    (12, "What governing law applies to this agreement?",                                                           "answerable"),
    (13, "In what city must disputes be brought under this agreement?",                                             "answerable"),
    (14, "For how many years after the Term must the receiving party keep Confidential Information confidential?",  "answerable"),
    (15, "What types of damages does each party waive under the limitation of liability provision?",                "answerable"),
    (16, "Does Snap warrant that the Snap Web Site will be free from bugs, faults, or defects?",                   "answerable"),
    (17, "Can either party assign this agreement without consent to a corporate affiliate?",                        "answerable"),
    (18, "Does this agreement create a partnership or joint venture between Snap and United Airlines?",             "answerable"),
    (19, "Who retains all right, title and interest in the Snap Web Site?",                                        "answerable"),
    (20, "What obligation does Sponsor have to promote and market Snap throughout the Term?",                       "answerable"),
    # ── UNANSWERABLE ──────────────────────────────────────────────────────────
    # Redacted / exhibit-only values
    (21, "What is the Up-Front Fee Sponsor pays to Snap upon execution?",                                           "unanswerable"),
    (22, "What are the quarterly fee payment amounts Sponsor pays Snap?",                                           "unanswerable"),
    (23, "What is the Exclusive Category identified in Exhibit A?",                                                 "unanswerable"),
    (24, "What specific Snap Marks are listed in Exhibit B?",                                                       "unanswerable"),
    (25, "What specific Sponsor Marks are listed in Exhibit B?",                                                    "unanswerable"),
    (26, "What specific content is described in Exhibit C?",                                                        "unanswerable"),
    (27, "What specific promotional and marketing services are listed in Exhibit D?",                               "unanswerable"),
    # Runtime / instance facts
    (28, "How many Snap end users have been enrolled in United's frequent flier program under this agreement?",     "unanswerable"),
    (29, "Has this agreement been amended since execution?",                                                        "unanswerable"),
    (30, "What specific travel-related content has been developed for the Co-Branded Pages to date?",               "unanswerable"),
    (31, "Has any dispute arisen between Snap and United Airlines under this agreement?",                           "unanswerable"),
    (32, "How many page views have the Co-Branded Pages received since launch?",                                    "unanswerable"),
    # Outside document / speculative
    (33, "Who is the current CEO of United Airlines?",                                                              "unanswerable"),
    (34, "What is United Airlines' current stock price?",                                                           "unanswerable"),
    (35, "What is Snap's current number of registered users?",                                                      "unanswerable"),
    (36, "What is Snap's current annual revenue?",                                                                  "unanswerable"),
    (37, "What will happen to this agreement if United Airlines is acquired by another airline?",                   "unanswerable"),
    (38, "What is Snap's internal assessment of United Airlines' financial reliability?",                           "unanswerable"),
    (39, "What specific Mileage Plus account integration has been implemented on the Co-Branded Pages?",            "unanswerable"),
    (40, "What amendments to this agreement have been signed since the Effective Date?",                            "unanswerable"),
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
