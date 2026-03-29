#!/usr/bin/env python3
"""Run 40-question benchmark against the eDiets/Women.com Co-Branding Agreement PDF."""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core_pipeline as cp

PDF = Path("/Users/jonathang/Downloads/EdietsComInc_20001030_10QSB_EX-10.4_2606646_EX-10.4_Co-Branding Agreement.pdf")
OUT = ROOT / "scripts" / "ediets_eval_result.json"

QUESTIONS = [
    # ── ANSWERABLE ────────────────────────────────────────────────────────────
    (1,  "What is the Effective Date of this agreement?",                                                           "answerable"),
    (2,  "What is Women.com's principal place of business address?",                                                "answerable"),
    (3,  "What state is Women.com Networks, Inc. incorporated in?",                                                 "answerable"),
    (4,  "How many days after the Effective Date must eDiets deliver initial content to Women.com?",                "answerable"),
    (5,  "On how many Women.com channels must links to the Diet Center be located?",                                "answerable"),
    (6,  "How long after the Launch Date must eDiets add a back button to Diet Center pages?",                      "answerable"),
    (7,  "How many days after the end of a quarter does Women.com have to deliver make-good impressions before eDiets may terminate?", "answerable"),
    (8,  "How many days after termination must Women.com reimburse eDiets for pre-paid impressions not delivered?", "answerable"),
    (9,  "Is the content license eDiets grants to Women.com exclusive or non-exclusive?",                           "answerable"),
    (10, "How many hours written notice must Women.com give before withdrawing permission to use Women.com Marks?", "answerable"),
    (11, "What is the maximum percentage by which Women.com can increase the Payment Schedule in any 12-month period?", "answerable"),
    (12, "What is the Quarterly Impression Guarantee for Advertising Promotions under this agreement?",             "answerable"),
    (13, "What percentage traffic differential triggers the dispute resolution procedure for media metrics?",       "answerable"),
    (14, "How many days does eDiets have to respond to Women.com's written notice about a Diet Promo opportunity?","answerable"),
    (15, "What percentage of the Quarterly Impression Guarantee must Women.com deliver each quarter to avoid a make-good obligation?", "answerable"),
    (16, "What is the initial term length of this agreement?",                                                      "answerable"),
    (17, "Who has sole responsibility for providing and maintaining the Diet Center beneath the Gateway Page?",     "answerable"),
    (18, "How many days after Women.com's submission does eDiets have to provide content revisions before failure is deemed approval?", "answerable"),
    (19, "Is Women.com registration required to enter and use the Gateway Page of the Diet Center?",               "answerable"),
    (20, "What system does Women.com use to generate advertising reports provided to eDiets?",                      "answerable"),
    # ── UNANSWERABLE ──────────────────────────────────────────────────────────
    # Redacted / exhibit-only values
    (21, "What is the monthly fee eDiets pays to Women.com under this agreement?",                                  "unanswerable"),
    (22, "What are the specific advertising rates and amounts in Exhibit B?",                                       "unanswerable"),
    (23, "What are the specific eDiets Content production specifications listed in Exhibit D?",                     "unanswerable"),
    (24, "What specific Women.com subchannels are designated for Diet Center links in Exhibit B?",                  "unanswerable"),
    (25, "What are the specific look-and-feel specifications for the Gateway Page in Exhibit A?",                   "unanswerable"),
    # Runtime / instance facts
    (26, "How many registered eDiets members are currently active?",                                                "unanswerable"),
    (27, "Has Women.com ever failed to meet its Quarterly Impression Guarantee under this agreement?",              "unanswerable"),
    (28, "What specific production schedule have the parties developed for eDiets Content delivery?",               "unanswerable"),
    (29, "What amendments to this agreement have been signed since the Effective Date?",                            "unanswerable"),
    (30, "Has any Women.com user filed a third-party claim based on eDiets nutritional facts?",                     "unanswerable"),
    # Outside document / speculative
    (31, "What is Women.com's current stock price?",                                                                "unanswerable"),
    (32, "What is the current CEO of Women.com Networks, Inc.?",                                                    "unanswerable"),
    (33, "What is eDiets' current diet program subscriber count?",                                                  "unanswerable"),
    (34, "What is eDiets' current annual revenue?",                                                                 "unanswerable"),
    (35, "What are the specific editorial guidelines Women.com has provided to eDiets?",                            "unanswerable"),
    (36, "What is Women.com's current privacy policy text?",                                                        "unanswerable"),
    (37, "What will happen to the Diet Center if eDiets files for bankruptcy next month?",                          "unanswerable"),
    (38, "How many total page views has the Diet Center received since the Launch Date?",                           "unanswerable"),
    (39, "What specific eDiets Content has been approved and published on the Diet Center to date?",                "unanswerable"),
    (40, "What is eDiets' current stock price?",                                                                    "unanswerable"),
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
