#!/usr/bin/env python3
"""Run 40-question benchmark against the HealthCentral/MediaLinx Co-Branding Agreement PDF."""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core_pipeline as cp

PDF = Path("/Users/jonathang/Downloads/HealthcentralCom_19991108_S-1A_EX-10.27_6623292_EX-10.27_Co-Branding Agreement.pdf")
OUT = ROOT / "scripts" / "healthcentral_eval_result.json"

QUESTIONS = [
    # ── ANSWERABLE ────────────────────────────────────────────────────────────
    (1,  "What is the date this agreement was made?",                                                               "answerable"),
    (2,  "What is MLX's principal address?",                                                                        "answerable"),
    (3,  "What is HCI's principal place of business address?",                                                      "answerable"),
    (4,  "What is the agreed launch date of the Co-Branded Site?",                                                  "answerable"),
    (5,  "How many days per week and hours per day must the Co-Branded Site be available?",                         "answerable"),
    (6,  "How many days advance written notice must be given before conducting an audit?",                          "answerable"),
    (7,  "Within how many days of the end of each quarter must remittances and accounting reports be delivered?",   "answerable"),
    (8,  "What is the initial term length of this agreement?",                                                      "answerable"),
    (9,  "What share of net advertising revenue exceeding the annual threshold is split between the parties?",      "answerable"),
    (10, "Which party is entitled to all proceeds generated from US companies on the Co-Branded Site?",             "answerable"),
    (11, "For how long after the Term does MLX's restriction on using Co-Branded Site data for competing health purposes apply?", "answerable"),
    (12, "What percentage of the HealthyWay title font size must the 'Powered By HealthCentral' tag line be approximately?", "answerable"),
    (13, "What is the maximum fraction of promotion that competing third-party health content may receive on the Sympatico Health home page?", "answerable"),
    (14, "If an audit determines the other party underpaid by 10% or more, who bears the audit cost?",              "answerable"),
    (15, "Is the Sympatico Content license granted to HCI exclusive or non-exclusive?",                             "answerable"),
    (16, "What URL shall the Co-Branded Site retain?",                                                              "answerable"),
    (17, "For how long from the Launch Date must the primary branding of the Co-Branded Site remain HealthyWay?",   "answerable"),
    (18, "How many days prior written notice is required to terminate this agreement for material breach?",         "answerable"),
    (19, "How many days must insolvency proceedings remain pending before a party may immediately terminate?",      "answerable"),
    (20, "Who retains all right, title and interest in the HealthyWay discussion forums?",                          "answerable"),
    # ── UNANSWERABLE ──────────────────────────────────────────────────────────
    # Redacted / schedule-only values
    (21, "What is the guaranteed fixed annual sum HCI pays MLX under this agreement?",                              "unanswerable"),
    (22, "What is the monthly payment amount HCI owes MLX?",                                                        "unanswerable"),
    (23, "What advertising revenue threshold does HCI receive exclusively before the 50-50 revenue split applies?", "unanswerable"),
    (24, "What specific promotion obligations for the Co-Branded Site are listed in Schedule 1?",                   "unanswerable"),
    (25, "What traffic levels are specified in Schedule 2 for triggering MLX's promotion obligations?",             "unanswerable"),
    (26, "What Canadian content requirements are specified in Schedule 3?",                                         "unanswerable"),
    (27, "What is the format for daily usage statistics set out in Schedule 4?",                                    "unanswerable"),
    # Runtime / instance facts
    (28, "How much total revenue has the Co-Branded Site generated to date under this agreement?",                  "unanswerable"),
    (29, "What specific Canadian advertisers have purchased advertising on the Co-Branded Site?",                   "unanswerable"),
    (30, "Has this agreement been amended since the Launch Date?",                                                   "unanswerable"),
    (31, "What specific e-commerce relationships have been approved and implemented on the Co-Branded Site?",       "unanswerable"),
    (32, "How many users has the Co-Branded Site registered since the Launch Date?",                                "unanswerable"),
    # Outside document / speculative
    (33, "What is MediaLinx's current stock price?",                                                                "unanswerable"),
    (34, "Who is the current CEO of HealthCentral.com?",                                                            "unanswerable"),
    (35, "What is HealthCentral.com's current annual revenue?",                                                     "unanswerable"),
    (36, "What is MediaLinx's current Canadian internet market share?",                                             "unanswerable"),
    (37, "Has any user of the Co-Branded Site filed a health-related complaint against either party?",              "unanswerable"),
    (38, "What will happen to this agreement if MediaLinx is acquired by a competitor?",                            "unanswerable"),
    (39, "What specific content has MLX approved for publication on the Co-Branded Site?",                          "unanswerable"),
    (40, "What is the current page view count on the Co-Branded Site?",                                             "unanswerable"),
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
