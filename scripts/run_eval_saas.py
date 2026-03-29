#!/usr/bin/env python3
"""Run 40-question benchmark against the CyberArk SaaS Terms of Service PDF.

Usage:
    TOPO_LLM_INTENT_MODE=llm_only CORE_PRESTEP_PARALLEL=1 \
        python scripts/run_eval_saas.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core_pipeline as cp

PDF = Path("/Users/jonathang/Downloads/SaaS-Terms-of-Service.pdf")
OUT = ROOT / "scripts" / "saas_eval_result.json"

QUESTIONS = [
    # ── ANSWERABLE ────────────────────────────────────────────────────────────
    (1,  "What is the late charge rate applied to unpaid invoices?",                                              "answerable"),
    (2,  "How many days does Customer have to pay CyberArk's invoice when paying directly?",                     "answerable"),
    (3,  "Will CyberArk use Customer Data to train generative AI models?",                                        "answerable"),
    (4,  "Does CyberArk guarantee the accuracy of outputs produced by AI/ML Features?",                          "answerable"),
    (5,  "What is the maximum aggregate liability cap for either party under this agreement?",                    "answerable"),
    (6,  "What types of damages are excluded from liability under Section 8.2?",                                  "answerable"),
    (7,  "How many days does a breaching party have to cure a material breach before the other party may terminate?", "answerable"),
    (8,  "How many months' prior written notice must Customer give to terminate an Order under the EU Data Act?", "answerable"),
    (9,  "How many days notice must CyberArk give before suspending Customer's access to the SaaS Products?",    "answerable"),
    (10, "How many days after sending does a notice sent by registered or certified mail become effective?",      "answerable"),
    (11, "How many days after deposit does a notice sent by overnight courier become effective?",                 "answerable"),
    (12, "What email address should electronic contract notices to CyberArk be sent to?",                        "answerable"),
    (13, "Is Customer's payment obligation excluded from force majeure protection?",                              "answerable"),
    (14, "What does CyberArk warrant about Third-Party Materials included in the SaaS Products?",                "answerable"),
    (15, "Who owns CyberArk products that incorporate Suggestions provided by Customer?",                        "answerable"),
    (16, "What is the revision date of this Terms of Service document?",                                         "answerable"),
    (17, "What happens to prepaid Subscription Fees when Customer terminates an Order under the EU Data Act?",   "answerable"),
    (18, "Who is responsible for the accuracy and quality of Customer Data?",                                    "answerable"),
    (19, "Where is the Data Processing Addendum (DPA) located for customers in the EEA?",                       "answerable"),
    (20, "Does this Agreement create a joint venture or partnership between CyberArk and Customer?",             "answerable"),
    # ── UNANSWERABLE ──────────────────────────────────────────────────────────
    # Customer-instance / order-specific
    (21, "What specific SaaS Products has this Customer ordered?",                                               "unanswerable"),
    (22, "What is the subscription price for this Customer's Order?",                                            "unanswerable"),
    (23, "What is the Subscription Term duration for this Customer?",                                            "unanswerable"),
    (24, "How many Authorized Users does this Customer currently have?",                                         "unanswerable"),
    (25, "Which specific Channel Partner does this Customer use?",                                               "unanswerable"),
    (26, "What discount did this Customer receive on their subscription?",                                       "unanswerable"),
    # External documents referenced but not included
    (27, "What are the specific data processing obligations set out in the CyberArk DPA?",                       "unanswerable"),
    (28, "What are the specific terms of the CyberArk Business Associate Agreement for HIPAA compliance?",       "unanswerable"),
    (29, "What does CyberArk's Responsible AI Policy say about data retention?",                                 "unanswerable"),
    (30, "What specific open source software components are included in the SaaS Products?",                     "unanswerable"),
    # Outside document / speculative
    (31, "What is CyberArk's current annual revenue?",                                                           "unanswerable"),
    (32, "What is CyberArk's uptime SLA percentage for the SaaS Products?",                                     "unanswerable"),
    (33, "Has CyberArk experienced any security incidents affecting Customer Data?",                              "unanswerable"),
    (34, "What is CyberArk's current stock price?",                                                              "unanswerable"),
    (35, "Who is the current CEO of CyberArk?",                                                                  "unanswerable"),
    (36, "Has this Customer filed any indemnification claims against CyberArk?",                                 "unanswerable"),
    (37, "What amendments to this agreement have been made for this specific Customer?",                         "unanswerable"),
    (38, "What are the specific security controls listed in the Documentation?",                                  "unanswerable"),
    (39, "What will CyberArk's liability be if it fails to provide the SaaS service next month?",               "unanswerable"),
    (40, "What governing law applies to this specific Customer's agreement with CyberArk?",                      "unanswerable"),
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
