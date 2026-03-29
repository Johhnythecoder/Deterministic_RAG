#!/usr/bin/env python3
"""Evaluate topology answerability scoring against a labeled CSV.

Expected CSV format:
- Has an `Expected` column with values like:
  - "answerable", "unanswerable"
  - "✓ ans", "✕ una"
- Includes topology-export columns (for best fidelity):
  Slot score, Path bottleneck, H0 life, Chain str, Trig wt, Sheaf cons,
  Wt sheaf, Wt Ricci μ, Bridge scr, Type H, Mismatch, Role inv,
  N targets, N well-conn, Slot type, etc.

This script does NOT require scikit-learn.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topology_metrics import _predict_answerability  # noqa: E402


BAD_TOKENS = {"", "—", "-", "na", "n/a", "none", "null", "nan"}


@dataclass
class EvalRow:
    idx: int
    q: str
    slot_type: str
    expected: int
    predicted: int
    score: float
    threshold: float
    confidence: str
    evidence_score: float | None
    risk_score: float | None
    disagreement: float | None


def parse_float(v: str | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in BAD_TOKENS:
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_int(v: str | None, default: int = 0) -> int:
    x = parse_float(v)
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def parse_expected(v: str | None) -> int | None:
    s = (v or "").strip().lower()
    if not s:
        return None
    if "una" in s:
        return 0
    if "ans" in s:
        return 1
    if s in {"true", "1", "yes"}:
        return 1
    if s in {"false", "0", "no"}:
        return 0
    return None


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def auc_from_scores(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        greater = sum(1 for n in neg if p > n)
        ties = sum(1 for n in neg if p == n)
        wins += greater + 0.5 * ties
    return wins / (len(pos) * len(neg))


def brier_score(scores: list[float], labels: list[int]) -> float:
    if not scores:
        return 0.0
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(scores)


def cohens_d(pos_vals: list[float], neg_vals: list[float]) -> float | None:
    if not pos_vals or not neg_vals:
        return None
    m1 = sum(pos_vals) / len(pos_vals)
    m0 = sum(neg_vals) / len(neg_vals)
    s1 = math.sqrt(sum((x - m1) ** 2 for x in pos_vals) / max(1, len(pos_vals)))
    s0 = math.sqrt(sum((x - m0) ** 2 for x in neg_vals) / max(1, len(neg_vals)))
    sp = math.sqrt((s1 * s1 + s0 * s0) / 2.0)
    if sp < 1e-12:
        return 0.0
    return (m1 - m0) / sp


def build_topology_from_csv_row(row: dict[str, str], force_drop: set[str] | None = None) -> dict[str, Any]:
    drop = force_drop or set()

    def maybe(name: str, val: Any) -> Any:
        return None if name in drop else val

    slot_type = (row.get("Slot type") or "").strip() or "GENERAL"

    path_exists_raw = parse_float(row.get("Path exists"))
    if path_exists_raw is None:
        path_exists = True
    else:
        path_exists = path_exists_raw > 0.0

    return {
        "typed_path_coverage": {
            "score": maybe("slot_score", parse_float(row.get("Slot score"))),
        },
        "answer_path": {
            "path_bottleneck": maybe("path_bottleneck", parse_float(row.get("Path bottleneck"))),
            "slot_type": slot_type,
            "path_exists": maybe("path_exists", path_exists),
            "n_targets": maybe("n_targets", parse_int(row.get("N targets"), default=0)),
            "n_well_connected": maybe("n_well_connected", parse_int(row.get("N well-conn"), default=0)),
        },
        "persistent_homology": {
            "h0_mean_lifetime": maybe("h0_life", parse_float(row.get("H0 life"))),
        },
        "chain": {
            "chain_strength": maybe("chain_strength", parse_float(row.get("Chain str"))),
            "trigger_edge_weight": maybe("trigger_edge_weight", parse_float(row.get("Trig wt"))),
        },
        "sheaf": {
            "consistent_frac": maybe("sheaf_consistent_frac", parse_float(row.get("Sheaf cons"))),
        },
        "weighted": {
            "weighted_consistent_frac": maybe("weighted_consistent_frac", parse_float(row.get("Wt sheaf"))),
            "weighted_mean": maybe("weighted_mean", parse_float(row.get("Wt Ricci μ"))),
            "anchor_bridge_score": maybe("anchor_bridge_score", parse_float(row.get("Bridge scr"))),
        },
        "type_metrics": {
            "type_entropy": maybe("type_entropy", parse_float(row.get("Type H"))),
        },
        "slot_mismatch_depth": {
            "exact_frac": maybe("exact_frac", parse_float(row.get("Path cov"))),
            "adjacent_frac": maybe("adjacent_frac", parse_float(row.get("Path cov"))),
            "distant_frac": maybe("distant_frac", parse_float(row.get("Mismatch"))),
        },
        "required_sequence": {
            "sequence_frac": maybe("sequence_frac", parse_float(row.get("SS frac"))),
        },
        "direction": {
            "party_mismatch_frac": maybe("party_mismatch_frac", parse_float(row.get("Mismatch"))),
            "edge_role_inversion": maybe("edge_role_inversion", parse_float(row.get("Role inv"))),
        },
        "value_tag_match": {
            "has_tags": False if "value_tag_match" not in drop else False,
            "match_score": maybe("value_tag_match", parse_float(row.get("Match frac"))),
        },
        "placeholder_values": {
            "placeholder_frac": maybe("placeholder_frac", parse_float(row.get("Placeholder frac"))),
        },
        "keyword_trap": {
            "flagged": None if "keyword_trap" not in drop else None,
            "confidence": maybe("keyword_trap_confidence", parse_float(row.get("KwTrap"))),
        },
    }


def evaluate_rows(rows: list[dict[str, str]], drop: set[str] | None = None) -> list[EvalRow]:
    out: list[EvalRow] = []
    for i, row in enumerate(rows, start=1):
        y = parse_expected(row.get("Expected"))
        if y is None:
            continue
        topo = build_topology_from_csv_row(row, force_drop=drop)
        pred = _predict_answerability(topo)
        yhat = 1 if pred.get("predicted") == "answerable" else 0
        meta = pred.get("meta") or {}
        out.append(
            EvalRow(
                idx=i,
                q=(row.get("Q") or "").strip(),
                slot_type=(row.get("Slot type") or "").strip() or "GENERAL",
                expected=y,
                predicted=yhat,
                score=float(pred.get("score") or 0.0),
                threshold=float(pred.get("threshold") or 0.5),
                confidence=str(pred.get("confidence") or "low"),
                evidence_score=(float(meta["evidence_score"]) if meta.get("evidence_score") is not None else None),
                risk_score=(float(meta["risk_score"]) if meta.get("risk_score") is not None else None),
                disagreement=(float(meta["disagreement"]) if meta.get("disagreement") is not None else None),
            )
        )
    return out


def summarize(rows: list[EvalRow]) -> dict[str, Any]:
    labels = [r.expected for r in rows]
    preds = [r.predicted for r in rows]
    scores = [r.score for r in rows]
    thresholds = [r.threshold for r in rows]
    n = len(rows)

    tp = sum(1 for y, yh in zip(labels, preds) if y == 1 and yh == 1)
    tn = sum(1 for y, yh in zip(labels, preds) if y == 0 and yh == 0)
    fp = sum(1 for y, yh in zip(labels, preds) if y == 0 and yh == 1)
    fn = sum(1 for y, yh in zip(labels, preds) if y == 1 and yh == 0)

    acc = safe_div(tp + tn, n)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
    auc = auc_from_scores(scores, labels)
    brier = brier_score(scores, labels)

    score_ans = [s for s, y in zip(scores, labels) if y == 1]
    score_una = [s for s, y in zip(scores, labels) if y == 0]

    by_slot: dict[str, dict[str, Any]] = {}
    slot_keys = sorted(set(r.slot_type for r in rows))
    for st in slot_keys:
        part = [r for r in rows if r.slot_type == st]
        nst = len(part)
        ok = sum(1 for r in part if r.expected == r.predicted)
        by_slot[st] = {
            "n": nst,
            "accuracy": round(safe_div(ok, nst), 4),
            "answerable_n": sum(1 for r in part if r.expected == 1),
            "unanswerable_n": sum(1 for r in part if r.expected == 0),
            "mean_score": round(sum(r.score for r in part) / max(1, nst), 4),
            "mean_threshold": round(sum(r.threshold for r in part) / max(1, nst), 4),
        }

    # Calibration bins
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]
    calibration: list[dict[str, Any]] = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        idxs = [k for k, s in enumerate(scores) if lo <= s < hi]
        if not idxs:
            continue
        ys = [labels[k] for k in idxs]
        ps = [scores[k] for k in idxs]
        calibration.append(
            {
                "bin": f"[{lo:.1f},{min(1.0, hi):.1f}]",
                "n": len(idxs),
                "mean_pred": round(sum(ps) / len(ps), 4),
                "empirical": round(sum(ys) / len(ys), 4),
            }
        )

    return {
        "n": n,
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4) if auc is not None else None,
            "brier": round(brier, 4),
        },
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "score_split": {
            "mean_answerable": round(sum(score_ans) / max(1, len(score_ans)), 4),
            "mean_unanswerable": round(sum(score_una) / max(1, len(score_una)), 4),
            "median_answerable": round(sorted(score_ans)[len(score_ans) // 2], 4) if score_ans else None,
            "median_unanswerable": round(sorted(score_una)[len(score_una) // 2], 4) if score_una else None,
            "mean_threshold": round(sum(thresholds) / max(1, len(thresholds)), 4),
        },
        "calibration": calibration,
        "by_slot_type": by_slot,
    }


def build_signal_diagnostics(rows: list[dict[str, str]]) -> dict[str, Any]:
    numeric_cols = [
        "Slot score",
        "Path bottleneck",
        "H0 life",
        "Chain str",
        "Trig wt",
        "Sheaf cons",
        "Wt sheaf",
        "Wt Ricci μ",
        "Bridge scr",
        "Type H",
        "Mismatch",
        "Role inv",
        "Match frac",
        "SS frac",
        "Path cov",
        "Path score",
        "Path len",
        "N targets",
        "N well-conn",
    ]

    labels = [parse_expected(r.get("Expected")) for r in rows]
    valid_idxs = [i for i, y in enumerate(labels) if y is not None]

    non_numeric: dict[str, float] = {}
    constants: list[str] = []
    separation: list[dict[str, Any]] = []

    for col in numeric_cols:
        vals_all = [parse_float(rows[i].get(col)) for i in valid_idxs]
        bad = sum(1 for v in vals_all if v is None)
        non_numeric[col] = round(bad / max(1, len(vals_all)), 4)

        vals = [v for v in vals_all if v is not None]
        uniq = sorted(set(round(v, 10) for v in vals))
        if len(uniq) <= 1:
            constants.append(col)

        pos = [vals_all[k] for k in range(len(vals_all)) if labels[valid_idxs[k]] == 1 and vals_all[k] is not None]
        neg = [vals_all[k] for k in range(len(vals_all)) if labels[valid_idxs[k]] == 0 and vals_all[k] is not None]
        d = cohens_d(pos, neg)
        if d is None:
            continue
        separation.append(
            {
                "signal": col,
                "cohens_d": round(d, 4),
                "abs_d": round(abs(d), 4),
                "mean_answerable": round(sum(pos) / max(1, len(pos)), 4) if pos else None,
                "mean_unanswerable": round(sum(neg) / max(1, len(neg)), 4) if neg else None,
                "n": len(vals),
            }
        )
    separation.sort(key=lambda x: x["abs_d"], reverse=True)

    return {
        "non_numeric_rate": non_numeric,
        "constant_or_near_constant_signals": constants,
        "separation_by_d": separation[:20],
    }


def ablation_report(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    base_rows = evaluate_rows(rows)
    base_acc = summarize(base_rows)["metrics"]["accuracy"]
    tests = [
        ("drop_path_bottleneck", {"path_bottleneck"}),
        ("drop_slot_score", {"slot_score"}),
        ("drop_h0_life", {"h0_life"}),
        ("drop_trigger_weight", {"trigger_edge_weight"}),
        ("drop_chain_strength", {"chain_strength"}),
        ("drop_type_entropy", {"type_entropy"}),
        ("drop_sheaf_consistency", {"sheaf_consistent_frac"}),
        ("drop_weighted_sheaf", {"weighted_consistent_frac"}),
        ("drop_weighted_ricci", {"weighted_mean"}),
        ("drop_anchor_bridge", {"anchor_bridge_score"}),
    ]
    out: list[dict[str, Any]] = []
    for name, drop in tests:
        rows_eval = evaluate_rows(rows, drop=drop)
        acc = summarize(rows_eval)["metrics"]["accuracy"]
        out.append(
            {
                "test": name,
                "accuracy": acc,
                "delta_vs_base": round(acc - base_acc, 4),
            }
        )
    out.sort(key=lambda x: x["delta_vs_base"])
    return out


def print_report(report: dict[str, Any], max_errors: int) -> None:
    m = report["summary"]["metrics"]
    c = report["summary"]["confusion"]
    ss = report["summary"]["score_split"]
    print("\n== Topology Scoring Evaluation ==")
    print(f"rows: {report['summary']['n']}")
    print(
        "metrics:",
        f"acc={m['accuracy']}",
        f"prec={m['precision']}",
        f"recall={m['recall']}",
        f"f1={m['f1']}",
        f"auc={m['auc']}",
        f"brier={m['brier']}",
    )
    print("confusion:", c)
    print(
        "score_split:",
        f"ans_mean={ss['mean_answerable']}",
        f"una_mean={ss['mean_unanswerable']}",
        f"mean_threshold={ss['mean_threshold']}",
    )

    print("\nTop Signal Separation (|d|):")
    for item in report["signal_diagnostics"]["separation_by_d"][:10]:
        print(
            f"  {item['signal']}: d={item['cohens_d']}, "
            f"ans={item['mean_answerable']}, una={item['mean_unanswerable']}"
        )

    print("\nMost Harmful Ablations:")
    for item in report["ablation"][:5]:
        print(f"  {item['test']}: acc={item['accuracy']} delta={item['delta_vs_base']}")

    print("\nErrors:")
    errors = report["errors"][:max_errors]
    if not errors:
        print("  none")
    for e in errors:
        print(
            f"  row={e['idx']} q={e['q']} expected={e['expected_label']} "
            f"pred={e['predicted_label']} score={e['score']} thr={e['threshold']} "
            f"evidence={e['evidence_score']} risk={e['risk_score']} conf={e['confidence']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate topology scoring on labeled CSV.")
    parser.add_argument("--csv", required=True, help="Path to labeled CSV.")
    parser.add_argument("--report-json", default=None, help="Optional output JSON report path.")
    parser.add_argument("--errors-csv", default=None, help="Optional output CSV path for error rows.")
    parser.add_argument("--max-errors", type=int, default=20, help="Max errors to print.")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    eval_rows = evaluate_rows(rows)
    summary = summarize(eval_rows)
    diagnostics = build_signal_diagnostics(rows)
    ablation = ablation_report(rows)

    errors: list[dict[str, Any]] = []
    for r in eval_rows:
        if r.expected == r.predicted:
            continue
        errors.append(
            {
                "idx": r.idx,
                "q": r.q,
                "slot_type": r.slot_type,
                "expected_label": "answerable" if r.expected == 1 else "unanswerable",
                "predicted_label": "answerable" if r.predicted == 1 else "unanswerable",
                "score": round(r.score, 4),
                "threshold": round(r.threshold, 4),
                "margin": round(abs(r.score - r.threshold), 4),
                "confidence": r.confidence,
                "evidence_score": round(r.evidence_score, 4) if r.evidence_score is not None else None,
                "risk_score": round(r.risk_score, 4) if r.risk_score is not None else None,
                "disagreement": round(r.disagreement, 4) if r.disagreement is not None else None,
            }
        )

    report = {
        "csv_path": str(csv_path),
        "summary": summary,
        "signal_diagnostics": diagnostics,
        "ablation": ablation,
        "errors": errors,
    }

    print_report(report, max_errors=max(1, args.max_errors))

    if args.report_json:
        out = Path(args.report_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote report JSON: {out}")

    if args.errors_csv:
        out = Path(args.errors_csv).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "idx",
            "q",
            "slot_type",
            "expected_label",
            "predicted_label",
            "score",
            "threshold",
            "margin",
            "confidence",
            "evidence_score",
            "risk_score",
            "disagreement",
        ]
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for e in errors:
                w.writerow(e)
        print(f"Wrote error CSV: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
