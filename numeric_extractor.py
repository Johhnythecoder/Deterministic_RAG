"""
numeric_extractor.py — Pre-LLM numeric map extraction from contract text.

Extracts every numeric value (digits, written words, mixed) with:
  - normalized value + unit
  - semantic category (notice_period, payment_term, etc.)
  - full context sentence
  - position in document

Designed to run on raw PDF text before any LLM or chunking.
"""
from __future__ import annotations
import re
import json
from pathlib import Path
from typing import Optional


# ── Word-to-number tables ────────────────────────────────────────────────────

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_MAGNITUDES = {
    "hundred": 100, "thousand": 1_000, "million": 1_000_000,
    "billion": 1_000_000_000,
}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40,
    "fiftieth": 50, "sixtieth": 60, "seventieth": 70, "eightieth": 80,
    "ninetieth": 90, "hundredth": 100, "thousandth": 1000,
}
_FRACTIONS = {
    "half": 0.5, "one-half": 0.5, "one half": 0.5,
    "third": 1/3, "one-third": 1/3, "one third": 1/3,
    "two-thirds": 2/3, "two thirds": 2/3,
    "quarter": 0.25, "one-quarter": 0.25, "one quarter": 0.25,
    "three-quarters": 0.75, "three quarters": 0.75,
}

_ALL_WORDS = set(_ONES) | set(_TENS) | set(_MAGNITUDES) | set(_ORDINALS)


def _word_sequence_to_number(tokens: list[str]) -> Optional[float]:
    """Convert a list of word tokens to a numeric value. Returns None if not parseable."""
    joined = " ".join(tokens).lower()
    # Check fractions first
    if joined in _FRACTIONS:
        return _FRACTIONS[joined]
    for frac, val in _FRACTIONS.items():
        if joined == frac:
            return val

    total = 0
    current = 0
    valid = False
    i = 0
    toks = [t.lower().rstrip("s") for t in tokens]  # "hundreds" → "hundred"

    while i < len(toks):
        tok = toks[i]
        # hyphenated compound: "twenty-five"
        if "-" in tok:
            parts = tok.split("-")
            sub = _word_sequence_to_number(parts)
            if sub is not None:
                current += sub
                valid = True
        elif tok in _ONES:
            current += _ONES[tok]
            valid = True
        elif tok in _ORDINALS:
            current += _ORDINALS[tok]
            valid = True
        elif tok in _TENS:
            current += _TENS[tok]
            valid = True
        elif tok in _MAGNITUDES:
            mag = _MAGNITUDES[tok]
            if mag == 100:
                current *= 100
            else:
                # "two thousand five hundred" → (current + total so far) * mag? No.
                # Standard: current * mag then add to total
                current = (current if current else 1) * mag
                total += current
                current = 0
            valid = True
        elif tok in ("and", "a", "an"):
            pass  # connectors
        else:
            break
        i += 1

    if not valid:
        return None
    return float(total + current)


# ── Unit normalization ───────────────────────────────────────────────────────

_UNIT_MAP = {
    # Time
    "day": "days", "days": "days",
    "business day": "business_days", "business days": "business_days",
    "calendar day": "days", "calendar days": "days",
    "week": "weeks", "weeks": "weeks",
    "month": "months", "months": "months",
    "calendar month": "months", "calendar months": "months",
    "year": "years", "years": "years",
    "calendar year": "years", "calendar years": "years",
    "hour": "hours", "hours": "hours",
    # Money
    "dollar": "USD", "dollars": "USD", "usd": "USD",
    "cent": "cents", "cents": "cents",
    # Percentage
    "percent": "%", "per cent": "%", "percentage": "%",
    # Count
    "time": "times", "times": "times",
    "user": "users", "users": "users",
    "copy": "copies", "copies": "copies",
    "seat": "seats", "seats": "seats",
}

def _normalize_unit(raw: str) -> str:
    return _UNIT_MAP.get(raw.lower().strip(), raw.lower().strip())


# ── Category detection from surrounding context ──────────────────────────────

_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("notice_period",    ["prior written notice", "written notice", "notice of termination",
                          "notify", "days notice", "days' notice", "months notice", "months' notice",
                          "advance notice", "days of notice"]),
    ("cure_period",      ["cure", "remedy", "remediat", "correct the breach", "cure period"]),
    ("late_charge",      ["late charge", "late fee", "interest rate", "penalty interest",
                          "overdue interest", "finance charge", "per month", "per annum",
                          "one and one-half", "1.5%", "accrues at"]),
    ("termination",      ["terminat", "cancel", "wind down", "discontinue"]),
    ("renewal",          ["renew", "auto-renew", "automatic renewal", "renewal term"]),
    ("payment_term",     ["payment", "invoice", "pay within", "due within", "net 30", "net 60",
                          "past due", "remittance", "remit", " shall pay", "must pay",
                          "after the end of each quarter", "days after"]),
    ("liability_cap",    ["liability", "aggregate liability", "maximum liability",
                          "liability cap", "cap on liability", "exceed"]),
    ("term_length",      ["initial term", "term of", "duration", "subscription term",
                          "agreement term", "term shall", "for a period of"]),
    ("threshold",        ["threshold", "minimum", "at least", "no less than",
                          "not less than", "floor", "base amount"]),
    ("revenue_share",    ["revenue share", "revenue split", "share of revenue",
                          "percentage of revenue", "net revenue"]),
    ("arbitration",      ["arbitrat", "adr", "dispute resolution", "arbitrator"]),
    ("audit",            ["audit", "inspection", "examination of records"]),
    ("confidentiality",  ["confidential", "non-disclosure", "nda"]),
    ("warranty",         ["warrant", "warranty", "guarantee", "representation"]),
    ("indemnification",  ["indemnif", "hold harmless", "defend"]),
    ("discount",         ["discount", "reduction", "rebate", "credit"]),
    ("impression",       ["impression", "page view", "page impression", "pageview"]),
    ("user_count",       ["registered user", "active user", "authorized user",
                          "user count", "number of users"]),
    ("penalty",          ["penalty", "liquidated damage", "forfeit"]),
    ("percentage",       ["%", "percent", "per cent"]),  # fallback for bare %
]

def _detect_category(context: str) -> str:
    cl = context.lower()
    for category, keywords in _CATEGORY_RULES:
        if any(kw in cl for kw in keywords):
            return category
    return "general"


# ── Core extraction patterns ─────────────────────────────────────────────────

# Matches written number sequences (up to 5 tokens to avoid over-matching)
_WORD_NUM_RE = re.compile(
    r'\b('
    r'(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)'
    r'(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?'
    r'|one|two|three|four|five|six|seven|eight|nine|ten'
    r'|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen'
    r'|(?:one|two|three|four|five|six|seven|eight|nine)?[ -]?hundred'
    r'(?:[ -](?:and[ -])?(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)'
    r'(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?)?'
    r'|(?:one|two|three|four|five|six|seven|eight|nine|ten|'
    r'twenty|thirty|forty|fifty)[ -]?(?:thousand|million|billion)'
    r'|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth'
    r'|eleventh|twelfth|thirteenth|fourteenth|fifteenth'
    r'|twentieth|thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth'
    r'|one-half|one half|two-thirds|two thirds|three-quarters|three quarters|one-quarter|one quarter'
    r')',
    re.IGNORECASE
)

# Matches digit-based numbers (integers, decimals, comma-separated, currency, percentages)
_DIGIT_RE = re.compile(
    r'(?:'
    r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,4})?'   # $1,000.00
    r'|USD\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,4})?'  # USD 1000
    r'|\d{1,3}(?:,\d{3})+(?:\.\d{1,4})?'         # 1,000,000
    r'|\d+(?:\.\d+)?%'                            # 10.5%
    r'|\d+(?:\.\d+)?'                             # plain number
    r')',
    re.IGNORECASE
)

# Unit patterns that follow a number
_UNIT_RE = re.compile(
    r'\b('
    r'business days?|calendar days?|days?'
    r'|business weeks?|calendar weeks?|weeks?'
    r'|calendar months?|months?'
    r'|calendar years?|fiscal years?|years?'
    r'|hours?|minutes?'
    r'|percent(?:age)?|per cent'
    r'|dollars?|USD|cents?'
    r'|times?|users?|seats?|copies|licen[sc]es?'
    r')',
    re.IGNORECASE
)

# Mixed pattern: "thirty (30)" or "30 (thirty)"
_MIXED_RE = re.compile(
    r'(\b(?:' + '|'.join(list(_ONES) + list(_TENS) + list(_ORDINALS)) + r')\b)'
    r'\s*\(\s*(\d+(?:\.\d+)?)\s*\)'
    r'|\b(\d+(?:\.\d+)?)\s*\(\s*('
    + '|'.join(list(_ONES) + list(_TENS) + list(_ORDINALS))
    + r')\s*\)',
    re.IGNORECASE
)


def _extract_sentence(text: str, pos: int, window: int = 300) -> str:
    """Extract the sentence containing position pos."""
    start = max(0, pos - window)
    end   = min(len(text), pos + window)
    snippet = text[start:end]
    # Find sentence boundaries
    # Look for sentence end before pos
    left_boundary = 0
    for m in re.finditer(r'[.;]\s+[A-Z]', snippet):
        if m.start() < (pos - start):
            left_boundary = m.start() + 1
    # Look for sentence end after pos
    right_boundary = len(snippet)
    for m in re.finditer(r'[.;]\s', snippet):
        if m.start() > (pos - start) + 5:
            right_boundary = m.end()
            break
    return snippet[left_boundary:right_boundary].strip()


# ── Main extractor ───────────────────────────────────────────────────────────

def extract_numerics(text: str) -> list[dict]:
    """
    Extract all numeric values from contract text.
    Returns list of dicts with: value, unit, normalized, category, context, position.
    """
    results: list[dict] = []
    seen_positions: set[int] = set()

    # Precompute section-reference positions to skip
    # Patterns: "2.1  HEADING", "Section 3.2", "Article 4", "Exhibit B", lines starting with "X.Y"
    _section_positions: set[int] = set()
    for sm in re.finditer(
        r'(?:^|\n)\s*(\d+(?:\.\d+)*)\s{2,}[A-Z]'      # "2.1   HEADING"
        r'|(?:section|article|clause|exhibit|schedule|appendix|annex|attachment)\s+(\d+(?:\.\d+)*)',
        text, re.IGNORECASE | re.MULTILINE
    ):
        for g in sm.groups():
            if g:
                gm = re.search(re.escape(g), text[sm.start():sm.end()+5])
                if gm:
                    _section_positions.add(sm.start() + gm.start())

    # Pass 1: digit-based numbers
    for m in _DIGIT_RE.finditer(text):
        pos   = m.start()
        raw   = m.group(0).strip()

        # Parse value
        clean = raw.replace(",", "").replace("$", "").replace("USD", "").strip()
        is_pct = raw.endswith("%")
        try:
            value = float(clean.rstrip("%").strip())
        except ValueError:
            continue

        # Look for unit immediately after (skip closing parens/punctuation)
        unit = ""
        after = re.sub(r'^[\s)\]]+', '', text[m.end():m.end()+50])
        um = _UNIT_RE.match(after)
        if um:
            unit = _normalize_unit(um.group(0))
        elif is_pct:
            unit = "%"

        # Skip section references (2.1, 3.2, etc. — X.Y with single decimal digit)
        if re.fullmatch(r'\d{1,2}\.\d{1,2}', raw):
            continue
        # Skip if position is flagged as a section reference
        if any(abs(pos - sp) < 5 for sp in _section_positions):
            continue
        # Skip numbers immediately followed by 3+ spaces + uppercase (section headers)
        if re.match(r'\s{3,}[A-Z]{2}', text[m.end():m.end()+10]):
            continue
        # Skip bare unitless small integers — likely entity names ("Network 1"),
        # section references ("Section 3"), or version numbers.
        if not unit and not is_pct and value < 2 and not raw.startswith("$"):
            continue
        # Skip unitless numbers that look like pixel dimensions (followed by "x" + digits)
        if not unit and re.match(r'\s*[xX×]\s*\d+', text[m.end():m.end()+10]):
            continue
        # Skip second dimension in pixel specs (e.g. "640x450" — the 450)
        if not unit and re.search(r'\d\s*[xX×]\s*$', text[max(0, pos - 10):pos]):
            continue
        # Skip years in date contexts (1900–2099 with no unit)
        if not unit and not is_pct and 1900 <= value <= 2099:
            continue
        # Skip zip/postal codes (4–7 digit codes with no unit in address context)
        if not unit and not is_pct and re.fullmatch(r'\d{4,7}', raw):
            before_z = text[max(0, pos - 60):pos]
            after_z  = text[m.end():m.end() + 40]
            if re.search(
                r'(?:israel|canada|turkey|ontario|india|australia|uk|england|london'
                r'|[A-Z]{2}\s*$'   # state abbreviation like "IL " or "CA "
                r'|p\.?\s*o\.?\s*box|po box|postal)',
                before_z + " " + after_z, re.IGNORECASE
            ) or (10_000 <= value <= 99_999):
                continue
        # Skip regulatory/rule references: numbers preceded by "Rule" or "§" or "Sec." within 10 chars
        if re.search(r'(?:rule|§|sec\.)\s*$', text[max(0, pos - 15):pos], re.IGNORECASE):
            continue
        # Skip street/address numbers: followed by street suffix within 20 chars
        if not unit and not is_pct and value < 10_000:
            after_addr = text[m.end():m.end() + 25]
            if re.search(
                r'\b(?:street|avenue|boulevard|road|drive|court|lane|place|floor|suite|way)\b'
                r'|(?:^|\s)(?:st|ave|rd|blvd|dr|ct|ln|pl|ste)[.,\s]',
                after_addr, re.IGNORECASE
            ):
                continue
        # Skip numbers immediately followed by a hyphen + time unit (compound adjective "30-day")
        # These will typically also appear as "30 days" in context
        if not unit and re.match(r'\s*-\s*(?:day|week|month|year|hour)s?\b', text[m.end():m.end()+15], re.IGNORECASE):
            continue

        # Skip small unitless integers in date/section/list contexts
        if not unit and not is_pct and 1 <= value <= 99 and value == int(value) and not raw.startswith("$"):
            before = text[max(0, pos - 30):pos]
            after_raw = text[m.end():m.end() + 20]
            # Date component: preceded by month name or "/" or followed by "/"
            if re.search(
                r'(?:january|february|march|april|may|june|july|august|september|october|november|december'
                r'|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)[,\s]*$'
                r'|/\s*$', before, re.IGNORECASE
            ):
                continue
            if after_raw.startswith("/"):
                continue
            # Page number: preceded by "PAGE" or "page"
            if re.search(r'\bpage\s*$', before, re.IGNORECASE):
                continue
            # Numbered list item: number immediately followed by "." + 2+ spaces or tab
            if re.match(r'\.\s{2,}', after_raw) or re.match(r'\.\t', after_raw):
                continue
            # Section/clause reference (Sections 5, 6, 7 and 8 or after section label)
            if re.search(r'\b(?:section|article|clause|exhibit|schedule|appendix|annex|attachment)s?\b',
                         before, re.IGNORECASE):
                continue
            # Preceded immediately by version/rev marker
            if re.search(r'(?:rev|ver|version|revision|page)\s*\.?\s*0*$', before, re.IGNORECASE):
                continue
            # Skip entirely if value < 3 and no compelling unit (too noisy)
            if value < 3:
                continue

        ctx = _extract_sentence(text, pos)
        cat = _detect_category(ctx)

        # Override category for clear percentage values
        if is_pct or unit == "%":
            if cat == "general":
                cat = "percentage"

        # Format normalized string
        if raw.startswith("$") or raw.upper().startswith("USD"):
            norm = f"${value:,.0f}" if value == int(value) else f"${value:,.2f}"
        elif unit == "%":
            norm = f"{value:g}%"
        elif unit:
            norm = f"{value:g} {unit}"
        else:
            norm = f"{value:,.0f}" if value >= 1000 and value == int(value) else f"{value:g}"

        results.append({
            "value":      value,
            "unit":       unit,
            "raw":        raw,
            "normalized": norm,
            "category":   cat,
            "context":    ctx,
            "position":   pos,
            "source":     "digit",
        })
        seen_positions.add(pos)

    # Pass 2: mixed "thirty (30)" patterns — update existing digit entry or add new
    for m in _MIXED_RE.finditer(text):
        pos = m.start()
        # Extract digit value from the match
        word_part = m.group(1) or m.group(4)
        digit_part = m.group(2) or m.group(3)
        if digit_part:
            try:
                value = float(digit_part)
            except ValueError:
                continue
            # Find and enrich the existing digit entry at this position range
            for r in results:
                if abs(r["position"] - pos) < 40 and r["value"] == value:
                    r["written_form"] = word_part
                    r["source"] = "mixed"
                    break

    # Pass 3: written-word numbers not already covered by a nearby digit
    for m in _WORD_NUM_RE.finditer(text):
        pos   = m.start()
        raw   = m.group(0)

        # Skip if there's already a digit-based match nearby (mixed form handled above)
        if any(abs(p - pos) < 60 for p in seen_positions):
            continue

        value = _word_sequence_to_number(raw.split())
        if value is None:
            continue

        # Skip ordinals used as common adjectives ("third party", "second quarter", etc.)
        raw_lower = raw.lower()
        if raw_lower in _ORDINALS or raw_lower in {"first", "second"}:
            after_word = text[m.end():m.end() + 20].lower().strip()
            if re.match(r'(?:party|parties|quarter|time|year|day|month|week|installment|renewal)\b',
                        after_word):
                continue

        # Look for unit after (skip closing parens/punctuation)
        unit = ""
        after = re.sub(r'^[\s)\]]+', '', text[m.end():m.end()+50])
        um = _UNIT_RE.match(after)
        if um:
            unit = _normalize_unit(um.group(0))

        ctx = _extract_sentence(text, pos)
        cat = _detect_category(ctx)

        # Skip word-form numbers with no unit and value < 2 (catches "one of the", "first")
        if not unit and value < 2:
            continue

        results.append({
            "value":      value,
            "unit":       unit,
            "raw":        raw,
            "normalized": f"{value:g} {unit}".strip() if unit else f"{value:g}",
            "category":   cat,
            "context":    ctx,
            "position":   pos,
            "source":     "word",
        })
        seen_positions.add(pos)

    # Sort by document position
    results.sort(key=lambda x: x["position"])
    return results


def build_numeric_map(text: str) -> dict:
    """
    Build a structured numeric map grouped by category.
    This is the artifact attached to each pipeline run.
    """
    entries = extract_numerics(text)

    by_category: dict[str, list[dict]] = {}
    for e in entries:
        cat = e["category"]
        by_category.setdefault(cat, []).append(e)

    # Deduplicate within each category: same value+unit = keep highest-context entry
    deduped: dict[str, list[dict]] = {}
    for cat, items in by_category.items():
        seen: dict[str, dict] = {}
        for item in items:
            key = f"{item['value']}_{item['unit']}"
            if key not in seen or len(item["context"]) > len(seen[key]["context"]):
                seen[key] = item
        deduped[cat] = sorted(seen.values(), key=lambda x: x["value"])

    return {
        "total":       len(entries),
        "by_category": deduped,
        "all":         entries,
    }


def numeric_map_summary(nmap: dict) -> str:
    """Return a compact human-readable summary for debugging."""
    lines = [f"Numeric map: {nmap['total']} values found\n"]
    for cat, items in sorted(nmap["by_category"].items()):
        lines.append(f"  [{cat}]")
        for item in items[:5]:  # cap at 5 per category for display
            lines.append(f"    {item['normalized']:<20}  \"{item['context'][:80]}\"")
    return "\n".join(lines)


# ── PDF text extraction (thin wrapper) ──────────────────────────────────────

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract raw text from PDF using pdfminer or pypdf."""
    pdf_path = Path(pdf_path)
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(pdf_path))
    except ImportError:
        pass
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        pass
    raise RuntimeError("Install pdfminer.six or pypdf to extract PDF text")


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Run on built-in test strings
        TEST_CASES = [
            "Customer shall provide thirty (30) days' prior written notice to terminate this Agreement.",
            "CyberArk may suspend access upon ten (10) days written notice for non-payment.",
            "The initial Term of this Agreement shall be two (2) years from the Effective Date.",
            "A late charge of one and one-half percent (1.5%) per month shall apply to overdue amounts.",
            "The aggregate liability of either party shall not exceed twelve (12) months of fees paid.",
            "Women.com shall pay eDiets within forty-five (45) days after the end of each quarter.",
            "The arbitration costs shall be shared equally between the parties.",
            "Network 1 shall pay Affiliate a fee equal to twenty percent (20%) of net revenue.",
            "Sponsor shall pay an Up-Front Fee of $250,000 upon execution.",
            "The revenue share threshold is $1,000,000 annually.",
            "Either party may terminate upon ninety days' written notice.",
            "CyberArk shall provide three months' prior written notice before any material change.",
            "The cure period for any breach is twenty-one (21) business days.",
            "Maximum liability shall not exceed the greater of $50,000 or fees paid in the prior six months.",
            "Interest accrues at the rate of one percent (1%) per month on outstanding balances.",
            "Snap has thirty (30) days to negotiate with Sponsor before dealing with third parties.",
        ]

        print("═" * 80)
        print("NUMERIC EXTRACTOR — Test Run")
        print("═" * 80)

        all_text = "\n".join(TEST_CASES)
        nmap = build_numeric_map(all_text)

        for i, test in enumerate(TEST_CASES):
            entries = extract_numerics(test)
            print(f"\n[{i+1}] {test[:90]}")
            if entries:
                for e in entries:
                    print(f"     → {e['normalized']:<20} cat={e['category']:<20} src={e['source']}")
            else:
                print("     → (nothing extracted)")

        print("\n" + "═" * 80)
        print(numeric_map_summary(nmap))

    else:
        # Run on a PDF
        pdf = Path(sys.argv[1])
        print(f"Extracting from: {pdf.name}")
        text = extract_text_from_pdf(pdf)
        print(f"Text length: {len(text):,} chars")
        nmap = build_numeric_map(text)
        print(numeric_map_summary(nmap))

        out = pdf.parent / f"{pdf.stem}_numeric_map.json"
        out.write_text(json.dumps(nmap, indent=2, default=str))
        print(f"\nSaved → {out.name}")
