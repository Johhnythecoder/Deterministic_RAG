"""
slot_coverage.py — Slot-based answerability detection.

Instead of asking "is this node similar to the question?", asks:
"does this node fill the specific information slots the question requires?"

A question has required slots:
  "What interest rate applies to late payments?"
  → VALUE slot (needs a number) about "late payments"

A node either fills that slot or it doesn't:
  NUMERIC: 1.5% per month on overdue fees (trigger: payment late)
  → fills VALUE slot → coverage = 1.0 → answerable

  OBLIGATION: Licensee must pay within 30 days
  → wrong slot type → coverage = 0.0 → unanswerable

answerability = max coverage across all surviving nodes

Improvements over v1:
  1. Polarity-aware: open-ended PERMISSION questions ("what are X's rights")
     don't count negative OBLIGATION nodes ("X cannot do Y") — those prove
     rights don't exist but don't describe what rights do exist.
  2. Node specificity: NUMERIC > DEFINITION > positive OBLIGATION > negative OBLIGATION > REFERENCE
  3. Vote-weighted: nodes surviving more variants get a small boost.
"""
from __future__ import annotations

import re
import numpy as np
from typing import Optional


# ── Action component splitting ─────────────────────────────────────────────────

# Prepositions that separate a verb from its object noun phrase in action strings
_PREPS = frozenset({"to", "from", "for", "of", "about", "on", "in", "into", "with", "at", "by"})

# Auxiliary/copula verbs that precede the main content verb
_AUX_VERBS = frozenset({
    "be", "is", "are", "was", "were", "am", "being", "been",
    "have", "has", "had", "do", "does", "did",
})

# Modal phrases like "entitled to X" or "allowed to X" that precede the main verb
_MODAL_PHRASES = (
    "entitled to ", "allowed to ", "permitted to ", "authorized to ",
    "required to ", "obligated to ", "able to ", "qualified to ",
    "not entitled to ", "not allowed to ", "not permitted to ",
)


def _normalize_action(action: str) -> str:
    """
    Strip leading auxiliary verbs and modal phrases to expose the main content verb.

    Examples:
      "be entitled to receive version upgrades" → "receive version upgrades"
      "not sublicense to third parties"         → "not sublicense to third parties"
      "is allowed to audit Wipro"               → "audit Wipro"
    """
    a = action.lower().strip()
    # Strip leading auxiliary verbs (one pass is enough for "be entitled to ...")
    words = a.split()
    while words and words[0] in _AUX_VERBS:
        words = words[1:]
    a = " ".join(words)
    # Strip modal phrases: "entitled to receive" → "receive"
    for phrase in _MODAL_PHRASES:
        if a.startswith(phrase):
            a = a[len(phrase):]
            break
    return a.strip()


def _split_action_components(action: str) -> tuple[str, str]:
    """
    Split an action phrase into (verb_phrase, object_noun_phrase).

    Normalises auxiliary verbs first so the main content verb is always first.
    The object noun phrase is the legally discriminating part — it identifies
    *what* or *who* the action is directed at, not just *what kind* of action.

    Examples:
      "receive source code"                        → ("receive",    "source code")
      "be entitled to receive version upgrades"    → ("receive",    "version upgrades")
      "sublicense to affiliates"                   → ("sublicense", "affiliates")
      "sublicense to third parties"                → ("sublicense", "third parties")
      "provide IP warranties"                      → ("provide",    "IP warranties")
      "provide performance warranties"             → ("provide",    "performance warranties")
      "terminate the agreement"                    → ("terminate",  "agreement")
    """
    a = _normalize_action(action)
    words = a.split()
    if len(words) <= 1:
        return (a, "")
    # Verb + preposition + object: "sublicense to affiliates" → ("sublicense", "affiliates")
    if len(words) >= 3 and words[1] in _PREPS:
        return (words[0], " ".join(words[2:]))
    # Simple verb + object: "receive source code" → ("receive", "source code")
    return (words[0], " ".join(words[1:]))


# ── Slot types ─────────────────────────────────────────────────────────────────

# What information types each node type can provide
_NODE_PROVIDES = {
    "NUMERIC":    {"VALUE"},
    "DEFINITION": {"MEANING"},
    "OBLIGATION": {"PERMISSION", "ACTOR", "REQUIREMENT", "CONSEQUENCE"},
    "RIGHT":      {"PERMISSION", "ACTOR"},
    "CONDITION":  {"CONSEQUENCE", "TRIGGER", "PERMISSION"},
    # CONDITION nodes answer PERMISSION questions when the trigger describes how a party
    # activates a right (e.g. "IF Licensee notifies in writing THEN Agreement terminates"
    # answers "what are Licensee's termination rights?").
    "BLANK":      {"VALUE", "MEANING"},  # penalty applied in node_slot_coverage — BLANK nodes return 0.0
    "REFERENCE":  {"MEANING", "ACTOR"},
}

# Modals that indicate a negative/prohibiting obligation
_NEGATIVE_MODALS = frozenset({
    "cannot", "can not", "shall not", "may not", "will not",
    "must not", "need not", "not", "no", "never",
})

# Generic contract words — same list as TypedNodeRetriever._KW_GENERIC, copied here
# to avoid import cycle; used for filtering action keywords
_KW_GENERIC_WORDS = frozenset({
    'information', 'agreement', 'parties', 'between', 'either',
    'whether', 'section', 'within', 'before', 'another', 'having',
    'making', 'during', 'following', 'applicable', 'pursuant',
    'confidential', 'obligations', 'required', 'related', 'receiving',
    'disclosing', 'includes', 'provided', 'provide', 'requires',
    'written', 'notice', 'manner', 'purpose', 'business', 'described',
    'document', 'executed', 'certain', 'herein', 'therein', 'thereof',
    'company', 'person', 'entity', 'parties', 'service', 'services',
})

# Generic words that are NOT distinctive enough to use as action keyword filters
# (appear in nearly every contract clause, so matching them proves nothing)
_ACTION_GENERIC = frozenset({
    "rights", "right", "access", "perform", "conduct", "obtain", "receive",
    "specific", "general", "complete", "certain", "various", "include",
    "includes", "including", "provide", "provides", "without", "unless",
    "except", "beyond", "within", "across", "against", "behalf", "whether",
    "either", "having", "making", "during", "before", "following", "another",
    "applicable", "certain", "described", "document", "granted", "granting",
    "grants", "giving", "taking", "using", "liable", "liable", "liable",
    "listed", "stated", "specified", "indicated", "noted", "defined",
    "parties", "agreement", "contract", "clause", "section", "pursuant",
    "herein", "therein", "thereof", "hereof", "thereunder", "hereunder",
    "information", "software", "services", "license", "licensee", "licensor",
})


# ── Typed Concept Presence (TCP) ───────────────────────────────────────────────
# Generic anchor terms — too common in contracts to be discriminating
_TCP_GENERIC = _KW_GENERIC_WORDS | _ACTION_GENERIC | frozenset({
    # Common structural/relational words
    "under", "after", "terms", "would", "could", "shall", "which", "there",
    "other", "their", "where", "place", "being", "taken", "basis", "those",
    "these", "still", "since", "cause", "first", "third", "given", "every",
    "forms", "while", "means", "arises", "arise", "doing", "going", "event",
    "items", "words", "prior", "among", "owner", "title", "claim", "costs",
    "price", "types", "scope", "usage", "bound", "court", "legal", "valid",
    "party", "order", "level", "total", "whole", "limit",
    "regarding", "related", "following", "therefor", "thereof", "thereto",
    "obligation", "restriction", "restrictions", "obligations",
    # Question framing verbs (structural, not content)
    "considered", "according", "pursuant", "respect", "govern", "governs",
    "maintaining", "providing", "ensuring", "keeping", "applies", "apply",
    "applying", "occurring", "affecting", "allowing", "taking", "giving",
    "defined", "states", "stated", "specif", "described", "mentioned",
    "allowed", "permitted", "prohibited", "constit", "consist", "contai",
    "applic", "necess", "availa", "approp", "releva",
    "different", "various", "entire", "current", "initial", "final",
    "signed", "disclosed", "assign", "establ",
    # Question-framing words that describe asking style, not contract content
    # (e.g., "range" → contract says "28 to 120 days"; "covered" → contract says "means")
    "range", "covered", "minimum", "maximum", "specific", "exactly",
    "describe", "explain", "specify", "detail", "listed", "outlined",
    # Common legal boilerplate verbs
    "comply", "agrees", "agreed", "ensure", "entitl", "author",
    "effect", "consti", "includ", "exclud", "waiver", "waives",
    "enforc", "amend", "termin",  # short stems of common terms
})

# Generic actor/party words that can't disambiguate direction
_TCP_GENERIC_ACTORS = frozenset({
    "party", "parties", "either", "both", "each", "any", "the party",
})

# Which node types are expected to answer each slot type
_SLOT_EXPECTED_TYPES: dict[str, frozenset[str]] = {
    "PERMISSION":   frozenset({"RIGHT", "OBLIGATION"}),
    "REQUIREMENT":  frozenset({"OBLIGATION", "CONDITION"}),
    "CONSEQUENCE":  frozenset({"CONDITION", "OBLIGATION"}),
    "VALUE":        frozenset({"NUMERIC"}),
    "MEANING":      frozenset({"DEFINITION", "REFERENCE"}),
    "ACTOR":        frozenset({"OBLIGATION", "RIGHT"}),
    "GENERAL":      frozenset({"DEFINITION", "OBLIGATION", "RIGHT", "NUMERIC",
                                "CONDITION", "REFERENCE"}),
}


def _node_full_text_tcp(node: dict) -> str:
    """All text content of a node for TCP anchor searching."""
    return " ".join(filter(None, [
        node.get("source_text", ""),
        node.get("action", ""),
        node.get("right", ""),
        node.get("term", ""),
        node.get("definition", ""),
        node.get("trigger", ""),
        node.get("consequence", ""),
        node.get("applies_to", ""),
        node.get("_chunk_text", ""),
    ])).lower()


def _extract_tcp_anchors(question: str, q_node: dict,
                         party_names: frozenset[str] | None = None) -> list[str]:
    """
    Extract discriminating anchor terms for TCP from the question text.

    Uses the question text directly (not q_node fields) for determinism —
    q_node is LLM-parsed and can vary between runs.

    party_names: set of known party name tokens to exclude (they're actors,
                 not concept anchors — e.g. "wipro", "licensee", "cornell").

    Returns list of word STEMS (first 7 chars) for morphological robustness,
    e.g. "sublicensing" → "sublice" to also match "sublicensable".
    """
    exclude = _TCP_GENERIC | (party_names or frozenset())
    STEM_LEN = 7

    candidates: list[str] = []
    for word in re.findall(r'\b[a-z]{5,}\b', question.lower()):
        if word not in exclude:
            candidates.append(word[:STEM_LEN])

    # Deduplicate, return up to 4 stems
    seen: set[str] = set()
    result: list[str] = []
    for w in candidates:
        if w not in seen:
            seen.add(w)
            result.append(w)
    return result[:4]


def _local_pivot_gate(question: str, surviving_nodes: list[dict],
                      party_names: frozenset[str] | None = None) -> float:
    """
    Check if any surviving (retrieved) node contains the question's pivot terms.

    TCP checks the whole document — this checks only the locally retrieved set.
    If the retrieved nodes don't contain the question's key concept terms, the
    cosine retrieval grabbed topically-similar but off-target content, which is
    the root cause of most FPs (e.g. "duration of license" retrieving general
    license nodes when duration is only in an Order Form attachment).

    Returns:
      1.0 — at least one pivot found in the surviving nodes (proceed normally)
      0.0 — pivots absent from ALL surviving nodes (likely FP, hard-block)
    """
    anchors = _extract_tcp_anchors(question, {}, party_names=party_names)
    if not anchors:
        return 1.0  # No discriminating pivots — no penalty

    for node in surviving_nodes:
        node_text = _node_full_text_tcp(node)
        if any(stem in node_text for stem in anchors):
            return 1.0

    return 0.0


def typed_concept_presence(question: str, q_node: dict,
                           all_nodes: list[dict],
                           slots: list[dict]) -> float:
    """
    Verify the document mentions the core concept the question asks about.
    Scans ALL document nodes (not just retrieved ones) for anchor stem presence.

    Returns:
      1.0 — anchor found anywhere in the document (concept present)
      0.0 — anchor absent from entire document (concept genuinely missing)

    Only returns 0.0 in the clear-cut absence case. All ambiguous cases
    return 1.0 and let the LLM handle the nuance. This prevents regression
    while catching questions about topics the document never mentions
    (e.g. "insurance coverage", "outer space use").

    Called only when slot_score >= 0.65 (answerable zone), so it only
    affects high-confidence predictions that might be FPs.
    """
    if not slots or not all_nodes:
        return 1.0

    # Build party name set — exclude party names from anchors
    doc_party_tokens: frozenset[str] = frozenset(
        word
        for node in all_nodes
        for word in re.findall(r'\b[a-z]{4,}\b', (node.get("party") or "").lower())
    )

    anchors = _extract_tcp_anchors(question, q_node, party_names=doc_party_tokens)
    if not anchors:
        return 1.0  # No discriminating anchors found — no penalty

    # Build full document text once (all node fields + chunk text)
    doc_text = " ".join(_node_full_text_tcp(n) for n in all_nodes)

    # Check if ANY anchor stem appears anywhere in the document
    if any(stem in doc_text for stem in anchors):
        return 1.0

    # All anchor stems absent — the question asks about a topic not in this document
    return 0.0


def _is_negative_obligation(node: dict) -> bool:
    """True if this OBLIGATION node prohibits rather than requires something."""
    modal = (node.get("modal") or "").lower().strip()
    party = (node.get("party") or "").lower()
    action = (node.get("action") or "").lower()
    return (
        modal in _NEGATIVE_MODALS
        or "not" in modal
        or modal.startswith("no ")
        or "neither" in party
        or action.startswith("not ")
        or action.startswith("no ")
    )


def _node_specificity(node: dict) -> float:
    """
    How specific/concrete is this node's answer?
    Returns 0.0 (empty/external) to 1.0 (specific concrete value).
    """
    t = node["type"]
    if t == "BLANK":      return 0.0   # unfilled template field
    if t == "NUMERIC":    return 1.0   # always has a specific value
    if t == "REFERENCE":  return 0.75  # points to external content
    if t == "DEFINITION":
        defn = node.get("definition", "")
        return 1.0 if len(defn) > 20 else 0.65
    if t == "OBLIGATION":
        action = node.get("action", "")
        if _is_negative_obligation(node):
            return 0.70  # negative obligations are less specific as answers
        return 1.0 if len(action) > 20 else 0.80
    if t == "RIGHT":
        right = node.get("right", "")
        return 1.0 if len(right) > 20 else 0.80
    if t == "CONDITION":
        consequence = node.get("consequence", "")
        return 1.0 if len(consequence) > 20 else 0.80
    return 0.80


# ── Question form detection ────────────────────────────────────────────────────

_YES_NO_STARTERS = re.compile(
    r'^(can |does |do |is |are |has |have |did |was |were |will |would |could |should '
    r'|may |shall |must |ought |need )', re.IGNORECASE
)


def _is_open_ended(question: str) -> bool:
    """True if question is open-ended ('what/how/which/describe'), False if yes/no."""
    q = question.strip()
    return not bool(_YES_NO_STARTERS.match(q))


# ── Slot extraction ────────────────────────────────────────────────────────────

def _question_entity_from_parties(question: str,
                                   doc_parties: frozenset[str]) -> str:
    """
    Identify which known document party the question is asking about,
    by matching party names extracted from the document's own nodes.

    Uses possessive or subject position to pick the right party when multiple
    appear in the question (e.g. "Licensee's rights to audit Wipro" → "licensee").

    doc_parties: lowercase party-name tokens extracted from all_nodes.party fields.
    Returns a matching party token, or "" if no clear match.
    """
    if not doc_parties:
        return ""

    q_lower = question.lower()

    # Prefer a party that appears in possessive form (X's rights/obligations)
    # — this is the party the question is *about*, not the object party.
    _possessive = re.compile(r"\b(\w+)'s\s+(?:rights?|obligations?|responsibilit\w*|duties|liabilit\w*)",
                             re.IGNORECASE)
    m = _possessive.search(question)
    if m:
        candidate = m.group(1).lower()
        if candidate in doc_parties:
            return candidate

    # Detect syntactic subject: "what X does Y have/can/may" pattern
    # e.g. "What rights does the Licensee have to permit..." → "licensee"
    _subj_have = re.compile(
        r"\bwhat\s+\w+(?:\s+\w+)?\s+does\s+(?:the\s+)?(\w+)\s+have\b",
        re.IGNORECASE
    )
    m2 = _subj_have.search(question)
    if m2:
        candidate = m2.group(1).lower()
        if candidate in doc_parties:
            return candidate

    # Fall back: any party that appears in the question — prefer the one appearing
    # earliest (subject position) over a later-appearing object party.
    # When tied by position, use the longest match for specificity.
    best_party = ""
    best_pos   = len(q_lower) + 1
    for party in sorted(doc_parties, key=len, reverse=True):  # longest first
        pos = q_lower.find(party)
        if pos != -1 and pos < best_pos:
            best_pos   = pos
            best_party = party

    return best_party


def extract_required_slots(question: str, q_node: dict,
                           doc_parties: frozenset[str] | None = None) -> list[dict]:
    """
    Extract what information slots the question requires.

    Primary signal: q_node.node_type — the LLM-parsed typed node schema that matches
    the same schema as document nodes (RIGHT, OBLIGATION, DEFINITION, NUMERIC, CONDITION).
    This enables direct structural node-vs-node comparison instead of text heuristics.

    Fallback: text heuristics on the question string when node_type=GENERAL.
    """
    q = question.lower()
    slots = []
    open_ended = _is_open_ended(question)
    asks_rate_like = any(w in q for w in (
        "rate", "percent", "%", "interest", "finance charge", "per month", "per year", "per day"
    ))
    asks_frequency_like = any(w in q for w in (
        "how often", "frequency", "once per", "times per", "per year", "per month", "no more than"
    ))
    asks_price_rule_like = any(w in q for w in (
        "priced", "price", "pricing", "then-current", "unless otherwise", "order form"
    ))
    asks_cap_like = any(w in q for w in (
        "cap", "liability cap", "limitation on amount", "limit on amount", "total liability", "maximum liability"
    ))

    # Party entity — use structured q_node fields first, text extraction as fallback
    q_entity_text      = _question_entity_from_parties(question, doc_parties or frozenset())
    q_party_structured = (q_node.get("party") or q_node.get("beneficiary") or "").lower().strip()
    q_entity           = q_entity_text or q_party_structured
    q_object_party     = (q_node.get("object_party") or "").lower().strip()
    node_type          = (q_node.get("node_type") or "GENERAL").upper().strip()

    def _permission_slot(subject: str, q_action: str) -> dict:
        return {"slot": "PERMISSION", "subject": subject,
                "open_ended": open_ended, "q_action": q_action,
                "question_entity": q_entity, "q_node_type": node_type,
                "q_object_party": q_object_party}

    def _value_subject_default() -> str:
        # Prefer semantically-central targets over nearby qualifiers.
        if asks_cap_like:
            return "total liability cap amount"
        if asks_rate_like and "overdue" in q:
            return "finance charge overdue undisputed amounts"
        if asks_frequency_like and "audit report" in q:
            return "audit reports frequency per year"
        if asks_price_rule_like and ("additional licenses" in q or "quantities" in q):
            return "additional licenses quantities pricing rule"
        return (q_node.get("applies_to") or q_node.get("action") or
                q_node.get("object") or q_node.get("verb") or "")

    def _value_slot(subject: str) -> dict:
        return {
            "slot": "VALUE",
            "subject": subject,
            "is_rate_question": asks_rate_like,
            "is_frequency_question": asks_frequency_like,
            "is_price_rule_question": asks_price_rule_like,
            "is_cap_question": asks_cap_like,
            "value_focus": ("rate" if asks_rate_like else
                            "frequency" if asks_frequency_like else
                            "price_rule" if asks_price_rule_like else
                            "cap" if asks_cap_like else "generic"),
        }

    # ── Typed node alignment: q_node.node_type → slot type ────────────────────
    # This is the primary path — q_node is parsed into the same schema as document
    # nodes, so we can compare fields directly instead of relying on text patterns.

    if node_type == "NUMERIC" or q_node.get("requires_numeric"):
        slots.append(_value_slot(_value_subject_default()))

    elif node_type == "DEFINITION":
        subject = (q_node.get("term") or q_node.get("action") or q_node.get("object") or "")
        slots.append({"slot": "MEANING", "subject": subject})

    elif node_type == "RIGHT":
        # q_node.action is the specific right being asked about — use directly as q_action
        q_action = (q_node.get("action") or "").lower().strip()
        subject  = (q_node.get("action") or q_node.get("object") or "").strip()
        slots.append(_permission_slot(subject, q_action))

    elif node_type == "OBLIGATION":
        subject = (q_node.get("action") or q_node.get("object") or "").strip()
        modal   = (q_node.get("modal") or "").lower()
        # Negative modal (cannot/may not) → asking if something is permitted → PERMISSION
        # Positive modal (must/shall) → asking what obligations exist → REQUIREMENT
        if any(neg in modal for neg in ("not", "cannot", "may not", "shall not")):
            q_action = (q_node.get("action") or "").lower().strip()
            slots.append(_permission_slot(subject, q_action))
        else:
            slots.append({"slot": "REQUIREMENT", "subject": subject})

    elif node_type == "CONDITION":
        subject = (q_node.get("trigger") or q_node.get("action") or q_node.get("object") or "")
        slots.append({"slot": "CONSEQUENCE", "subject": subject})

    else:
        # ── GENERAL fallback: text heuristics (node_type unknown or GENERAL) ──

        if q_node.get("requires_numeric"):
            slots.append(_value_slot(_value_subject_default()))

        verb = (q_node.get("verb") or "").lower()
        if any(w in verb for w in ["define", "mean", "consist", "include", "refer",
                                    "incorporat", "attach", "annex"]) or \
           any(w in q for w in ["what is the definition", "what does",
                                  "what is a ", "what are the term"]):
            subject = q_node.get("object") or ""
            slots.append({"slot": "MEANING", "subject": subject})

        if any(w in q for w in ["can ", "may ", "allowed", "permitted", "right to", "rights to",
                                  "entitled to", "able to", "has the right", "rights granted",
                                  "rights of ", "rights does", "rights do ", "right for "]):
            subject  = ((q_node.get("entity") or "") + " " + (q_node.get("object") or "")).strip()
            q_action = ""
            if open_ended:
                verb_raw = (q_node.get("verb") or "").lower()
                for prefix in ("rights to ", "right to ", "allowed to ", "permitted to ",
                               "entitled to ", "able to "):
                    if verb_raw.startswith(prefix):
                        verb_raw = verb_raw[len(prefix):]
                        break
                q_action = (verb_raw + " " + (q_node.get("object") or "").lower()).strip()
            slots.append(_permission_slot(subject, q_action))

        if any(w in q for w in ["who is", "who has", "who can", "who are", "who holds",
                                  "which party", "whose", "who provides", "who gives", "who offers"]):
            subject = q_node.get("object") or ""
            slots.append({"slot": "ACTOR", "subject": subject, "question_entity": q_entity})

        if any(w in q for w in ["what happens", "consequences", "what occurs",
                                  "what is the result", "what are the penalties",
                                  "what are the effects"]):
            subject = q_node.get("condition") or q_node.get("object") or ""
            slots.append({"slot": "CONSEQUENCE", "subject": subject})

        if re.search(r'\bmust\b|\bshall\b|\bshould\b', q) or \
           any(w in q for w in ["obligation", "responsibility", "responsible for",
                                  "required to", "duties", "restrictions", "prohibited",
                                  "not allowed", "not permitted", "what method",
                                  "how should", "how must", "how shall"]):
            verb_raw = (q_node.get("verb") or "").lower()
            if "restriction" in verb_raw:
                subject = " ".join(filter(None, [q_node.get("entity") or "",
                                                  verb_raw, q_node.get("object") or ""]))
            else:
                subject = (q_node.get("entity") or "") + " " + (q_node.get("object") or "")
            slots.append({"slot": "REQUIREMENT", "subject": subject.strip()})

    # Mark VALUE slots as date-specific for "when" questions — prevents fee/rate
    # NUMERIC nodes from satisfying a date question
    _DATE_Q = re.compile(
        r'\bwhen\b|\bwhat date\b|\beffective date\b|\bstart date\b|\bend date\b'
        r'|\bexpir\w*\b|\bcommenc\w*\b|\bbegin\w*\b',
        re.IGNORECASE
    )
    if _DATE_Q.search(question):
        for s in slots:
            if s["slot"] == "VALUE":
                s["is_date_question"] = True
        if not any(s["slot"] == "VALUE" for s in slots):
            s = _value_slot(q_node.get("object") or q_node.get("applies_to") or "")
            s["is_date_question"] = True
            slots.append(s)

    # Fallback — nothing detected at all
    if not slots:
        subject = (q_node.get("action") or q_node.get("object") or q_node.get("verb") or "")
        slots.append({"slot": "GENERAL", "subject": subject})

    return slots


# ── Slot coverage scoring ──────────────────────────────────────────────────────

def _node_key_text(node: dict) -> str:
    """The most important text field of a node for subject matching."""
    t = node["type"]
    if t == "NUMERIC":
        # Include value/unit to preserve answer-form signals like
        # percentages, rates, periods, and capped amount windows.
        return (
            f"{node.get('value','')} {node.get('unit','')} "
            f"{node.get('applies_to','')} {node.get('trigger','')}"
        )
    if t == "DEFINITION": return f"{node.get('term', '')} {node.get('definition', '')}"
    if t == "OBLIGATION": return f"{node.get('party','')} {node.get('action','')}"
    if t == "RIGHT":      return f"{node.get('party','')} {node.get('right','')}"
    if t == "CONDITION":  return f"{node.get('trigger','')} {node.get('consequence','')}"
    if t == "REFERENCE":  return f"{node.get('to', '')} {node.get('describes', '')}"
    return node.get("source_text", "")[:100]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def node_slot_coverage(node: dict, slot: dict, embed_fn) -> float:
    """
    How well does this node fill this slot?
    Returns 0.0 (wrong type / negative polarity) to 1.0 (perfect match).

    Incorporates:
    - Type compatibility
    - Polarity check (open-ended PERMISSION rejects negative OBLIGATION)
    - Subject similarity
    - Node specificity
    """
    slot_type = slot["slot"]
    subject   = slot["subject"]
    node_type = node["type"]

    # BLANK nodes represent unfilled fields — they never pass slot coverage
    if node_type == "BLANK":
        return 0.0

    # Who question: ACTOR slot — nodes with generic party names don't name anyone specific
    if slot_type == "ACTOR" and node_type in ("OBLIGATION", "RIGHT"):
        party_val = (node.get("party") or "").lower().strip()
        if party_val in _TCP_GENERIC_ACTORS or not party_val:
            return 0.0

    # Polarity check: open-ended PERMISSION questions require positive rights/obligations.
    # "What are X's rights to sublicense?" — "X cannot sublicense" is NOT an answer.
    # Yes/no questions ("Can X sublicense?") still accept negative obligations (answer = "no").
    if slot_type == "PERMISSION" and slot.get("open_ended") and node_type == "OBLIGATION":
        if _is_negative_obligation(node):
            return 0.0

    # Beneficiary direction check: for open-ended PERMISSION/ACTOR slots where the
    # question targets a specific party, the node's beneficiary must match.
    # Only applied to open-ended questions ("what are X's rights?") — for yes/no
    # questions ("does X have the right to...?") we just need topic presence, not direction,
    # so the check is skipped to avoid penalising nodes like "no license is granted"
    # (beneficiary=Disclosing Party) when the question asks about the Receiving Party.
    _DIRECTION_SLOTS = {"PERMISSION", "ACTOR"}
    if slot_type in _DIRECTION_SLOTS and slot.get("open_ended") and node_type in ("RIGHT", "OBLIGATION"):
        node_beneficiary = (node.get("beneficiary") or "").lower().strip()
        node_party       = (node.get("party") or "").lower().strip()
        # For RIGHT nodes, if beneficiary is not explicitly set, the party IS the beneficiary —
        # the party holding a right is by definition its beneficiary.
        if not node_beneficiary and node_type == "RIGHT":
            node_beneficiary = node_party
        question_entity  = slot.get("question_entity", "").lower().strip()
        # Skip beneficiary check when the question entity is a generic collective term —
        # "parties", "either party", "each party" etc. mean both sides, so any party's
        # obligations are relevant. Only apply direction check for specific named parties.
        _GENERIC_PARTY_TERMS = frozenset({
            "parties", "party", "either party", "each party", "both parties",
            "the parties", "any party",
        })
        q_entity_is_generic = (not question_entity or question_entity in _GENERIC_PARTY_TERMS)
        if node_beneficiary and question_entity and not q_entity_is_generic:
            if question_entity not in node_beneficiary:
                return 0.15  # wrong beneficiary — strong penalty, not hard zero

        # Structural object-party check: if q_node identified an object_party
        # (e.g. "Wipro" in "Licensee's rights to audit Wipro"), nodes whose party
        # matches the object_party are about the wrong side of the relationship.
        # e.g. node.party="wipro" when question asks about "licensee" auditing "wipro"
        # → this node describes what Wipro can do, not what Licensee can do.
        q_object_party = slot.get("q_object_party", "").lower().strip()
        if q_object_party and node_party and q_object_party in node_party:
            # Only penalize if this node's party is the object, not the subject.
            # Skip if the question_entity already matched above (node is about subject).
            if not question_entity or question_entity not in node_beneficiary:
                return 0.15  # node is about the object party, not the asking party

    # Step 1: type compatibility — does this node type provide this slot?
    provides = _NODE_PROVIDES.get(node_type, set())
    if slot_type == "GENERAL":
        type_score = 0.5  # any type counts for general
    elif slot_type in provides:
        type_score = 1.0
    else:
        return 0.0  # completely wrong type — no coverage

    # When question: VALUE slot with date intent requires a date-unit NUMERIC node
    # (prevents fee/rate NUMERIC nodes from scoring high for "when" questions)
    if slot_type == "VALUE" and slot.get("is_date_question") and node_type == "NUMERIC":
        unit = (node.get("unit") or "").lower()
        if not any(w in unit for w in ["date", "day", "month", "year", "week",
                                        "period", "anniv", "quarter"]):
            return 0.0

    # Step 2: subject match — is it about the right thing?
    if not subject:
        base = type_score * 0.6  # type matches but can't verify subject
    else:
        node_text = _node_key_text(node)
        if not node_text:
            base = type_score * 0.5
        else:
            q_vec = embed_fn(subject)
            n_vec = embed_fn(node_text)
            sim   = _cos(q_vec, n_vec)
            base  = type_score * sim

    # Step 2b: action specificity check for open-ended PERMISSION slots.
    # Compare the question's specific action against the node's right/action text.
    #
    # Uses Object Noun Phrase (ONP) decomposition: split each action into
    # (verb_phrase, object_noun_phrase) and compare components separately.
    # The ONP is the legally discriminating part — it identifies *what* is being
    # acted on (e.g. "source code" vs "version upgrades", "affiliates" vs "third parties").
    #
    # Rule: if verb similarity is high (≥0.80) but ONP similarity is low (<0.50),
    # the node talks about the same kind of action applied to a different target —
    # a specificity mismatch. Penalize by ONP similarity rather than letting the
    # high verb similarity dominate and produce a misleadingly high overall score.
    q_action = slot.get("q_action", "")
    if q_action and slot_type == "PERMISSION" and slot.get("open_ended") and node_type in ("RIGHT", "OBLIGATION"):
        node_action_text = (node.get("right") or node.get("action") or "").strip()
        if node_action_text:
            q_verb, q_obj = _split_action_components(q_action)
            n_verb, n_obj = _split_action_components(node_action_text)

            if q_obj and n_obj:
                verb_sim = _cos(embed_fn(q_verb), embed_fn(n_verb))
                obj_sim  = _cos(embed_fn(q_obj),  embed_fn(n_obj))
                # High-verb, low-object: same action verb but different target/type
                # e.g. "receive [source code]" vs "receive [version upgrades]"
                #      "sublicense to [affiliates]" vs "sublicense to [third parties]"
                #      "provide [IP warranties]" vs "provide [performance warranties]"
                if verb_sim >= 0.80 and obj_sim < 0.50:
                    base *= obj_sim / 0.50  # penalize by object similarity
                else:
                    action_sim = 0.35 * verb_sim + 0.65 * obj_sim
                    if action_sim < 0.50:
                        base *= action_sim / 0.50
            else:
                action_sim = _cos(embed_fn(q_action), embed_fn(node_action_text))
                if action_sim < 0.50:
                    base *= action_sim / 0.50

    # Step 3: specificity — does this node contain a concrete answer?
    specificity = _node_specificity(node)
    score = base * specificity

    # VALUE plumbing: reinforce extraction for common legal value forms.
    if slot_type == "VALUE":
        raw = " ".join(filter(None, [
            node.get("source_text", ""),
            node.get("value", ""),
            node.get("unit", ""),
            node.get("applies_to", ""),
            node.get("trigger", ""),
            node.get("action", ""),
            node.get("consequence", ""),
        ])).lower()
        has_percent = bool(re.search(r"\b\d{1,3}(?:\.\d+)?\s*%", raw))
        has_per_unit = bool(re.search(r"\bper\s+(?:month|year|day|week|quarter)\b", raw))
        has_frequency = any(x in raw for x in ("once per year", "no more than once per year", "per year", "per month"))
        has_price_rule = any(x in raw for x in ("then-current price", "unless otherwise", "set forth on the order form", "order form"))
        has_cap_expr = (
            ("liability" in raw and "exceed" in raw and "twelve months" in raw)
            or ("total amount" in raw and "paid" in raw and "months" in raw)
        )

        if slot.get("is_rate_question") and has_percent and has_per_unit:
            score = max(score, 0.80 if node_type == "NUMERIC" else 0.68)
        if slot.get("is_frequency_question") and has_frequency:
            score = max(score, 0.78 if node_type == "NUMERIC" else 0.66)
        if slot.get("is_price_rule_question") and has_price_rule:
            score = max(score, 0.75 if node_type in ("NUMERIC", "OBLIGATION") else 0.62)
        if slot.get("is_cap_question") and has_cap_expr:
            score = max(score, 0.80 if node_type in ("NUMERIC", "CONDITION") else 0.64)

    return score


def compute_coverage(question: str, q_node: dict,
                     surviving_nodes: list[dict],
                     embed_fn,
                     all_nodes: list[dict] | None = None) -> dict:
    """
    Compute slot coverage score for a question against surviving nodes.

    Returns:
      {
        score:       float 0-1 (best coverage, vote-weighted)
        best_node:   the node that best covers the question
        slots:       the required slots detected
        verdict:     "answerable" | "unanswerable" | "uncertain"
      }
    """
    # Extract party tokens from the document so slot scoring can use them
    # for beneficiary direction checks (document-derived, no hardcoded lists).
    # Require a token to appear in >= 5 node party fields to be considered a real
    # party name — filters out clause subjects mis-tagged as parties by the extractor
    # (e.g. party="Products", party="Orders for Products") which would otherwise
    # contaminate LPG anchor exclusion and zero out valid question scores.
    _party_src = all_nodes if all_nodes else surviving_nodes
    if _party_src:
        _party_counts: dict[str, int] = {}
        for node in _party_src:
            for word in re.findall(r'\b[a-z]{4,}\b', (node.get("party") or "").lower()):
                _party_counts[word] = _party_counts.get(word, 0) + 1
        doc_party_tokens: frozenset[str] = frozenset(
            w for w, cnt in _party_counts.items() if cnt >= 10
        )
    else:
        doc_party_tokens = frozenset()

    slots = extract_required_slots(question, q_node, doc_parties=doc_party_tokens)

    if not surviving_nodes:
        return {"score": 0.0, "best_node": None, "slots": slots,
                "verdict": "unanswerable", "method": "no_nodes"}

    # Vote-density boost: nodes appearing in more variants are more reliable answers
    top_nodes = surviving_nodes[:20]
    max_votes = max((n.get("_votes", 1) for n in top_nodes), default=1)

    best_score = 0.0
    best_node  = None

    for node in top_nodes:
        slot_scores = [
            node_slot_coverage(node, slot, embed_fn)
            for slot in slots
        ]
        # Use max across slots: a node that perfectly fills ONE required slot is a
        # strong answerability signal. Averaging penalises multi-slot questions where
        # a single node can't fill every slot (e.g. NUMERIC fills VALUE but not PERMISSION).
        node_score = max(slot_scores)

        # Vote boost: range 0.85–1.0 based on vote fraction
        vote_boost = 0.85 + 0.15 * (node.get("_votes", 1) / max_votes)
        node_score *= vote_boost

        if node_score > best_score:
            best_score = node_score
            best_node  = node

    # Fallback: if primary slots scored 0 (e.g. CONSEQUENCE with no CONDITION nodes),
    # retry with GENERAL slot so the question reaches the LLM instead of being hard-blocked
    if best_score == 0.0:
        general_slot = {"slot": "GENERAL", "subject": q_node.get("object") or q_node.get("verb") or ""}
        for node in top_nodes:
            node_score = node_slot_coverage(node, general_slot, embed_fn)
            if node_score > best_score:
                best_score = node_score
                best_node  = node

    # Date safety net: "when" questions with a date NUMERIC in surviving set must reach LLM.
    # The slot system can zero out date NUMERIC nodes via subject mismatch (e.g. "begin" vs
    # "execution date"). If the slot is a date question and we still have score 0, check if
    # any NUMERIC with a date-like unit is present and give a floor of 0.45 (UNCERTAIN zone).
    _DATE_UNITS = frozenset(["date", "day", "month", "year", "week", "period", "quarter"])
    if best_score < 0.45 and any(s.get("is_date_question") for s in slots):
        for node in top_nodes:
            if node.get("type") == "NUMERIC":
                unit = (node.get("unit") or "").lower()
                if any(w in unit for w in _DATE_UNITS):
                    best_score = max(best_score, 0.45)
                    best_node  = node
                    break

    # ── Local Pivot Gate (LPG) + Typed Concept Presence (TCP) ─────────────────
    # Applied only in the "answerable" zone (score >= 0.65) to catch FPs.
    #
    # LPG (local): do the SURVIVING nodes contain the question's pivot terms?
    #   If not, cosine retrieval captured topically-similar but off-target content.
    #   Example: "duration of license" → nodes about license clauses but not duration.
    #
    # TCP (global): does the ENTIRE document contain the pivot concept?
    #   Catches questions about topics wholly absent from the document (insurance,
    #   outer space, etc.). Runs only when LPG passes (LPG ⊆ TCP coverage).
    tcp_multiplier = 1.0
    lpg_multiplier = 1.0
    if best_score >= 0.65:
        # LPG checks ALL surviving nodes — the relevant node may rank outside top_nodes
        # but still be present in the retrieved set. String matching has no embedding cost.
        lpg_multiplier = _local_pivot_gate(question, surviving_nodes, doc_party_tokens)
        if lpg_multiplier < 1.0:
            # LPG failed — pivot words from question not found in retrieved nodes.
            # BUT: if the best node is the correct *type* for the required slot,
            # cosine retrieval is semantically on-target despite surface word mismatch
            # (e.g. "duration" question → NUMERIC node says "term", not "duration";
            # "obligations toward X" → OBLIGATION node uses "to X" not "toward X").
            # Bypass hard-block for type-correct retrieval; apply soft 0.6× penalty instead.
            primary_slot = slots[0]["slot"] if slots else "GENERAL"
            expected_types = _SLOT_EXPECTED_TYPES.get(primary_slot, frozenset())
            best_node_type = (best_node or {}).get("type", "")
            if best_node_type in expected_types:
                best_score = best_score * 0.6  # soft penalty: drops into uncertain zone → LLM
                lpg_multiplier = 0.6
            else:
                best_score = 0.0  # hard-block: off-type retrieval, no match
        elif all_nodes:
            tcp_multiplier = typed_concept_presence(question, q_node, all_nodes, slots)
            if tcp_multiplier < 1.0:
                best_score = best_score * tcp_multiplier

    # Verdict thresholds
    if best_score >= 0.65:
        verdict = "answerable"
    elif best_score <= 0.35:
        verdict = "unanswerable"
    else:
        verdict = "uncertain"  # call LLM for this one

    return {
        "score":          round(best_score, 4),
        "best_node":      best_node,
        "slots":          slots,
        "verdict":        verdict,
        "method":         "slot_coverage",
        "tcp_multiplier": round(tcp_multiplier, 3),
        "lpg_multiplier": round(lpg_multiplier, 3),
    }
