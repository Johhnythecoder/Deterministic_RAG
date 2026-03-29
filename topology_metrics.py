"""
topology_metrics.py — Raw topology / graph math over the retrieved node subgraph.

Computes numbers only — no verdicts or thresholds applied anywhere.
Intended for exploratory analysis and correlation studies.

Metrics returned:
  1. typed_path_coverage  — matched required slot-paths / total required
  2. betti                — β0 (components), β1 (cycles), normalised ratios
  3. persistent_homology  — mean H0/H1 lifetimes from edge-weight filtration
  4. hodge                — gradient vs. harmonic energy fractions of edge flow
  5. sheaf                — fraction of edges where field compatibility holds
  6. conductance          — cut(S) / vol(S) for survivor set S in full graph
"""
from __future__ import annotations

import numpy as np
import re
import os
import json
from collections import defaultdict

_LLM_INTENT_CACHE: dict[tuple[str, str], dict] = {}
_LLM_EVIDENCE_CACHE: dict[tuple[str, str], dict] = {}


# ── Utilities ──────────────────────────────────────────────────────────────────

def _node_score(node: dict) -> float:
    return float(node.get("_score", 0.5))


def _node_key(node: dict) -> str:
    t = node.get("type", "")
    if t == "DEFINITION":
        return f"DEF::{(node.get('term', '') or '').lower()}"
    if t == "OBLIGATION":
        return f"OBL::{(node.get('party', '') or '').lower()}::{(node.get('action', '') or '').lower()[:80]}"
    if t == "RIGHT":
        return f"RIG::{(node.get('party', '') or '').lower()}::{(node.get('right', '') or '').lower()[:80]}"
    if t == "NUMERIC":
        return (
            f"NUM::{node.get('value', '')}::{(node.get('unit', '') or '').lower()}"
            f"::{(node.get('applies_to', '') or '').lower()[:60]}"
        )
    if t == "CONDITION":
        return (
            f"CON::{(node.get('trigger', '') or '').lower()[:60]}"
            f"::{(node.get('consequence', '') or '').lower()[:60]}"
        )
    if t == "BLANK":
        return f"BLK::{(node.get('field', '') or '').lower()}"
    if t == "REFERENCE":
        return f"REF::{(node.get('to', '') or '').lower()}"
    return f"UNK::{str(node)[:80]}"


def _edge_weight(src: dict, dst: dict, etype: str) -> float:
    """
    Edge weight = mean node score × edge-type semantic reliability.
    Scale values match _EDGE_TYPE_DECAY so the Laplacian-based metrics
    (Ricci, VN entropy, heat kernel, etc.) see the same type hierarchy
    as the BFS propagation.
    """
    base = (_node_score(src) + _node_score(dst)) / 2.0
    scale = {
        "TRIGGERS":     0.90,   # causal — highest reliability
        "USES_TERM":    0.82,
        "PARTY_HAS":    0.82,
        "SAME_SECTION": 0.30,   # proximity artifact — low reliability
        "CONTRADICTS":  0.20,   # adversarial link
    }.get(etype, 0.65)
    return base * scale


def _union_find(n: int):
    parent = list(range(n))
    rank   = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    return find, union


def _build_subgraph(survivors: list[dict], graph_index):
    """
    Build node/edge structures for the survivor subgraph.

    Returns:
      n             — number of survivor nodes
      dir_edges     — directed [(i, j, etype, weight)] from graph_index._fwd
      undir_edges   — undirected deduped [(i, j, etype, weight)] for topology
      keys          — [node_key_str] for each survivor
      key_to_idx    — {node_key_str: idx}
    """
    _gkey = _node_key

    keys       = [_gkey(n) for n in survivors]
    key_to_idx = {k: i for i, k in enumerate(keys)}
    n          = len(keys)

    dir_edges:   list[tuple] = []
    undir_seen:  set[tuple]  = set()
    undir_edges: list[tuple] = []

    for i, k in enumerate(keys):
        for (to_key, etype) in graph_index._fwd.get(k, []):
            if to_key in key_to_idx:
                j = key_to_idx[to_key]
                w = _edge_weight(survivors[i], survivors[j], etype)
                dir_edges.append((i, j, etype, w))

                pair = (min(i, j), max(i, j), etype)
                if pair not in undir_seen:
                    undir_seen.add(pair)
                    undir_edges.append((i, j, etype, w))

    return n, dir_edges, undir_edges, keys, key_to_idx


# ── 1. Typed Path Coverage ─────────────────────────────────────────────────────

# Terminal node types that directly satisfy each slot type
_SLOT_TERMINAL: dict[str, frozenset[str]] = {
    "VALUE":       frozenset({"NUMERIC"}),
    "MEANING":     frozenset({"DEFINITION"}),
    "REQUIREMENT": frozenset({"OBLIGATION"}),
    "PERMISSION":  frozenset({"RIGHT", "OBLIGATION"}),
    "CONSEQUENCE": frozenset({"CONDITION", "OBLIGATION"}),
    "ACTOR":       frozenset({"OBLIGATION", "RIGHT"}),
    "GENERAL":     frozenset({"DEFINITION", "OBLIGATION", "RIGHT", "NUMERIC", "CONDITION"}),
}

# Two-hop typed paths (from_type, edge_type, to_type) that also satisfy each slot
_SLOT_PATHS: dict[str, list[tuple[str, str, str]]] = {
    "VALUE":       [("CONDITION",  "TRIGGERS",  "NUMERIC")],
    "REQUIREMENT": [("CONDITION",  "TRIGGERS",  "OBLIGATION")],
    "MEANING":     [("OBLIGATION", "USES_TERM", "DEFINITION"),
                    ("RIGHT",      "USES_TERM", "DEFINITION")],
    "PERMISSION":  [],
    "CONSEQUENCE": [("CONDITION",  "TRIGGERS",  "OBLIGATION")],
    "ACTOR":       [],
    "GENERAL":     [],
}


def _typed_path_coverage(survivors: list[dict], keys: list[str],
                          key_to_idx: dict, dir_edges: list[tuple],
                          slots: list[dict]) -> dict:
    if not slots:
        return {"score": None, "matched": 0, "total": 0, "details": []}

    # Build type → node-index lookup
    idx_by_type: dict[str, set[int]] = defaultdict(set)
    for i, node in enumerate(survivors):
        idx_by_type[node["type"]].add(i)

    # Build (src_idx, etype) → set of reachable dst_idx for two-hop checks
    out_edge_idx: dict[tuple, set[int]] = defaultdict(set)
    for (i, j, etype, _) in dir_edges:
        out_edge_idx[(i, etype)].add(j)

    matched  = 0
    total    = len(slots)
    details  = []

    for slot in slots:
        stype    = slot.get("slot", "GENERAL")
        terminal = _SLOT_TERMINAL.get(stype, frozenset())
        paths    = _SLOT_PATHS.get(stype, [])

        # Direct: any survivor is of a terminal type for this slot
        direct = any(bool(idx_by_type.get(t)) for t in terminal)

        # Two-hop: (from_type, edge_type, to_type) chain present
        path_found = direct
        if not path_found:
            for (ft, et, tt) in paths:
                for i in idx_by_type.get(ft, set()):
                    if any(survivors[j]["type"] == tt for j in out_edge_idx.get((i, et), set())):
                        path_found = True
                        break
                if path_found:
                    break

        if path_found:
            matched += 1

        details.append({"slot": stype, "matched": path_found, "direct": direct})

    score = matched / total if total > 0 else None
    return {
        "score":   round(score, 4) if score is not None else None,
        "matched": matched,
        "total":   total,
        "details": details,
    }


# ── 2. Betti numbers β0, β1 ────────────────────────────────────────────────────

def _betti(n: int, undir_edges: list[tuple]) -> dict:
    if n == 0:
        return {"beta0": 0, "beta1": 0, "beta0_per_v": None, "beta1_per_e": None}

    find, union = _union_find(n)
    for (i, j, etype, w) in undir_edges:
        union(i, j)

    beta0 = len({find(i) for i in range(n)})
    e     = len(undir_edges)
    # Euler characteristic: V - E + F = χ; for graphs F=0 → β1 = E - V + β0
    beta1 = max(0, e - n + beta0)

    return {
        "beta0":       beta0,
        "beta1":       beta1,
        "beta0_per_v": round(beta0 / n, 4),
        "beta1_per_e": round(beta1 / e, 4) if e > 0 else 0.0,
    }


# ── 3. Persistent Homology ─────────────────────────────────────────────────────

def _persistent_homology(n: int, undir_edges: list[tuple]) -> dict:
    """
    Vietoris-Rips filtration over edge weights (descending = high-weight edges added first).

    H0 barcodes:
      All n nodes are born at filtration value 0 as separate components.
      Each edge merge kills one component; lifetime = edge weight at merge.
      The final β0 components are never killed (infinite bar, excluded from mean).

    H1 barcodes (simplified):
      Each non-tree edge in the Kruskal MST construction closes a cycle.
      Birth ≈ the edge weight. Death requires 2-cell coboundaries (not available);
      we report count and mean birth weight as a proxy for cycle persistence.
    """
    if n == 0:
        return {"h0_count": 0, "h0_mean_lifetime": None,
                "h1_count": 0, "h1_mean_birth":    None}

    sorted_edges = sorted(undir_edges, key=lambda x: x[3], reverse=True)

    find, union = _union_find(n)
    h0_lifetimes = []
    h1_births    = []

    for (i, j, etype, w) in sorted_edges:
        if union(i, j):
            h0_lifetimes.append(w)   # merge event: component pair merges at weight w
        else:
            h1_births.append(w)      # redundant edge: closes an independent cycle

    return {
        "h0_count":         len(h0_lifetimes),
        "h0_mean_lifetime": round(float(np.mean(h0_lifetimes)), 4) if h0_lifetimes else None,
        "h1_count":         len(h1_births),
        "h1_mean_birth":    round(float(np.mean(h1_births)),    4) if h1_births    else None,
    }


# ── 4. Hodge Decomposition ─────────────────────────────────────────────────────

def _hodge(n: int, dir_edges: list[tuple], survivors: list[dict]) -> dict:
    """
    Decompose the edge flow vector f into gradient and harmonic components.

    Edge flow f[e] = edge weight (semantic relevance pressure along each edge).

    Signed incidence matrix B1: shape (|E|, |V|)
      B1[e, tail] = -1,  B1[e, head] = +1

    Node Laplacian L0 = B1^T @ B1

    Gradient projection:  f_grad = B1 @ L0^+ @ B1^T @ f
      Measures how much flow is explained by a smooth node-potential function.

    Harmonic residual:    f_harm = f - f_grad
      Flow not explained by gradients — lives in the kernel of B1^T (no 2-cells,
      so coexact/curl component is zero; harmonic = everything non-gradient).

    Fractions: gradient_energy_frac = ||f_grad||² / ||f||²
               harmonic_energy_frac = ||f_harm||² / ||f||²
    """
    e = len(dir_edges)
    if n < 2 or e == 0:
        return {"gradient_energy_frac": None, "harmonic_energy_frac": None, "flow_norm": None}

    B1 = np.zeros((e, n), dtype=float)
    f  = np.zeros(e,      dtype=float)

    for ei, (i, j, etype, w) in enumerate(dir_edges):
        B1[ei, i] = -1.0   # tail
        B1[ei, j] = +1.0   # head
        f[ei]     = w

    flow_norm = float(np.linalg.norm(f))
    if flow_norm < 1e-10:
        return {"gradient_energy_frac": None, "harmonic_energy_frac": None, "flow_norm": 0.0}

    L0      = B1.T @ B1
    L0_pinv = np.linalg.pinv(L0)
    f_grad  = B1 @ (L0_pinv @ (B1.T @ f))
    f_harm  = f - f_grad

    total_energy = float(np.dot(f, f))
    grad_frac    = float(np.dot(f_grad, f_grad)) / total_energy
    harm_frac    = float(np.dot(f_harm, f_harm)) / total_energy

    return {
        "gradient_energy_frac": round(grad_frac, 4),
        "harmonic_energy_frac": round(harm_frac, 4),
        "flow_norm":            round(flow_norm, 4),
    }


# ── 5. Sheaf Consistency ──────────────────────────────────────────────────────

def _sheaf_consistency(survivors: list[dict], dir_edges: list[tuple]) -> dict:
    """
    For each typed directed edge (src → dst), check field compatibility:

      USES_TERM:    dst is DEFINITION; its 'term' should appear in src node's text.
      TRIGGERS:     src is CONDITION; significant words of its 'trigger' should
                    appear in dst node's text.
      SAME_SECTION: src and dst share the same _chunk_id.
      PARTY_HAS:    src 'party' == dst 'party' (if both fields are populated).
      CONTRADICTS:  always inconsistent by definition.
      other:        neutral (counted as consistent).

    Returns consistent_frac = consistent_edges / total_edges_checked.
    """
    def _text(node: dict) -> str:
        return " ".join(filter(None, [
            node.get("source_text", ""),
            node.get("action", ""),
            node.get("right", ""),
            node.get("definition", ""),
            node.get("trigger", ""),
            node.get("consequence", ""),
            node.get("applies_to", ""),
            node.get("term", ""),
        ])).lower()

    if not dir_edges:
        return {"consistent_frac": None, "n_edges_checked": 0, "per_type": {}}

    type_results: dict[str, list[bool]] = defaultdict(list)

    for (i, j, etype, _) in dir_edges:
        src, dst = survivors[i], survivors[j]

        if etype == "USES_TERM":
            term = (dst.get("term") or "").lower().strip()
            ok   = bool(term) and term in _text(src)

        elif etype == "TRIGGERS":
            trigger_words = [w for w in (src.get("trigger") or "").lower().split() if len(w) >= 5]
            dst_text      = _text(dst)
            ok            = any(w in dst_text for w in trigger_words) if trigger_words else False

        elif etype == "SAME_SECTION":
            ok = src.get("_chunk_id") == dst.get("_chunk_id")

        elif etype == "PARTY_HAS":
            sp = (src.get("party") or "").lower()
            dp = (dst.get("party") or "").lower()
            ok = bool(sp) and bool(dp) and (sp == dp or sp in dp or dp in sp)

        elif etype == "CONTRADICTS":
            ok = False

        else:
            ok = True

        type_results[etype].append(ok)

    all_results = [v for vals in type_results.values() for v in vals]
    consistent_frac = sum(all_results) / len(all_results) if all_results else None

    per_type = {
        etype: {
            "consistent": sum(vals),
            "total":      len(vals),
            "frac":       round(sum(vals) / len(vals), 4) if vals else None,
        }
        for etype, vals in type_results.items()
    }

    return {
        "consistent_frac": round(consistent_frac, 4) if consistent_frac is not None else None,
        "n_edges_checked": len(all_results),
        "per_type":        per_type,
    }


# ── 6. Graph Conductance ──────────────────────────────────────────────────────

def _conductance(survivors: list[dict], graph_index) -> dict:
    """
    Conductance of the survivor set S within the full document graph.

      phi(S) = cut(S, V\\S) / vol(S)

    cut(S, V\\S) = edges from S to nodes NOT in S (in full graph).
    vol(S)      = total degree of all nodes in S (in full graph, undirected).

    Low conductance → S is a well-separated island (dense internally, few exits).
    High conductance → loosely attached to the rest of the document graph.
    """
    _gkey = _node_key

    if not survivors:
        return {"conductance": None, "cut": 0, "vol": 0}

    keys_set = {_gkey(node) for node in survivors}
    total_cut = 0
    total_vol = 0

    for node in survivors:
        k        = _gkey(node)
        out_nbrs = graph_index._fwd.get(k, [])
        in_nbrs  = graph_index._rev.get(k, [])
        all_nbrs = out_nbrs + in_nbrs
        total_vol += len(all_nbrs)
        total_cut += sum(1 for (nk, _) in all_nbrs if nk not in keys_set)

    conductance = total_cut / total_vol if total_vol > 0 else None

    return {
        "conductance": round(conductance, 4) if conductance is not None else None,
        "cut":         total_cut,
        "vol":         total_vol,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

# ── 7. Ollivier-Ricci Curvature (approximation) ───────────────────────────────

def _ollivier_ricci(n: int, undir_edges: list[tuple]) -> dict:
    """
    Lin-Lu-Yau approximation of Ollivier-Ricci curvature for each edge.

    For edge (u,v), treat each node's neighborhood as a uniform probability
    measure over its neighbors. The curvature is:

        κ(u,v) = |N(u) ∩ N(v)| * (1/deg_u + 1/deg_v) - 1

    Derivation:
      W_1(m_u, m_v) ≈ 1 - |shared neighbors| * (1/deg_u + 1/deg_v)
      κ = 1 - W_1 = |shared| * (1/deg_u + 1/deg_v) - 1

    Range: -1 (bridge — no shared neighbors, all mass must travel distance ≥2)
           to  1 (clique — all neighbors shared)

    Negative curvature on an edge = bottleneck / bridge between clusters.
    Report: mean, min (most negative = biggest bottleneck), max, std.
    """
    if n < 2 or not undir_edges:
        return {"mean": None, "min": None, "max": None, "std": None,
                "n_negative": 0, "n_edges": 0}

    # Build adjacency sets and degrees
    adj: list[set[int]] = [set() for _ in range(n)]
    for (i, j, _, _) in undir_edges:
        adj[i].add(j)
        adj[j].add(i)

    curvatures = []
    for (i, j, _, _) in undir_edges:
        deg_i = len(adj[i])
        deg_j = len(adj[j])
        if deg_i == 0 or deg_j == 0:
            continue
        shared = len(adj[i] & adj[j])
        kappa = shared * (1.0 / deg_i + 1.0 / deg_j) - 1.0
        curvatures.append(kappa)

    if not curvatures:
        return {"mean": None, "min": None, "max": None, "std": None,
                "n_negative": 0, "n_edges": 0}

    arr = np.array(curvatures)
    return {
        "mean":       round(float(np.mean(arr)), 4),
        "min":        round(float(np.min(arr)),  4),
        "max":        round(float(np.max(arr)),  4),
        "std":        round(float(np.std(arr)),  4),
        "n_negative": int(np.sum(arr < 0)),
        "n_edges":    len(curvatures),
    }


# ── 8. Spectral Gap (Fiedler value) ───────────────────────────────────────────

def _spectral_gap(n: int, undir_edges: list[tuple]) -> dict:
    """
    Fiedler value = second-smallest eigenvalue of the normalised graph Laplacian.

    L = D - A  (combinatorial Laplacian)
    Eigenvalues: 0 = λ1 ≤ λ2 ≤ ... ≤ λn

    λ2 (Fiedler value / algebraic connectivity):
      = 0  → graph is disconnected
      small → easy to split (two weakly connected islands)
      large → highly robust, one tight block

    Also return λ_max for the spectral ratio λ2/λ_max (normalised gap).
    """
    if n < 2:
        return {"fiedler": None, "lambda_max": None, "spectral_ratio": None}

    # Build adjacency matrix and degree matrix
    A = np.zeros((n, n), dtype=float)
    for (i, j, _, w) in undir_edges:
        A[i, j] += w
        A[j, i] += w

    D = np.diag(A.sum(axis=1))
    L = D - A

    try:
        eigenvalues = np.linalg.eigvalsh(L)   # sorted ascending, real
        eigenvalues = np.sort(eigenvalues)
        fiedler   = float(eigenvalues[1])  if n > 1 else None
        lam_max   = float(eigenvalues[-1])
        ratio     = (fiedler / lam_max) if (lam_max > 1e-10 and fiedler is not None) else None
    except np.linalg.LinAlgError:
        return {"fiedler": None, "lambda_max": None, "spectral_ratio": None}

    return {
        "fiedler":       round(fiedler, 4) if fiedler is not None else None,
        "lambda_max":    round(lam_max, 4),
        "spectral_ratio": round(ratio, 4) if ratio is not None else None,
    }


# ── 9. K-Core Decomposition ───────────────────────────────────────────────────

def _kcore(n: int, undir_edges: list[tuple], survivors: list[dict]) -> dict:
    """
    Assign each node its k-core number: the highest k such that the node
    survives after repeatedly pruning nodes with degree < k.

    Algorithm: Batagelj & Zaversnik O(n + m).

    Core number intuition:
      core=1  → peripheral (leaf-like, connected by a single chain)
      core=k  → embedded in a k-clique-like dense subgraph

    Returns: max_core, mean_core, per-type mean cores, and the fraction of
    nodes in the top core (most central).
    """
    if n == 0:
        return {"max_core": None, "mean_core": None, "top_core_frac": None, "per_type": {}}

    degree = [0] * n
    adj: list[list[int]] = [[] for _ in range(n)]
    for (i, j, _, _) in undir_edges:
        adj[i].append(j)
        adj[j].append(i)
        degree[i] += 1
        degree[j] += 1

    # Batagelj-Zaversnik
    core = list(degree)
    order = sorted(range(n), key=lambda x: degree[x])
    pos   = [0] * n
    for idx, v in enumerate(order):
        pos[v] = idx

    for v in order:
        for u in adj[v]:
            if core[u] > core[v]:
                core[u] = max(core[v], core[u] - 1)

    max_core = max(core) if core else 0
    mean_core = float(np.mean(core)) if core else None
    top_core_frac = sum(1 for c in core if c == max_core) / n if n > 0 else None

    # Per node-type mean core
    type_cores: dict[str, list[int]] = defaultdict(list)
    for i, node in enumerate(survivors):
        type_cores[node.get("type", "?")].append(core[i])
    per_type = {t: round(float(np.mean(cs)), 2) for t, cs in type_cores.items()}

    return {
        "max_core":      max_core,
        "mean_core":     round(mean_core, 3) if mean_core is not None else None,
        "top_core_frac": round(top_core_frac, 4) if top_core_frac is not None else None,
        "per_type":      per_type,
    }


# ── 10. Path Entropy ──────────────────────────────────────────────────────────

def _path_entropy(n: int, undir_edges: list[tuple]) -> dict:
    """
    Shannon entropy of the shortest-path length distribution over all
    reachable node pairs.

    Procedure:
      1. BFS from every node → collect all pairwise distances
      2. Build distribution P(length=l) = count(pairs at distance l) / total_pairs
      3. H = -∑ P(l) * log2 P(l)

    Interpretation:
      Low H  → paths concentrate at one or two lengths (clear "golden path")
      High H → paths spread across many lengths (ambiguous, many alternatives)

    Also return: mean_path_length, diameter (longest shortest path).
    """
    from collections import deque

    if n < 2:
        return {"entropy": None, "mean_path_length": None, "diameter": None}

    adj: list[list[int]] = [[] for _ in range(n)]
    for (i, j, _, _) in undir_edges:
        adj[i].append(j)
        adj[j].append(i)

    dist_counts: dict[int, int] = defaultdict(int)
    total_pairs = 0

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if visited[u] == -1:
                    visited[u] = visited[v] + 1
                    dist_counts[visited[u]] += 1
                    total_pairs += 1
                    queue.append(u)

    if total_pairs == 0:
        return {"entropy": None, "mean_path_length": None, "diameter": None}

    # Shannon entropy
    H = -sum((c / total_pairs) * np.log2(c / total_pairs)
             for c in dist_counts.values() if c > 0)

    mean_len = sum(l * c for l, c in dist_counts.items()) / total_pairs
    diameter = max(dist_counts.keys())

    return {
        "entropy":          round(float(H),        4),
        "mean_path_length": round(float(mean_len), 4),
        "diameter":         int(diameter),
    }


# ── 11. Bipartite Projection ──────────────────────────────────────────────────

def _bipartite_projection(survivors: list[dict], dir_edges: list[tuple]) -> dict:
    """
    Project the typed graph onto its bipartite structure:
      Entity nodes: OBLIGATION, RIGHT, CONDITION  (the "clause" layer)
      Term nodes:   DEFINITION, REFERENCE         (the "vocabulary" layer)

    Two entity nodes are "connected" in the projection if they both link
    to the same term node via USES_TERM edges.

    Bipartite density = projected_edges / max_possible_projected_edges
    Low density → clause nodes don't share vocabulary → they're not "talking"
    to each other through defined terms → sheaf consistency is less meaningful.

    Also return: n_entity_nodes, n_term_nodes, projected_edges.
    """
    ENTITY_TYPES = {"OBLIGATION", "RIGHT", "CONDITION"}
    TERM_TYPES   = {"DEFINITION", "REFERENCE"}

    entity_idx = [i for i, s in enumerate(survivors) if s.get("type") in ENTITY_TYPES]
    term_idx   = {i for i, s in enumerate(survivors) if s.get("type") in TERM_TYPES}

    n_entity = len(entity_idx)
    n_term   = len(term_idx)

    # For each term node, collect which entity nodes link to it via USES_TERM
    term_to_entities: dict[int, set[int]] = defaultdict(set)
    for (i, j, etype, _) in dir_edges:
        if etype == "USES_TERM" and j in term_idx and i in set(entity_idx):
            term_to_entities[j].add(i)
        if etype == "USES_TERM" and i in term_idx and j in set(entity_idx):
            term_to_entities[i].add(j)

    # Count distinct projected entity-entity pairs sharing at least one term
    projected_pairs: set[tuple[int, int]] = set()
    for entities in term_to_entities.values():
        elist = sorted(entities)
        for a in range(len(elist)):
            for b in range(a + 1, len(elist)):
                projected_pairs.add((elist[a], elist[b]))

    projected_edges = len(projected_pairs)
    max_possible    = n_entity * (n_entity - 1) // 2 if n_entity > 1 else 0
    density         = projected_edges / max_possible if max_possible > 0 else None

    return {
        "n_entity_nodes":  n_entity,
        "n_term_nodes":    n_term,
        "projected_edges": projected_edges,
        "density":         round(density, 4) if density is not None else None,
    }


# ── 12. Centrality-Weighted Metrics ("Flow Math") ────────────────────────────


# Edge-type decay multipliers for weighted BFS propagation.
# How much weight flows through each edge type when moving outward from anchors.
# SAME_SECTION is proximity-only (chunking artifact) so decays aggressively.
# TRIGGERS / PARTY_HAS / USES_TERM are semantically meaningful — decay slowly.
_EDGE_TYPE_DECAY: dict[str, float] = {
    "TRIGGERS":     0.80,   # causal chain — strongest logical link
    "USES_TERM":    0.72,   # definitional dependency — strong
    "PARTY_HAS":    0.72,   # role/obligation ownership — strong
    "SAME_SECTION": 0.25,   # co-location only, no semantic link — weak
    "CONTRADICTS":  0.15,   # adversarial — barely propagate
}
_EDGE_DECAY_DEFAULT = 0.50


def _anchor_weights(survivors: list[dict], q_node: dict,
                    slots: list[dict], undir_edges: list[tuple]) -> list[float]:
    """
    Assign per-node proximity weights via edge-type weighted propagation
    from question-anchor nodes.

    Anchor detection:
      A survivor is an anchor (weight=1.0) if it shares a meaningful field
      value with q_node OR its type matches the primary slot terminal type.

    Propagation (Dijkstra max-weight):
      Instead of flat BFS hops, weight flows multiplicatively through edges,
      scaled by edge type:
        TRIGGERS / USES_TERM / PARTY_HAS  → strong propagation (0.72–0.80×)
        generic edges                      → moderate (0.50×)
        SAME_SECTION                       → weak (0.25×)  ← chunking artifact
        CONTRADICTS                        → near-zero (0.15×)

      weight[neighbor] = max(weight[neighbor], weight[current] × decay[etype])

    This means a node 2 hops through TRIGGERS edges scores ~0.64, while a node
    2 hops through SAME_SECTION scores ~0.06 — correctly distinguishing causal
    chains from proximity noise.

    Floor: 0.02 for all reachable nodes, 0.01 for disconnected nodes.
    """
    import heapq

    n = len(survivors)
    if n == 0:
        return []

    # --- Step 1: extract q_node fields for anchor detection ---
    _Q_FIELDS = ("party", "term", "action", "applies_to", "field",
                 "describes", "right", "consequence", "trigger")
    q_values: list[str] = []
    for f in _Q_FIELDS:
        v = (q_node.get(f) or "").strip().lower()
        if len(v) >= 4:
            q_values.append(v)

    primary_slot   = (slots[0].get("slot") if slots else None) or "GENERAL"
    expected_types = _SLOT_TERMINAL.get(primary_slot, frozenset())

    def _survivor_text(node: dict) -> str:
        return " ".join(filter(None, [
            node.get("source_text", ""),
            node.get("action",      ""),
            node.get("party",       ""),
            node.get("term",        ""),
            node.get("right",       ""),
            node.get("definition",  ""),
            node.get("trigger",     ""),
            node.get("consequence", ""),
            node.get("applies_to",  ""),
        ])).lower()

    # --- Step 2: identify anchors ---
    is_anchor = [False] * n
    for i, node in enumerate(survivors):
        if node.get("type") in expected_types:
            is_anchor[i] = True
            continue
        if q_values and any(qv in _survivor_text(node) for qv in q_values):
            is_anchor[i] = True

    if not any(is_anchor):
        return [0.1] * n   # fallback: no anchors found, neutral weights

    # --- Step 3: build typed adjacency list [(neighbor, etype)] ---
    adj: list[list[tuple[int, str]]] = [[] for _ in range(n)]
    for (i, j, etype, _w) in undir_edges:
        adj[i].append((j, etype))
        adj[j].append((i, etype))

    # --- Step 4: Dijkstra-style max-weight propagation ---
    # Use a max-heap (negate weights for Python's min-heap)
    weights = [0.01] * n
    for i in range(n):
        if is_anchor[i]:
            weights[i] = 1.0

    # heap entries: (-weight, node_idx)
    heap = [(-weights[i], i) for i in range(n) if is_anchor[i]]
    heapq.heapify(heap)

    while heap:
        neg_w, v = heapq.heappop(heap)
        w_v = -neg_w
        if w_v < weights[v] - 1e-9:
            continue   # stale entry
        for (u, etype) in adj[v]:
            decay   = _EDGE_TYPE_DECAY.get(etype, _EDGE_DECAY_DEFAULT)
            new_w   = w_v * decay
            floor_w = max(new_w, 0.02)   # reachable floor
            if floor_w > weights[u] + 1e-9:
                weights[u] = floor_w
                heapq.heappush(heap, (-floor_w, u))

    return weights


def _weighted_sheaf(survivors: list[dict], dir_edges: list[tuple],
                    node_weights: list[float]) -> dict:
    """
    Sheaf consistency weighted by edge proximity to question anchors.
    Each edge's contribution is scaled by mean(weight[i], weight[j]).
    Returns weighted_consistent_frac = sum(w*ok) / sum(w).
    """
    def _text(node: dict) -> str:
        return " ".join(filter(None, [
            node.get("source_text", ""),
            node.get("action", ""),
            node.get("right", ""),
            node.get("definition", ""),
            node.get("trigger", ""),
            node.get("consequence", ""),
            node.get("applies_to", ""),
            node.get("term", ""),
        ])).lower()

    if not dir_edges:
        return {"weighted_consistent_frac": None, "anchor_weighted_edges": 0,
                "n_anchors": sum(1 for w in node_weights if w >= 1.0)}

    total_w  = 0.0
    consist_w = 0.0

    for (i, j, etype, _) in dir_edges:
        src, dst = survivors[i], survivors[j]
        edge_w = (node_weights[i] + node_weights[j]) / 2.0

        if etype == "USES_TERM":
            term = (dst.get("term") or "").lower().strip()
            ok   = bool(term) and term in _text(src)
        elif etype == "TRIGGERS":
            trigger_words = [w for w in (src.get("trigger") or "").lower().split() if len(w) >= 5]
            dst_text      = _text(dst)
            ok            = any(w in dst_text for w in trigger_words) if trigger_words else False
        elif etype == "SAME_SECTION":
            ok = src.get("_chunk_id") == dst.get("_chunk_id")
        elif etype == "PARTY_HAS":
            sp = (src.get("party") or "").lower()
            dp = (dst.get("party") or "").lower()
            ok = bool(sp) and bool(dp) and (sp == dp or sp in dp or dp in sp)
        elif etype == "CONTRADICTS":
            ok = False
        else:
            ok = True

        total_w  += edge_w
        consist_w += edge_w if ok else 0.0

    frac = consist_w / total_w if total_w > 0 else None
    return {
        "weighted_consistent_frac": round(frac, 4) if frac is not None else None,
        "anchor_weighted_edges":    round(total_w, 3),
        "n_anchors":                sum(1 for w in node_weights if w >= 1.0),
    }


def _weighted_hodge(n: int, dir_edges: list[tuple], node_weights: list[float]) -> dict:
    """
    Hodge decomposition where each edge's flow is scaled by
    mean(weight[i], weight[j]) — anchoring the gradient/harmonic split
    to paths that are relevant to the question.
    """
    e = len(dir_edges)
    if n < 2 or e == 0:
        return {"weighted_gradient_frac": None, "weighted_harmonic_frac": None}

    B1 = np.zeros((e, n), dtype=float)
    f  = np.zeros(e,      dtype=float)

    for ei, (i, j, etype, w) in enumerate(dir_edges):
        proximity = (node_weights[i] + node_weights[j]) / 2.0
        B1[ei, i] = -1.0
        B1[ei, j] = +1.0
        f[ei]     = w * proximity

    flow_norm = float(np.linalg.norm(f))
    if flow_norm < 1e-10:
        return {"weighted_gradient_frac": None, "weighted_harmonic_frac": None}

    L0      = B1.T @ B1
    L0_pinv = np.linalg.pinv(L0)
    f_grad  = B1 @ (L0_pinv @ (B1.T @ f))
    f_harm  = f - f_grad

    total_energy = float(np.dot(f, f))
    grad_frac    = float(np.dot(f_grad, f_grad)) / total_energy
    harm_frac    = float(np.dot(f_harm, f_harm)) / total_energy

    return {
        "weighted_gradient_frac": round(grad_frac, 4),
        "weighted_harmonic_frac": round(harm_frac, 4),
    }


def _weighted_ricci(n: int, undir_edges: list[tuple],
                    node_weights: list[float]) -> dict:
    """
    Ollivier-Ricci curvature weighted by proximity to question anchors.
    Returns weighted mean/min κ — bridges near the question are penalised more.
    """
    if n < 2 or not undir_edges:
        return {"weighted_mean": None, "weighted_min": None, "anchor_bridge_score": None}

    adj: list[set[int]] = [set() for _ in range(n)]
    for (i, j, _, _) in undir_edges:
        adj[i].add(j)
        adj[j].add(i)

    curvatures = []
    weights    = []
    for (i, j, _, _) in undir_edges:
        deg_i = len(adj[i])
        deg_j = len(adj[j])
        if deg_i == 0 or deg_j == 0:
            continue
        shared = len(adj[i] & adj[j])
        kappa  = shared * (1.0 / deg_i + 1.0 / deg_j) - 1.0
        edge_w = (node_weights[i] + node_weights[j]) / 2.0
        curvatures.append(kappa)
        weights.append(edge_w)

    if not curvatures:
        return {"weighted_mean": None, "weighted_min": None, "anchor_bridge_score": None}

    arr  = np.array(curvatures)
    warr = np.array(weights)
    w_mean = float(np.average(arr, weights=warr))

    # anchor_bridge_score: weighted mean κ of negative-curvature edges near anchors
    neg_mask = arr < 0
    anchor_bridge = (float(np.average(arr[neg_mask], weights=warr[neg_mask]))
                     if neg_mask.any() else None)

    # Weighted minimum: most negative curvature weighted by proximity
    # (i.e., a bridge far from the question matters less)
    w_min = float(np.min(arr * warr / (warr.sum() + 1e-10) * len(arr)))

    return {
        "weighted_mean":        round(w_mean, 4),
        "weighted_min":         round(w_min, 4),
        "anchor_bridge_score":  round(anchor_bridge, 4) if anchor_bridge is not None else None,
    }


# ── 13. Directional Mismatch (C) ─────────────────────────────────────────────

def _directional_mismatch(survivors: list[dict], q_node: dict,
                           node_weights: list[float],
                           dir_edges: list[tuple]) -> dict:
    """
    Detect party-role swap errors: "Licensee can audit" vs "Licensor can audit".

    For each anchor node (weight >= 1.0) that carries a party field:
      - MATCH    if node.party contains q_party (same side)
      - MISMATCH if node.party is populated but doesn't contain q_party

    Also checks PARTY_HAS edge direction among anchors:
      A PARTY_HAS edge (i→j) means node-i's party "has" node-j's right/obligation.
      If q_party is found only as the *target* (j) of PARTY_HAS edges, that's a
      role-inversion signal.

    Returns:
      party_match_frac    — anchors where party matches / total party-carrying anchors
      party_mismatch_frac — complement
      edge_role_inversion — fraction of PARTY_HAS edges among anchors where q_party
                            appears only on the receiving end
      q_party_found       — bool, was q_node.party non-empty?
    """
    q_party = (q_node.get("party") or "").strip().lower()

    if not q_party:
        return {
            "q_party_found":       False,
            "party_match_frac":    None,
            "party_mismatch_frac": None,
            "edge_role_inversion": None,
        }

    # Only look at anchor nodes (weight >= 1.0) that have a party field
    anchor_party_nodes: list[tuple[int, str]] = []
    for i, node in enumerate(survivors):
        if node_weights[i] >= 1.0:
            np_ = (node.get("party") or "").strip().lower()
            if np_:
                anchor_party_nodes.append((i, np_))

    if not anchor_party_nodes:
        return {
            "q_party_found":       True,
            "party_match_frac":    None,
            "party_mismatch_frac": None,
            "edge_role_inversion": None,
        }

    matches    = sum(1 for (_, np_) in anchor_party_nodes
                     if q_party in np_ or np_ in q_party)
    mismatches = len(anchor_party_nodes) - matches
    match_frac    = matches    / len(anchor_party_nodes)
    mismatch_frac = mismatches / len(anchor_party_nodes)

    # PARTY_HAS edge direction check among anchor indices
    anchor_idx = {i for (i, _) in anchor_party_nodes}
    party_has_edges = [(i, j) for (i, j, et, _) in dir_edges
                       if et == "PARTY_HAS" and i in anchor_idx and j in anchor_idx]

    edge_role_inversion = None
    if party_has_edges:
        # q_party appears as src → it "has" something (correct subject)
        # q_party appears only as dst → it's the object being "had" (role swap)
        inversion_count = 0
        for (i, j) in party_has_edges:
            src_party = (survivors[i].get("party") or "").lower()
            dst_party = (survivors[j].get("party") or "").lower()
            src_match = q_party in src_party or src_party in q_party
            dst_match = q_party in dst_party or dst_party in q_party
            # inversion: q_party is only the dst (object), not the src (subject)
            if dst_match and not src_match:
                inversion_count += 1
        edge_role_inversion = round(inversion_count / len(party_has_edges), 4)

    return {
        "q_party_found":       True,
        "party_match_frac":    round(match_frac,    4),
        "party_mismatch_frac": round(mismatch_frac, 4),
        "edge_role_inversion": edge_role_inversion,
    }


# ── 14. Condition Chain Strength (B) ─────────────────────────────────────────

def _condition_chain_strength(survivors: list[dict], node_weights: list[float],
                               dir_edges: list[tuple],
                               undir_edges: list[tuple]) -> dict:
    """
    Measures how strongly anchor nodes are connected to CONDITION nodes.

    Legal conditional logic: "You have Right X, PROVIDED that Condition Y."
    If the retrieval pulls the Right node but the Condition node is only weakly
    attached (low-weight bridge edge), the answer is incomplete / risky.

    Algorithm:
      1. Collect anchor indices (weight >= 1.0) and condition indices (type=CONDITION).
      2. Run Dijkstra (max-weight path) from all anchors simultaneously.
         Edge weight = existing edge weight (higher = stronger connection).
      3. For each condition node, record the max-weight path from any anchor.
         chain_strength = min of those per-condition max-weights
         (weakest link in the best path to any condition).
      4. Also check TRIGGERS edges specifically (anchor→CONDITION or CONDITION→anchor).

    Returns:
      n_condition_nodes    — CONDITION nodes in survivor set
      reachable_conditions — conditions reachable from any anchor
      chain_strength       — min(best-path-weight per condition), None if no conditions
      trigger_edge_weight  — mean weight of TRIGGERS edges among anchors
      conditions_required  — True if slots include CONSEQUENCE or REQUIREMENT
    """
    import heapq

    condition_idx = [i for i, s in enumerate(survivors) if s.get("type") == "CONDITION"]
    anchor_idx    = {i for i, w in enumerate(node_weights) if w >= 1.0}
    n = len(survivors)

    # Determine if this question type even requires conditions
    # (CONSEQUENCE/REQUIREMENT slots care about conditionals)
    slot_types_present = {s.get("type") for s in survivors}
    conditions_required = bool(condition_idx)  # report if any conditions exist

    if not condition_idx:
        return {
            "n_condition_nodes":    0,
            "reachable_conditions": 0,
            "chain_strength":       None,
            "trigger_edge_weight":  None,
            "conditions_required":  False,
        }

    if not anchor_idx:
        return {
            "n_condition_nodes":    len(condition_idx),
            "reachable_conditions": 0,
            "chain_strength":       None,
            "trigger_edge_weight":  None,
            "conditions_required":  conditions_required,
        }

    # Build weighted adjacency (undirected for reachability, use edge weight)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j, _, w) in undir_edges:
        adj[i].append((j, w))
        adj[j].append((i, w))

    # Dijkstra for maximum-weight path (negate weights to use min-heap)
    best = [-0.0] * n  # best path weight from any anchor to node i
    heap: list[tuple[float, int]] = []
    for a in anchor_idx:
        best[a] = 1.0  # anchors start at full strength
        heapq.heappush(heap, (-1.0, a))

    while heap:
        neg_w, v = heapq.heappop(heap)
        cur_w = -neg_w
        if cur_w < best[v]:
            continue
        for (u, ew) in adj[v]:
            path_w = min(cur_w, ew)  # bottleneck path (weakest link)
            if path_w > best[u]:
                best[u] = path_w
                heapq.heappush(heap, (-path_w, u))

    # Condition node strengths
    condition_strengths = [best[i] for i in condition_idx if best[i] > 0]
    reachable = len(condition_strengths)
    chain_strength = min(condition_strengths) if condition_strengths else None

    # TRIGGERS edge weights specifically (most direct condition link)
    trigger_weights = [w for (i, j, et, w) in dir_edges
                       if et == "TRIGGERS"
                       and (i in anchor_idx or j in anchor_idx)]
    trigger_edge_weight = (round(float(np.mean(trigger_weights)), 4)
                           if trigger_weights else None)

    return {
        "n_condition_nodes":    len(condition_idx),
        "reachable_conditions": reachable,
        "chain_strength":       round(chain_strength, 4) if chain_strength is not None else None,
        "trigger_edge_weight":  trigger_edge_weight,
        "conditions_required":  conditions_required,
    }


# ── 15. Global Efficiency on Largest Connected Component ─────────────────────

def _global_efficiency_lcc(n: int, undir_edges: list[tuple]) -> dict:
    """
    Global efficiency computed on the Largest Connected Component (LCC) only.

    E_LCC = 1/(n_lcc*(n_lcc-1)) * Σ_{i≠j ∈ LCC} 1/d(i,j)

    Computed on LCC because all retrieved subgraphs are disconnected (Fiedler=0);
    measuring efficiency across disconnected fragments would give meaningless inf terms.

    High E → short paths between all nodes in the main cluster → logical compactness.
    Low E  → long detours, many bridges, fragmented reasoning.
    Target: E > 0.45 correlates with successful answers.
    """
    from collections import deque

    if n < 2:
        return {"efficiency": None, "n_lcc": n, "lcc_frac": 1.0 if n == 1 else None}

    adj: list[list[int]] = [[] for _ in range(n)]
    for (i, j, _, _) in undir_edges:
        adj[i].append(j)
        adj[j].append(i)

    # Find all components via BFS
    comp_id = [-1] * n
    components: list[list[int]] = []
    for start in range(n):
        if comp_id[start] != -1:
            continue
        comp: list[int] = []
        queue: deque[int] = deque([start])
        comp_id[start] = len(components)
        while queue:
            v = queue.popleft()
            comp.append(v)
            for u in adj[v]:
                if comp_id[u] == -1:
                    comp_id[u] = len(components)
                    queue.append(u)
        components.append(comp)

    lcc = max(components, key=len)
    n_lcc = len(lcc)

    if n_lcc < 2:
        return {"efficiency": 0.0, "n_lcc": n_lcc, "lcc_frac": round(n_lcc / n, 4)}

    lcc_set = set(lcc)

    # BFS from each LCC node, sum 1/d(i,j) for all reachable j
    total_inv = 0.0
    for start in lcc:
        dist: dict[int, int] = {start: 0}
        queue2: deque[int] = deque([start])
        while queue2:
            v = queue2.popleft()
            for u in adj[v]:
                if u in lcc_set and u not in dist:
                    dist[u] = dist[v] + 1
                    total_inv += 1.0 / dist[u]
                    queue2.append(u)

    efficiency = total_inv / (n_lcc * (n_lcc - 1))

    return {
        "efficiency": round(efficiency, 4),
        "n_lcc":      n_lcc,
        "lcc_frac":   round(n_lcc / n, 4),
    }


# ── 16. Type-Level Metrics ────────────────────────────────────────────────────

def _type_metrics(survivors: list[dict]) -> dict:
    """
    Obligation fraction and Shannon type entropy.

    obl_frac    = n_OBLIGATION / n_total
      Calibration: correct answerable = 0.24, wrong = 0.17, unanswerable = 0.13.
      Low obl_frac → system retrieved vocabulary (definitions) but not rules.

    type_entropy = -Σ P(t)*log2(P(t))
      Low entropy (<1.5) → monolithic retrieval (e.g. 20 definitions, 0 obligations).
      High entropy (>2.0) → diverse mix of node types, richer context.
    """
    if not survivors:
        return {"obl_frac": None, "type_entropy": None, "type_counts": {}}

    counts: dict[str, int] = defaultdict(int)
    for node in survivors:
        counts[node.get("type", "?")] += 1

    total = len(survivors)
    obl_frac = round(counts.get("OBLIGATION", 0) / total, 4)
    H = -sum((c / total) * np.log2(c / total) for c in counts.values() if c > 0)

    return {
        "obl_frac":    obl_frac,
        "type_entropy": round(float(H), 4),
        "type_counts":  dict(counts),
    }


# ── 17. Clustering Coefficient ────────────────────────────────────────────────

def _clustering_coefficient(n: int, undir_edges: list[tuple]) -> dict:
    """
    Local clustering coefficient (LCC) averaged over nodes with degree ≥ 2.

    For node v:  CC(v) = (# edges among N(v)) / C(deg(v), 2)

    High mean CC → interlocking facts (Fact A, B, C all point to each other).
      Triangulated corroboration is harder for the LLM to hallucinate away.
    Low mean CC  → star/chain topology — one central node, isolated spokes.

    Also returns:
      global_cc  = 3·n_triangles / n_connected_triples  (Watts-Strogatz)
      n_triangles = number of closed triangles in the subgraph
    """
    if n < 3 or not undir_edges:
        return {"mean_local_cc": None, "global_cc": None, "n_triangles": 0}

    adj: list[set[int]] = [set() for _ in range(n)]
    for (i, j, _, _) in undir_edges:
        adj[i].add(j)
        adj[j].add(i)

    local_ccs: list[float] = []
    tri_sum = 0  # sum over nodes of closed triangles at v

    for v in range(n):
        deg = len(adj[v])
        if deg < 2:
            continue
        nbrs = list(adj[v])
        closed = sum(1 for a in range(len(nbrs))
                     for b in range(a + 1, len(nbrs))
                     if nbrs[b] in adj[nbrs[a]])
        tri_sum += closed
        local_ccs.append(2 * closed / (deg * (deg - 1)))

    n_triangles = tri_sum // 3  # each triangle counted 3× (once per vertex)
    mean_cc = float(np.mean(local_ccs)) if local_ccs else None

    # Global CC (Watts-Strogatz)
    total_triples = sum(len(a) * (len(a) - 1) // 2 for a in adj)
    global_cc = (3 * n_triangles / total_triples) if total_triples > 0 else None

    return {
        "mean_local_cc": round(mean_cc, 4) if mean_cc is not None else None,
        "global_cc":     round(global_cc, 4) if global_cc is not None else None,
        "n_triangles":   n_triangles,
    }


# ── 18. Effective Resistance between anchor nodes ─────────────────────────────

def _effective_resistance(n: int, undir_edges: list[tuple],
                           node_weights: list[float]) -> dict:
    """
    Effective resistance R_ij = (e_i - e_j)^T L^+ (e_i - e_j) for each pair
    of anchor nodes (weight >= 1.0) via the Moore-Penrose pseudoinverse of L.

    Low mean_resistance = anchors connected by many parallel paths (robust).
    High mean_resistance = anchors only reachable through single bridges (fragile).
    """
    if n < 2:
        return {"mean_resistance": None, "max_resistance": None,
                "n_anchor_pairs": 0, "resistances": []}

    anchor_ids = [i for i, w in enumerate(node_weights) if w >= 1.0]
    if len(anchor_ids) < 2:
        return {"mean_resistance": None, "max_resistance": None,
                "n_anchor_pairs": 0, "resistances": []}

    # Build combinatorial Laplacian (unweighted — consistent with spectral metric)
    L = np.zeros((n, n))
    for (i, j, _etype, _w) in undir_edges:
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1

    try:
        L_plus = np.linalg.pinv(L)
    except np.linalg.LinAlgError:
        return {"mean_resistance": None, "max_resistance": None,
                "n_anchor_pairs": 0, "resistances": []}

    resistances = []
    for a in range(len(anchor_ids)):
        for b in range(a + 1, len(anchor_ids)):
            i, j = anchor_ids[a], anchor_ids[b]
            r = float(L_plus[i, i] + L_plus[j, j] - 2 * L_plus[i, j])
            resistances.append(round(max(0.0, r), 4))

    if not resistances:
        return {"mean_resistance": None, "max_resistance": None,
                "n_anchor_pairs": 0, "resistances": []}

    return {
        "mean_resistance": round(float(np.mean(resistances)), 4),
        "max_resistance":  round(float(np.max(resistances)),  4),
        "n_anchor_pairs":  len(resistances),
        "resistances":     resistances,
    }


# ── 19. Betweenness centrality of anchor nodes ────────────────────────────────

def _anchor_betweenness(n: int, undir_edges: list[tuple],
                         node_weights: list[float]) -> dict:
    """
    For each anchor node (weight >= 1.0), compute its betweenness centrality
    in the retrieved subgraph: fraction of all-pairs shortest paths that pass
    through this node.

    High mean anchor betweenness = anchors are structural bridges → fragile.
    Low = anchors are embedded in a dense neighborhood → robust.
    """
    if n < 3:
        return {"mean_anchor_bc": None, "max_anchor_bc": None, "anchor_bc": []}

    anchor_ids = set(i for i, w in enumerate(node_weights) if w >= 1.0)
    if not anchor_ids:
        return {"mean_anchor_bc": None, "max_anchor_bc": None, "anchor_bc": []}

    # Build adjacency list
    adj: list[list[int]] = [[] for _ in range(n)]
    for (i, j, _etype, _w) in undir_edges:
        adj[i].append(j)
        adj[j].append(i)

    # BFS-based betweenness (Brandes algorithm, unweighted)
    betweenness = [0.0] * n
    from collections import deque

    for s in range(n):
        stack = []
        pred  = [[] for _ in range(n)]
        sigma = [0] * n;  sigma[s] = 1
        dist  = [-1]  * n; dist[s]  = 0
        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    q.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = [0.0] * n
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # Normalise by (n-1)(n-2)/2 for undirected
    norm = (n - 1) * (n - 2) / 2.0 if n > 2 else 1.0
    bc_norm = [round(b / norm, 4) for b in betweenness]

    anchor_bc = [bc_norm[i] for i in sorted(anchor_ids)]
    if not anchor_bc:
        return {"mean_anchor_bc": None, "max_anchor_bc": None, "anchor_bc": []}

    return {
        "mean_anchor_bc": round(float(np.mean(anchor_bc)), 4),
        "max_anchor_bc":  round(float(np.max(anchor_bc)),  4),
        "anchor_bc":      anchor_bc,
    }


# ── 20. Forman-Ricci curvature ────────────────────────────────────────────────

def _forman_ricci(n: int, undir_edges: list[tuple]) -> dict:
    """
    Edge-only Forman-Ricci curvature (unweighted version):
        F(u,v) = 4 - deg(u) - deg(v)

    This is the combinatorial Forman formula with uniform edge/vertex weights.
    Range: very negative for high-degree hubs; near 0 or positive for sparse graphs.

    More negative mean = more hub-dominated (possibly bridge-heavy).
    Compare against Ollivier-Ricci to see which signal separates correct/wrong better.
    """
    if not undir_edges:
        return {"mean": None, "min": None, "max": None, "n_edges": 0}

    # Degree count
    deg = [0] * n
    for (i, j, _etype, _w) in undir_edges:
        deg[i] += 1
        deg[j] += 1

    curvatures = []
    for (i, j, _etype, _w) in undir_edges:
        f = 4 - deg[i] - deg[j]
        curvatures.append(float(f))

    return {
        "mean":   round(float(np.mean(curvatures)),   4),
        "min":    round(float(np.min(curvatures)),    4),
        "max":    round(float(np.max(curvatures)),    4),
        "n_edges": len(curvatures),
    }


# ── 21. Type transition entropy ───────────────────────────────────────────────

def _type_transition_entropy(survivors: list[dict], dir_edges: list[tuple]) -> dict:
    """
    For each directed edge, record the ordered type pair (src_type → dst_type).
    Compute Shannon entropy over the distribution of type pairs.

    Low entropy = monolithic transitions (e.g. all OBL→OBL).
    High entropy = diverse multi-type reasoning chain.
    Hypothesis: correct answers for complex questions need high transition entropy.
    """
    if not dir_edges:
        return {"entropy": None, "n_pairs": 0, "top_transitions": []}

    from collections import Counter
    pair_counts: Counter = Counter()
    for (i, j, _etype, _w) in dir_edges:
        src_t = survivors[i].get("type", "UNKNOWN")
        dst_t = survivors[j].get("type", "UNKNOWN")
        pair_counts[(src_t, dst_t)] += 1

    total = sum(pair_counts.values())
    probs = [c / total for c in pair_counts.values()]
    entropy = float(-sum(p * np.log2(p) for p in probs if p > 0))

    top = sorted(pair_counts.items(), key=lambda x: -x[1])[:5]
    top_transitions = [{"from": a, "to": b, "count": c} for (a, b), c in top]

    return {
        "entropy":         round(entropy, 4),
        "n_pairs":         len(pair_counts),
        "top_transitions": top_transitions,
    }


# ── 22. Weighted Fiedler (anchor-weight Laplacian) ────────────────────────────

def _weighted_fiedler(n: int, undir_edges: list[tuple],
                       node_weights: list[float]) -> dict:
    """
    Build a node-weighted Laplacian where diagonal entries use anchor weights
    instead of plain degree. This rescues the spectral gap signal: the unweighted
    Fiedler is always 0 (all subgraphs disconnected), but the weighted version
    can be non-zero, measuring how well-anchored the connected components are.

    λ2_weighted > 0 means the weighted graph is "connected enough" in the
    anchor-weight sense; higher = more robustly connected from anchor perspective.
    """
    if n < 2 or not undir_edges:
        return {"lambda2_weighted": None, "spectral_gap_weighted": None}

    w = np.array(node_weights, dtype=float)

    # Weighted Laplacian: L_w[i,i] = w[i], L_w[i,j] = -edge_weight * sqrt(w[i]*w[j])
    L = np.zeros((n, n))
    for (i, j, _etype, ew) in undir_edges:
        coupling = ew * np.sqrt(max(w[i], 1e-6) * max(w[j], 1e-6))
        L[i, i] += coupling
        L[j, j] += coupling
        L[i, j] -= coupling
        L[j, i] -= coupling

    try:
        eigvals = np.linalg.eigvalsh(L)
        eigvals_sorted = np.sort(eigvals)
        # λ0 ≈ 0 (or negative due to floating point), λ1 is Fiedler
        nz = eigvals_sorted[eigvals_sorted > 1e-8]
        lambda2 = float(nz[0]) if len(nz) > 0 else 0.0
        gap     = float(nz[1] - nz[0]) if len(nz) > 1 else 0.0
    except np.linalg.LinAlgError:
        return {"lambda2_weighted": None, "spectral_gap_weighted": None}

    return {
        "lambda2_weighted":     round(lambda2, 4),
        "spectral_gap_weighted": round(gap,    4),
    }


# ── 23. Anchor neighborhood overlap (Jaccard) ────────────────────────────────

def _anchor_jaccard(n: int, undir_edges: list[tuple],
                     node_weights: list[float]) -> dict:
    """
    For each pair of anchor nodes (weight >= 1.0), compute Jaccard similarity
    of their 1-hop neighborhoods (excluding the anchors themselves).

    High mean Jaccard = anchors are redundant (retrieved same region twice).
    Low = anchors are diverse, covering different document regions.
    Redundant anchors signal retrieval only found one relevant region.
    """
    if n < 2:
        return {"mean_jaccard": None, "min_jaccard": None,
                "n_anchor_pairs": 0, "jaccards": []}

    anchor_ids = [i for i, w in enumerate(node_weights) if w >= 1.0]
    if len(anchor_ids) < 2:
        return {"mean_jaccard": None, "min_jaccard": None,
                "n_anchor_pairs": 0, "jaccards": []}

    # Build neighbor sets (1-hop, excluding self)
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for (i, j, _etype, _w) in undir_edges:
        neighbors[i].add(j)
        neighbors[j].add(i)

    jaccards = []
    for a in range(len(anchor_ids)):
        for b in range(a + 1, len(anchor_ids)):
            i, j = anchor_ids[a], anchor_ids[b]
            ni = neighbors[i] - {j}
            nj = neighbors[j] - {i}
            inter = len(ni & nj)
            union = len(ni | nj)
            jac = inter / union if union > 0 else 0.0
            jaccards.append(round(jac, 4))

    if not jaccards:
        return {"mean_jaccard": None, "min_jaccard": None,
                "n_anchor_pairs": 0, "jaccards": []}

    return {
        "mean_jaccard":  round(float(np.mean(jaccards)), 4),
        "min_jaccard":   round(float(np.min(jaccards)),  4),
        "n_anchor_pairs": len(jaccards),
        "jaccards":       jaccards,
    }


# ── 25. Von Neumann Entropy of graph density matrix ──────────────────────────

def _von_neumann_entropy(n: int, undir_edges: list[tuple]) -> dict:
    """
    Treat the normalised combinatorial Laplacian as a quantum density matrix:
        ρ = L / Tr(L)   (Tr(L) = 2|E| for unweighted graphs)

    Von Neumann entropy:
        S = -Tr(ρ log ρ) = -Σ λ̃ᵢ log λ̃ᵢ   where λ̃ᵢ = λᵢ / Σλᵢ

    Interpretation:
        Low S  → ordered, structured retrieval (few dominant modes)
        High S → thermodynamically "hot", diffuse, random-looking subgraph

    Hypothesis: correct answers have lower VN entropy (more structured).
    """
    if n < 2 or not undir_edges:
        return {"entropy": None, "normalised_entropy": None, "n_nonzero": 0}

    L = np.zeros((n, n))
    for (i, j, _etype, _w) in undir_edges:
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1

    trace_L = float(np.trace(L))
    if trace_L < 1e-10:
        return {"entropy": None, "normalised_entropy": None, "n_nonzero": 0}

    try:
        eigvals = np.linalg.eigvalsh(L)
    except np.linalg.LinAlgError:
        return {"entropy": None, "normalised_entropy": None, "n_nonzero": 0}

    # Density matrix eigenvalues: ρ̃ᵢ = λᵢ / Tr(L)
    rho = eigvals / trace_L
    # Only positive eigenvalues contribute
    pos = rho[rho > 1e-12]
    n_nonzero = int(len(pos))
    if not len(pos):
        return {"entropy": 0.0, "normalised_entropy": 0.0, "n_nonzero": 0}

    entropy = float(-np.sum(pos * np.log(pos)))
    # Normalise by log(n_nonzero) so S ∈ [0,1]
    max_entropy = float(np.log(n_nonzero)) if n_nonzero > 1 else 1.0
    normalised  = round(entropy / max_entropy, 4) if max_entropy > 1e-12 else 0.0

    return {
        "entropy":            round(entropy,   4),
        "normalised_entropy": normalised,
        "n_nonzero":          n_nonzero,
    }


# ── 26. Non-backtracking (Hashimoto) spectral radius ─────────────────────────

def _nonbacktracking_radius(n: int, undir_edges: list[tuple]) -> dict:
    """
    The non-backtracking (Hashimoto) matrix B is defined on *directed* edge pairs:
        For each undirected edge {u,v} we create two directed edges (u→v) and (v→u).
        B[(u→v), (w→x)] = 1   iff  v == w  AND  u != x   (no immediate backtrack)

    The spectral radius ρ(B) is a graph invariant that:
      - Avoids the tree-back-propagation noise that plagues the adjacency spectrum
      - On a d-regular Ramanujan graph, ρ(B) = √(d-1)  (theoretical optimum)
      - The ratio ρ(B) / √(mean_deg - 1) measures how close to optimal the
        information propagation structure is.

    High ratio → efficient multi-path information flow (robust retrieval).
    Low ratio  → sparse, chain-like, fragile.
    """
    if n < 2 or len(undir_edges) < 2:
        return {"spectral_radius": None, "ramanujan_ratio": None, "n_dir_edges": 0}

    # Build directed edge list: each undirected edge → two directed edges
    dir_pairs: list[tuple[int, int]] = []
    for (i, j, _etype, _w) in undir_edges:
        dir_pairs.append((i, j))
        dir_pairs.append((j, i))

    m2 = len(dir_pairs)
    edge_idx = {e: k for k, e in enumerate(dir_pairs)}

    # Build B as a sparse-style list of (row, col) = (1 entries)
    rows, cols_list = [], []
    for k, (u, v) in enumerate(dir_pairs):
        # All outgoing directed edges from v except v→u
        for (w, x) in dir_pairs:
            if w == v and x != u:
                j_idx = edge_idx.get((w, x))
                if j_idx is not None:
                    rows.append(k)
                    cols_list.append(j_idx)

    if not rows:
        return {"spectral_radius": None, "ramanujan_ratio": None, "n_dir_edges": m2}

    # For small graphs build dense matrix; for larger use sparse approach
    if m2 <= 200:
        B = np.zeros((m2, m2))
        for r, c in zip(rows, cols_list):
            B[r, c] = 1.0
        try:
            eigvals = np.linalg.eigvals(B)
            radius  = float(np.max(np.abs(eigvals)))
        except np.linalg.LinAlgError:
            return {"spectral_radius": None, "ramanujan_ratio": None, "n_dir_edges": m2}
    else:
        # Power iteration for largest eigenvalue magnitude
        v = np.random.randn(m2)
        for _ in range(30):
            v_new = np.zeros(m2)
            for r, c in zip(rows, cols_list):
                v_new[r] += v[c]
            norm = np.linalg.norm(v_new)
            if norm < 1e-12:
                break
            v = v_new / norm
        radius = float(norm)

    # Mean degree for Ramanujan baseline
    deg = [0] * n
    for (i, j, _etype, _w) in undir_edges:
        deg[i] += 1
        deg[j] += 1
    mean_deg = float(np.mean(deg))
    ramanujan_base = float(np.sqrt(max(mean_deg - 1, 1e-6)))
    ratio = round(radius / ramanujan_base, 4) if ramanujan_base > 0 else None

    return {
        "spectral_radius":  round(radius, 4),
        "ramanujan_ratio":  ratio,
        "n_dir_edges":      m2,
    }


# ── 27. Magnetic Laplacian frustration index ─────────────────────────────────

def _magnetic_frustration(n: int, undir_edges: list[tuple],
                           dir_edges: list[tuple]) -> dict:
    """
    The magnetic Laplacian L_q is a Hermitian matrix parameterised by q ∈ [0,1):

        L_q[u, u] = deg(u)
        L_q[u, v] = -e^{i 2π q · σ_{uv}}   for each edge {u,v}
        L_q[v, u] = -e^{-i 2π q · σ_{uv}}

    where σ_{uv} ∈ {+1, -1, 0} encodes the *net directed orientation*:
        +1 if u→v dominates, -1 if v→u dominates, 0 if bidirectional.

    Frustration index = λ_min(L_q) minimised over q ∈ [0,1).

        Low frustration  → directed flows are *consistent* (no fighting cycles)
        High frustration → obligation/condition chains contradict each other

    For contract documents, high frustration may signal retrieval of logically
    inconsistent clauses — a strong hallucination precursor.
    """
    if n < 2 or not undir_edges:
        return {"frustration": None, "optimal_q": None, "lambda_min_q0": None}

    # Build net orientation per undirected edge
    fwd_set: set[tuple[int, int]] = {(i, j) for (i, j, _e, _w) in dir_edges}
    sigma: dict[tuple[int, int], float] = {}
    for (i, j, _etype, _w) in undir_edges:
        has_fwd = (i, j) in fwd_set
        has_bwd = (j, i) in fwd_set
        if has_fwd and not has_bwd:
            sigma[(i, j)] = 1.0
        elif has_bwd and not has_fwd:
            sigma[(i, j)] = -1.0
        else:
            sigma[(i, j)] = 0.0   # bidirectional or undirected

    def build_Lq(q: float) -> np.ndarray:
        L = np.zeros((n, n), dtype=complex)
        for (i, j, _etype, _w) in undir_edges:
            s = sigma.get((i, j), 0.0)
            phase = np.exp(1j * 2 * np.pi * q * s)
            L[i, i] += 1.0
            L[j, j] += 1.0
            L[i, j] -= phase
            L[j, i] -= phase.conjugate()
        return L

    # Scan q ∈ [0, 0.5] (symmetric, so [0,0.5] suffices)
    q_vals     = np.linspace(0.0, 0.5, 40)
    min_eigval = float("inf")
    optimal_q  = 0.0

    for q in q_vals:
        try:
            eigvals = np.linalg.eigvalsh(build_Lq(float(q)))
            lam_min = float(np.min(eigvals))
            if lam_min < min_eigval:
                min_eigval = lam_min
                optimal_q  = float(q)
        except np.linalg.LinAlgError:
            continue

    # λ_min at q=0 is just the standard Fiedler for comparison
    try:
        lambda_q0 = float(np.min(np.linalg.eigvalsh(build_Lq(0.0))))
    except Exception:
        lambda_q0 = None

    frustration = round(max(0.0, min_eigval), 4)
    return {
        "frustration":    frustration,
        "optimal_q":      round(optimal_q, 4),
        "lambda_min_q0":  round(lambda_q0, 4) if lambda_q0 is not None else None,
    }


# ── 28. Persistent Laplacian (0th order) ─────────────────────────────────────

def _persistent_laplacian(n: int, undir_edges: list[tuple],
                           node_weights: list[float]) -> dict:
    """
    Persistent 0-Laplacian (Mémoli, Wan & Wang, 2020).

    Filter function: f(v) = anchor_weight(v).
    Filtration thresholds: [1.0, 0.5, 0.1, 0.05, 0.0]  (high → low weight).

    At each threshold a, the subcomplex K_a includes all nodes v with f(v) >= a
    and all edges with both endpoints in K_a.

    We track:
      • β0(a)     — number of connected components at threshold a
      • λ2(a)     — Fiedler value (algebraic connectivity) at threshold a
      • Δλ2       — total drop in λ2 across filtration (geometry stability)
      • persistence profile — λ2 values at each threshold

    The *persistent* aspect: compare how λ2 changes from one scale to the next.
    Fast collapse → geometry is fragile (bad). Stable λ2 → robust (good).
    """
    if n < 2 or not undir_edges:
        return {
            "lambda2_profile": [],
            "beta0_profile":   [],
            "delta_lambda2":   None,
            "stability":       None,
        }

    THRESHOLDS = [1.0, 0.5, 0.1, 0.05, 0.0]
    w = np.array(node_weights, dtype=float)

    lambda2_profile = []
    beta0_profile   = []

    for thr in THRESHOLDS:
        # Nodes active at this threshold
        active = [i for i in range(n) if w[i] >= thr]
        if len(active) < 2:
            lambda2_profile.append(0.0)
            beta0_profile.append(len(active))
            continue

        active_set = set(active)
        idx_map = {v: k for k, v in enumerate(active)}
        m = len(active)

        L = np.zeros((m, m))
        for (i, j, _etype, _wt) in undir_edges:
            if i in active_set and j in active_set:
                ii, jj = idx_map[i], idx_map[j]
                L[ii, ii] += 1
                L[jj, jj] += 1
                L[ii, jj] -= 1
                L[jj, ii] -= 1

        try:
            eigvals = np.sort(np.linalg.eigvalsh(L))
            # β0 = number of near-zero eigenvalues
            beta0 = int(np.sum(eigvals < 1e-6))
            # λ2 = first non-trivial eigenvalue
            nz    = eigvals[eigvals > 1e-6]
            lam2  = float(nz[0]) if len(nz) > 0 else 0.0
        except np.linalg.LinAlgError:
            beta0 = 1
            lam2  = 0.0

        lambda2_profile.append(round(lam2, 4))
        beta0_profile.append(beta0)

    # Stability: negative total change in λ2 (monotone decrease expected)
    delta = round(lambda2_profile[0] - lambda2_profile[-1], 4) if lambda2_profile else None

    # Stability score: how smoothly does λ2 decay (low variance of decrements = stable)
    decrements = [lambda2_profile[i] - lambda2_profile[i+1]
                  for i in range(len(lambda2_profile)-1)]
    stability  = round(1.0 - float(np.std(decrements)) / (float(np.mean(np.abs(decrements))) + 1e-6), 4) \
                 if decrements else None

    return {
        "lambda2_profile": lambda2_profile,
        "beta0_profile":   beta0_profile,
        "delta_lambda2":   delta,
        "stability":       stability,
    }


# ── 29. Heat Kernel Trace K(t) = Tr(e^{-tL}) ─────────────────────────────────

def _heat_kernel_trace(n: int, undir_edges: list[tuple]) -> dict:
    """
    The heat kernel K(t) = Tr(e^{-tL}) = Σᵢ e^{-λᵢ t}

    This is a multi-scale spectral fingerprint of the graph:
      • Small t → local structure (triangles, degree heterogeneity)
      • Large t → global structure (connectivity, diameter, bottlenecks)

    We evaluate K(t) at a geometric sequence of time scales and return:
      • k_vals       — K(t) at t ∈ {0.1, 0.5, 1, 2, 5, 10, 50}
      • k_normalised — K(t)/n  (removes size effect)
      • decay_rate   — slope of log K(t) vs log t  (−d/2 for d-manifold)
      • t_half       — smallest t where K(t) < K(0)/2  (mixing time proxy)

    Correct answers should show faster heat diffusion (lower t_half) because
    the retrieved subgraph has more redundant paths.
    """
    T_VALS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

    if n < 2 or not undir_edges:
        return {
            "k_vals":       [float(n)] * len(T_VALS),
            "k_normalised": [1.0]      * len(T_VALS),
            "decay_rate":   None,
            "t_half":       None,
        }

    L = np.zeros((n, n))
    for (i, j, _etype, _w) in undir_edges:
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1

    try:
        eigvals = np.linalg.eigvalsh(L)
    except np.linalg.LinAlgError:
        return {"k_vals": None, "k_normalised": None, "decay_rate": None, "t_half": None}

    k_vals = []
    for t in T_VALS:
        k = float(np.sum(np.exp(-eigvals * t)))
        k_vals.append(round(k, 4))

    k_norm = [round(k / n, 4) for k in k_vals]

    # Decay rate: linear regression of log(K) vs log(t) for t ∈ [0.5, 10]
    mid_idx  = [1, 2, 3, 4, 5]   # indices for t=0.5..10
    log_t    = np.log([T_VALS[i] for i in mid_idx])
    log_k    = np.array([np.log(max(k_vals[i], 1e-12)) for i in mid_idx])
    if len(log_t) > 1:
        slope = float(np.polyfit(log_t, log_k, 1)[0])
        decay_rate = round(slope, 4)
    else:
        decay_rate = None

    # t_half: smallest t where K(t) < K(t=0.1) * 0.5  (use index 0 as reference)
    k0     = k_vals[0]
    t_half = None
    for t, k in zip(T_VALS, k_vals):
        if k < k0 * 0.5:
            t_half = round(t, 2)
            break

    return {
        "k_vals":       k_vals,
        "k_normalised": k_norm,
        "decay_rate":   decay_rate,
        "t_half":       t_half,
    }


# ── 30. Answer-Path Topology ─────────────────────────────────────────────────

# Maps question slot type → the node type(s) that would directly answer it
_ANSWER_TARGET_TYPES: dict[str, frozenset[str]] = {
    "VALUE":       frozenset({"NUMERIC"}),
    "MEANING":     frozenset({"DEFINITION"}),
    "REQUIREMENT": frozenset({"OBLIGATION"}),
    "PERMISSION":  frozenset({"RIGHT", "OBLIGATION"}),
    "CONSEQUENCE": frozenset({"CONDITION", "OBLIGATION"}),
    "ACTOR":       frozenset({"OBLIGATION", "RIGHT"}),
    "GENERAL":     frozenset({"OBLIGATION", "RIGHT", "NUMERIC", "DEFINITION", "CONDITION"}),
}


def _answer_path_topology(n: int, survivors: list[dict], q_node: dict,
                           slots: list[dict], undir_edges: list[tuple],
                           node_weights: list[float]) -> dict:
    """
    Find the best semantic path from any anchor node to any answer-target node
    and compute topology metrics on that path.

    "Best" = maximises the product of edge weights (= minimises -log sum),
    so the path prefers TRIGGERS/PARTY_HAS chains over SAME_SECTION hops.

    Key signals:
      path_exists          — False = target type absent from retrieval → likely unanswerable
      path_length          — hop count to best-reachable target
      path_bottleneck      — minimum edge weight on path (weakest link)
      path_score           — product of edge weights (end-to-end semantic flow)
      same_section_frac    — fraction of path edges that are SAME_SECTION (bad)
      n_targets            — how many answer-target nodes were retrieved
      n_well_connected     — targets reachable via weight ≥ 0.4 (strong semantic path)
      target_weight_mean   — mean anchor-propagated weight of all target nodes
      target_weight_max    — best-connected target's weight
    """
    import heapq
    import math

    null_result = {
        "path_exists":        False,
        "path_length":        None,
        "path_bottleneck":    None,
        "path_score":         None,
        "path_edge_types":    [],
        "same_section_frac":  None,
        "n_targets":          0,
        "n_well_connected":   0,
        "target_weight_mean": None,
        "target_weight_max":  None,
        "slot_type":          "GENERAL",
    }

    if n < 2 or not undir_edges:
        return null_result

    # Determine target node types from slot
    primary_slot = (slots[0].get("slot") if slots else None) or "GENERAL"
    target_types = _ANSWER_TARGET_TYPES.get(primary_slot,
                       _ANSWER_TARGET_TYPES["GENERAL"])
    null_result["slot_type"] = primary_slot

    # Q-context anchors: nodes matching question KEYWORDS only (not type).
    # This is intentionally different from the BFS anchor detection —
    # we want nodes that *mention* the question topic, not nodes that
    # happen to be the right type. The path FROM these TO target-type
    # nodes is the answer chain we're measuring.
    _Q_FIELDS = ("party", "term", "action", "applies_to",
                 "describes", "right", "consequence", "trigger")
    q_values: list[str] = []
    for f in _Q_FIELDS:
        v = (q_node.get(f) or "").strip().lower()
        if len(v) >= 4:
            q_values.append(v)

    def _text(nd: dict) -> str:
        return " ".join(filter(None, [
            nd.get("source_text",""), nd.get("action",""), nd.get("party",""),
            nd.get("term",""), nd.get("right",""), nd.get("definition",""),
            nd.get("trigger",""), nd.get("consequence",""), nd.get("applies_to",""),
        ])).lower()

    # Anchors must NOT be of the target type — we want the path FROM context
    # nodes TO answer nodes, not the trivial self-path.
    non_target_ids = {i for i, nd in enumerate(survivors)
                      if nd.get("type") not in target_types}

    anchor_ids: list[int] = []
    if q_values:
        anchor_ids = [i for i, nd in enumerate(survivors)
                      if i in non_target_ids
                      and any(qv in _text(nd) for qv in q_values)]

    # Fallback: high-weight non-target nodes
    if not anchor_ids:
        anchor_ids = [i for i, w in enumerate(node_weights)
                      if w >= 0.5 and i in non_target_ids]
    if not anchor_ids:
        anchor_ids = [i for i in non_target_ids]   # last resort: all non-targets

    # Answer targets: nodes of the output type (explicitly NOT anchors)
    target_ids = [i for i, nd in enumerate(survivors)
                  if nd.get("type") in target_types]

    n_targets = len(target_ids)
    null_result["n_targets"] = n_targets

    # Target weight stats from BFS propagation (encodes semantic reachability)
    t_weights = [node_weights[i] for i in target_ids]
    if t_weights:
        null_result["target_weight_mean"] = round(float(np.mean(t_weights)), 4)
        null_result["target_weight_max"]  = round(float(np.max(t_weights)),  4)
        null_result["n_well_connected"]   = sum(1 for w in t_weights if w >= 0.4)

    if not anchor_ids or not target_ids:
        return null_result

    # Targets that directly match question keywords — path_length=0
    direct_hits = set(anchor_ids) & set(target_ids)
    if direct_hits:
        best_direct = max(direct_hits, key=lambda i: node_weights[i])
        return {
            **null_result,
            "path_exists":       True,
            "path_length":       0,
            "path_bottleneck":   1.0,
            "path_score":        1.0,
            "path_edge_types":   [],
            "same_section_frac": 0.0,
        }

    # Build adjacency: {node: [(neighbor, etype, weight)]}
    adj: list[list[tuple[int, str, float]]] = [[] for _ in range(n)]
    for (i, j, etype, w) in undir_edges:
        adj[i].append((j, etype, w))
        adj[j].append((i, etype, w))

    # Dijkstra on -log(weight) to find max-weight path from any anchor
    # cost[v] = accumulated -log(product of weights) = sum of -log(w_e)
    INF = float("inf")
    cost   = [INF] * n
    parent = [-1]  * n
    pedge  = [None] * n   # edge type used to reach each node

    heap = []
    for a in anchor_ids:
        cost[a] = 0.0
        heapq.heappush(heap, (0.0, a))

    target_set = set(target_ids)
    best_target = -1

    while heap:
        c, v = heapq.heappop(heap)
        if c > cost[v] + 1e-12:
            continue
        if v in target_set:
            best_target = v
            break          # first (lowest cost = highest weight) target found
        for (u, etype, w) in adj[v]:
            safe_w   = max(w, 1e-9)
            new_cost = c + (-math.log(safe_w))
            if new_cost < cost[u] - 1e-12:
                cost[u]   = new_cost
                parent[u] = v
                pedge[u]  = etype
                heapq.heappush(heap, (new_cost, u))

    if best_target == -1 or cost[best_target] == INF:
        return null_result   # no path found

    # Reconstruct path
    path_edges: list[str] = []
    path_weights: list[float] = []
    cur = best_target
    while parent[cur] != -1:
        etype = pedge[cur]
        path_edges.append(etype)
        # recover edge weight from adjacency
        prev = parent[cur]
        w = next((w for nb, et, w in adj[prev] if nb == cur and et == etype), 0.5)
        path_weights.append(w)
        cur = prev
    path_edges.reverse()
    path_weights.reverse()

    path_length     = len(path_edges)
    path_bottleneck = round(float(min(path_weights)), 4) if path_weights else 1.0
    path_score      = round(float(np.prod(path_weights)), 4) if path_weights else 1.0
    ss_count        = sum(1 for e in path_edges if e == "SAME_SECTION")
    same_sec_frac   = round(ss_count / len(path_edges), 4) if path_edges else 0.0

    return {
        "path_exists":        True,
        "path_length":        path_length,
        "path_bottleneck":    path_bottleneck,
        "path_score":         path_score,
        "path_edge_types":    path_edges,
        "same_section_frac":  same_sec_frac,
        "n_targets":          n_targets,
        "n_well_connected":   null_result["n_well_connected"],
        "target_weight_mean": null_result["target_weight_mean"],
        "target_weight_max":  null_result["target_weight_max"],
        "slot_type":          primary_slot,
    }


# ── 31. Target Centrality ─────────────────────────────────────────────────────

def _target_centrality(survivors: list[dict], node_weights: list[float],
                        undir_edges: list[tuple], slots: list[dict]) -> dict:
    """
    Structural properties of answer-target nodes in the retrieved subgraph.

    KEY SIGNAL for "missed retrieval" pattern:
      Wrong answers = target node EXISTS but is ISOLATED (low degree, no causal path).
      Correct answers = target well-embedded in a dense neighbourhood.

    Returns:
      target_mean_degree  — avg degree of target-type nodes (low = isolated)
      target_min_degree   — worst-case isolation
      target_orphan_frac  — fraction with degree ≤ 1 (pulled out of context)
      target_type_frac    — n_target_nodes / n_total (low = retrieval missed answer)
      n_targets           — count of answer-type nodes
    """
    primary_slot = (slots[0].get("slot") if slots else None) or "GENERAL"
    target_types = _ANSWER_TARGET_TYPES.get(primary_slot, _ANSWER_TARGET_TYPES["GENERAL"])

    n = len(survivors)
    if n == 0:
        return {
            "target_mean_degree": None, "target_min_degree": None,
            "target_orphan_frac": None, "target_type_frac": None, "n_targets": 0,
        }

    # Degree of each node in the subgraph
    degree = [0] * n
    for (i, j, _etype, _w) in undir_edges:
        degree[i] += 1
        degree[j] += 1

    target_ids = [i for i, nd in enumerate(survivors) if nd.get("type") in target_types]
    n_targets  = len(target_ids)

    if n_targets == 0:
        return {
            "target_mean_degree": None, "target_min_degree": None,
            "target_orphan_frac": None,
            "target_type_frac":   0.0,
            "n_targets":          0,
        }

    degs    = [degree[i] for i in target_ids]
    orphans = sum(1 for d in degs if d <= 1)

    return {
        "target_mean_degree": round(float(np.mean(degs)), 4),
        "target_min_degree":  int(min(degs)),
        "target_orphan_frac": round(orphans / n_targets, 4),
        "target_type_frac":   round(n_targets / n, 4),
        "n_targets":          n_targets,
    }


# ── 32. Causal Reachability ───────────────────────────────────────────────────

def _causal_reachability(survivors: list[dict], node_weights: list[float],
                          dir_edges: list[tuple], slots: list[dict]) -> dict:
    """
    Reachability from anchor nodes to answer-target nodes using ONLY semantic
    edges (TRIGGERS, PARTY_HAS, USES_TERM) — no SAME_SECTION, no CONTRADICTS.

    Distinguishes:
      "Causal path" — answer reached via meaningful legal chain (TRIGGERS cascade)
      "Proximity path" — answer reached only via same-section co-location (noise)

    If answer_path exists but causal_reachable = 0, the LLM answer chain
    relies on proximity, not logic — high hallucination risk.

    Returns:
      causal_reachable      — n target nodes reachable via semantic edges only
      causal_frac           — causal_reachable / n_targets
      semantic_edge_frac    — fraction of all edges that are semantic (not SAME_SECTION/CONTRADICTS)
      anchor_trigger_count  — TRIGGERS edges incident to anchor nodes (direct causal hooks)
    """
    SEMANTIC_TYPES = {"TRIGGERS", "PARTY_HAS", "USES_TERM"}

    primary_slot = (slots[0].get("slot") if slots else None) or "GENERAL"
    target_types = _ANSWER_TARGET_TYPES.get(primary_slot, _ANSWER_TARGET_TYPES["GENERAL"])

    n = len(survivors)
    if n == 0:
        return {
            "causal_reachable": 0, "causal_frac": None,
            "semantic_edge_frac": None, "anchor_trigger_count": 0,
        }

    anchor_ids = {i for i, w in enumerate(node_weights) if w >= 1.0}
    target_ids = {i for i, nd in enumerate(survivors) if nd.get("type") in target_types}

    # Semantic-only adjacency (directed + reversed = undirected reachability)
    semantic_adj: list[set[int]] = [set() for _ in range(n)]
    all_edges = 0
    semantic_edges = 0
    for (i, j, etype, _w) in dir_edges:
        all_edges += 1
        if etype in SEMANTIC_TYPES:
            semantic_adj[i].add(j)
            semantic_adj[j].add(i)
            semantic_edges += 1

    semantic_edge_frac = round(semantic_edges / all_edges, 4) if all_edges > 0 else None

    # BFS from anchors over semantic-only graph
    visited: set[int] = set(anchor_ids)
    queue   = list(anchor_ids)
    while queue:
        v    = queue.pop()
        for u in semantic_adj[v]:
            if u not in visited:
                visited.add(u)
                queue.append(u)

    causal_reachable = len(target_ids & visited)
    causal_frac      = round(causal_reachable / len(target_ids), 4) if target_ids else None

    # TRIGGERS edges directly touching anchors
    anchor_trigger_count = sum(
        1 for (i, j, etype, _w) in dir_edges
        if etype == "TRIGGERS" and (i in anchor_ids or j in anchor_ids)
    )

    return {
        "causal_reachable":    causal_reachable,
        "causal_frac":         causal_frac,
        "semantic_edge_frac":  semantic_edge_frac,
        "anchor_trigger_count": anchor_trigger_count,
    }


# ── 33. Anchor Convergence ────────────────────────────────────────────────────

def _anchor_convergence(survivors: list[dict], node_weights: list[float],
                         undir_edges: list[tuple], slots: list[dict]) -> dict:
    """
    How many distinct anchors can independently reach the best answer-target?

    Multi-path corroboration: if 3 different anchor nodes each have an independent
    path to the same target, that answer is harder to hallucinate away.

    Returns:
      max_anchors_to_target  — highest n_anchors that can reach a single target
      anchor_agreement_score — max_anchors / total_anchors (0→1)
      best_target_type       — type of the most-agreed-upon target node
      n_reachable_targets    — targets reachable from ANY anchor
    """
    from collections import deque

    primary_slot = (slots[0].get("slot") if slots else None) or "GENERAL"
    target_types = _ANSWER_TARGET_TYPES.get(primary_slot, _ANSWER_TARGET_TYPES["GENERAL"])

    n = len(survivors)
    anchor_ids = [i for i, w in enumerate(node_weights) if w >= 1.0]
    target_ids = [i for i, nd in enumerate(survivors) if nd.get("type") in target_types]

    null = {
        "max_anchors_to_target":  0,
        "anchor_agreement_score": None,
        "best_target_type":       None,
        "n_reachable_targets":    0,
    }

    if not anchor_ids or not target_ids or n < 2:
        return null

    # Build undirected adjacency for reachability
    adj: list[list[int]] = [[] for _ in range(n)]
    for (i, j, _etype, _w) in undir_edges:
        adj[i].append(j)
        adj[j].append(i)

    # For each anchor, BFS to find reachable targets
    target_to_anchor_count: dict[int, int] = defaultdict(int)

    for a in anchor_ids:
        visited: set[int] = {a}
        q: deque[int] = deque([a])
        while q:
            v = q.popleft()
            if v in set(target_ids):
                target_to_anchor_count[v] += 1
            for u in adj[v]:
                if u not in visited:
                    visited.add(u)
                    q.append(u)

    if not target_to_anchor_count:
        return null

    best_target      = max(target_to_anchor_count, key=target_to_anchor_count.get)
    max_anchors      = target_to_anchor_count[best_target]
    n_reachable      = len(target_to_anchor_count)
    agreement_score  = round(max_anchors / len(anchor_ids), 4) if anchor_ids else None
    best_type        = survivors[best_target].get("type")

    return {
        "max_anchors_to_target":  max_anchors,
        "anchor_agreement_score": agreement_score,
        "best_target_type":       best_type,
        "n_reachable_targets":    n_reachable,
    }


# ── 39. Random Walk Hitting Time ─────────────────────────────────────────────

def _hitting_time(n: int, survivors: list[dict], node_weights: list[float],
                   undir_edges: list[tuple], slots: list[dict]) -> dict:
    """
    Expected steps for a random walk starting at any anchor node to first reach
    any answer-target node.

    Uses the fundamental matrix of an absorbing Markov chain:
      - Answer-target nodes are absorbing states
      - All other nodes transition uniformly to neighbours
      - h[i] = expected steps from node i to absorption

    Low hitting time  = answer is structurally easy to reach (many paths)
    High hitting time = answer is isolated even if technically connected
    None              = answer-type nodes absent or anchors == targets
    """
    primary_slot = (slots[0].get("slot") if slots else None) or "GENERAL"
    target_types = _ANSWER_TARGET_TYPES.get(primary_slot, _ANSWER_TARGET_TYPES["GENERAL"])

    target_ids = {i for i, nd in enumerate(survivors) if nd.get("type") in target_types}
    anchor_ids = [i for i, w in enumerate(node_weights) if w >= 1.0]

    null = {"mean_hitting_time": None, "min_hitting_time": None, "n_transient": 0}
    if not target_ids or not anchor_ids or n < 3:
        return null

    # Build degree and adjacency
    deg = [0] * n
    adj: list[list[int]] = [[] for _ in range(n)]
    for (i, j, _e, _w) in undir_edges:
        adj[i].append(j); adj[j].append(i)
        deg[i] += 1;      deg[j] += 1

    # Transient states: non-target nodes with at least one edge
    transient = [i for i in range(n) if i not in target_ids and deg[i] > 0]
    if not transient:
        return null

    m = len(transient)
    idx = {v: k for k, v in enumerate(transient)}

    # Build (I - Q) where Q is the transient-to-transient sub-matrix
    IQ = np.eye(m)
    for k, v in enumerate(transient):
        if deg[v] == 0:
            continue
        for u in adj[v]:
            if u in idx:
                IQ[k, idx[u]] -= 1.0 / deg[v]

    ones = np.ones(m)
    try:
        h = np.linalg.solve(IQ, ones)
    except np.linalg.LinAlgError:
        return null

    anchor_times = [float(h[idx[a]]) for a in anchor_ids if a in idx]
    if not anchor_times:
        return null

    return {
        "mean_hitting_time": round(float(np.mean(anchor_times)), 4),
        "min_hitting_time":  round(float(np.min(anchor_times)),  4),
        "n_transient":       m,
    }


# ── 40. PageRank of Answer Nodes ──────────────────────────────────────────────

def _answer_pagerank(n: int, survivors: list[dict],
                      dir_edges: list[tuple], slots: list[dict]) -> dict:
    """
    Run PageRank on the directed subgraph and report statistics for
    answer-target nodes specifically.

    High answer PageRank = many paths converge on answer nodes = well-supported.
    Low  answer PageRank = answer nodes are peripheral / weakly cited.

    Uses the standard power-iteration PageRank (damping=0.85, 50 iterations).
    Falls back to uniform if no directed edges exist.
    """
    primary_slot = (slots[0].get("slot") if slots else None) or "GENERAL"
    target_types = _ANSWER_TARGET_TYPES.get(primary_slot, _ANSWER_TARGET_TYPES["GENERAL"])
    target_ids   = [i for i, nd in enumerate(survivors) if nd.get("type") in target_types]

    null = {"answer_pr_mean": None, "answer_pr_max": None,
            "answer_pr_rank": None, "n_targets": len(target_ids)}
    if n < 2:
        return null

    DAMPING = 0.85
    pr = np.ones(n) / n

    # Build column-normalised adjacency (in-links for PageRank)
    in_edges: list[list[int]] = [[] for _ in range(n)]
    out_deg = [0] * n
    for (i, j, _e, _w) in dir_edges:
        in_edges[j].append(i)
        out_deg[i] += 1

    for _ in range(50):
        new_pr = np.ones(n) * (1 - DAMPING) / n
        for j in range(n):
            if in_edges[j]:
                new_pr[j] += DAMPING * sum(pr[i] / max(out_deg[i], 1) for i in in_edges[j])
        pr = new_pr / (new_pr.sum() or 1.0)

    if not target_ids:
        return null

    tpr = [float(pr[i]) for i in target_ids]
    # Rank: fraction of ALL nodes with lower PageRank than the mean answer PR
    mean_tpr = float(np.mean(tpr))
    rank_frac = round(float(np.mean(pr < mean_tpr)), 4)

    return {
        "answer_pr_mean": round(mean_tpr, 6),
        "answer_pr_max":  round(float(np.max(tpr)), 6),
        "answer_pr_rank": rank_frac,   # 0.9 = answers rank in top 10% of all nodes
        "n_targets":      len(target_ids),
    }


# ── 41. Feed-Forward Loop Count ───────────────────────────────────────────────

def _feedforward_loops(n: int, survivors: list[dict],
                        dir_edges: list[tuple]) -> dict:
    """
    Count directed feed-forward loops (FFL): A→B, A→C, B→C.
    These are the directed equivalent of triangles — two independent causal
    paths converging on the same node.

    More FFLs = more redundant causal corroboration = harder to hallucinate.

    Also counts type-constrained FFLs:
      CONDITION→TRIGGERS→OBLIGATION with CONDITION→TRIGGERS→OBLIGATION (strong legal chain)
      i.e. two CONDITION nodes both triggering the same OBLIGATION.

    Returns:
      n_ffl           — total feed-forward loops
      ffl_per_edge    — normalised by edge count
      typed_ffl       — FFLs where terminal node is an answer-type node
    """
    if n < 3 or not dir_edges:
        return {"n_ffl": 0, "ffl_per_edge": 0.0, "typed_ffl": 0}

    # Build out-adjacency set for fast lookup
    out_adj: dict[int, set[int]] = defaultdict(set)
    for (i, j, _e, _w) in dir_edges:
        out_adj[i].add(j)

    ffl_count  = 0
    typed_ffl  = 0
    e = len(dir_edges)

    for (i, j, _e1, _w1) in dir_edges:       # edge A→B
        for k in out_adj[i]:                   # edge A→C
            if k != j and j in out_adj[k]:     # edge B→C completes the FFL
                ffl_count += 1
                # Check if C (terminal) is an answer-relevant type
                c_type = survivors[k].get("type", "")
                if c_type in {"OBLIGATION", "RIGHT", "NUMERIC", "DEFINITION"}:
                    typed_ffl += 1

    return {
        "n_ffl":        ffl_count,
        "ffl_per_edge": round(ffl_count / e, 4) if e > 0 else 0.0,
        "typed_ffl":    typed_ffl,
    }


# ── 42. Resistance Distance Anchor-to-Answer ─────────────────────────────────

def _anchor_to_answer_resistance(n: int, survivors: list[dict],
                                   undir_edges: list[tuple],
                                   node_weights: list[float],
                                   slots: list[dict]) -> dict:
    """
    Effective resistance from anchor nodes TO answer-target nodes.
    (Complements eff_resistance which measures anchor-to-anchor.)

    Low  = many parallel paths connect question context to answer = robust.
    High = single bridge to the answer = fragile, easy to miss.
    None = no anchor-target pairs exist.
    """
    primary_slot = (slots[0].get("slot") if slots else None) or "GENERAL"
    target_types = _ANSWER_TARGET_TYPES.get(primary_slot, _ANSWER_TARGET_TYPES["GENERAL"])
    target_ids   = [i for i, nd in enumerate(survivors) if nd.get("type") in target_types]
    anchor_ids   = [i for i, w in enumerate(node_weights) if w >= 1.0]

    null = {"mean_a2a_resistance": None, "min_a2a_resistance": None, "n_pairs": 0}
    if not anchor_ids or not target_ids or n < 2:
        return null

    # Build combinatorial Laplacian
    L = np.zeros((n, n))
    for (i, j, _e, _w) in undir_edges:
        L[i,i] += 1; L[j,j] += 1
        L[i,j] -= 1; L[j,i] -= 1

    try:
        L_plus = np.linalg.pinv(L)
    except np.linalg.LinAlgError:
        return null

    resistances = []
    for a in anchor_ids:
        for t in target_ids:
            if a == t:
                continue
            r = float(L_plus[a,a] + L_plus[t,t] - 2*L_plus[a,t])
            resistances.append(max(0.0, r))

    if not resistances:
        return null

    return {
        "mean_a2a_resistance": round(float(np.mean(resistances)), 4),
        "min_a2a_resistance":  round(float(np.min(resistances)),  4),
        "n_pairs":             len(resistances),
    }


# ── 43. Degree Sequence Entropy ───────────────────────────────────────────────

def _degree_sequence_entropy(n: int, undir_edges: list[tuple]) -> dict:
    """
    Entropy of the degree distribution.

    Flat (uniform) distribution → balanced retrieval, no dominant hubs.
    Spiky (power-law) distribution → hub-dominated, fragile single-point retrieval.

    Also returns:
      hub_frac     — fraction of nodes with degree > mean+std (outlier hubs)
      isolated_frac — fraction of degree-0 nodes (unreachable nodes)
      gini          — Gini coefficient of degree sequence (0=equal, 1=maximally unequal)
    """
    if n < 2:
        return {"deg_entropy": None, "hub_frac": None,
                "isolated_frac": None, "gini": None}

    deg = [0] * n
    for (i, j, _e, _w) in undir_edges:
        deg[i] += 1; deg[j] += 1

    deg_arr = np.array(deg, dtype=float)
    total   = deg_arr.sum()

    # Shannon entropy of degree distribution
    if total > 0:
        p = deg_arr / total
        entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    else:
        entropy = 0.0

    mean_d = float(np.mean(deg_arr))
    std_d  = float(np.std(deg_arr))
    hub_frac      = round(float(np.mean(deg_arr > mean_d + std_d)), 4)
    isolated_frac = round(float(np.mean(deg_arr == 0)), 4)

    # Gini coefficient
    sorted_d = np.sort(deg_arr)
    idx_arr  = np.arange(1, n + 1)
    gini     = float(2 * np.sum(idx_arr * sorted_d) / (n * sorted_d.sum()) - (n+1)/n) \
               if sorted_d.sum() > 0 else 0.0

    return {
        "deg_entropy":    round(entropy, 4),
        "hub_frac":       hub_frac,
        "isolated_frac":  isolated_frac,
        "gini":           round(gini, 4),
    }


# ── 44. Directed Acyclicity Score ─────────────────────────────────────────────

def _directed_acyclicity(n: int, dir_edges: list[tuple]) -> dict:
    """
    What fraction of the directed subgraph is acyclic?

    Legal reasoning should flow forward: conditions trigger obligations,
    not back. Cycles indicate contradictory or circular references.

    Algorithm: DFS-based back-edge detection.
      acyclicity_score = 1 - (back_edges / total_directed_edges)
      1.0 = fully acyclic (DAG) — ideal legal chain
      0.0 = every edge is part of a cycle — contradictory retrieval

    Also returns:
      n_back_edges   — directed edges that create cycles
      n_scc_gt1      — strongly connected components with >1 node (cycle clusters)
    """
    if n < 2 or not dir_edges:
        return {"acyclicity_score": 1.0, "n_back_edges": 0, "n_scc_gt1": 0}

    out_adj: list[list[int]] = [[] for _ in range(n)]
    for (i, j, _e, _w) in dir_edges:
        out_adj[i].append(j)

    # DFS back-edge count
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    back_edges = 0

    for start in range(n):
        if color[start] != WHITE:
            continue
        stack = [(start, iter(out_adj[start]))]
        color[start] = GRAY
        while stack:
            v, it = stack[-1]
            try:
                u = next(it)
                if color[u] == GRAY:
                    back_edges += 1
                elif color[u] == WHITE:
                    color[u] = GRAY
                    stack.append((u, iter(out_adj[u])))
            except StopIteration:
                color[v] = BLACK
                stack.pop()

    # Kosaraju SCC for n_scc_gt1
    # Pass 1: finish order on original graph
    visited = [False] * n
    finish  = []

    def dfs1(s):
        stk = [(s, iter(out_adj[s]))]
        visited[s] = True
        while stk:
            v, it = stk[-1]
            try:
                u = next(it)
                if not visited[u]:
                    visited[u] = True
                    stk.append((u, iter(out_adj[u])))
            except StopIteration:
                finish.append(v)
                stk.pop()

    for v in range(n):
        if not visited[v]:
            dfs1(v)

    # Pass 2: DFS on reversed graph in reverse finish order
    rev_adj: list[list[int]] = [[] for _ in range(n)]
    for (i, j, _e, _w) in dir_edges:
        rev_adj[j].append(i)

    visited2 = [False] * n
    n_scc_gt1 = 0

    for v in reversed(finish):
        if visited2[v]:
            continue
        comp = []
        stk  = [v]
        visited2[v] = True
        while stk:
            u = stk.pop()
            comp.append(u)
            for w in rev_adj[u]:
                if not visited2[w]:
                    visited2[w] = True
                    stk.append(w)
        if len(comp) > 1:
            n_scc_gt1 += 1

    e = len(dir_edges)
    score = round(1.0 - back_edges / e, 4) if e > 0 else 1.0

    return {
        "acyclicity_score": score,
        "n_back_edges":     back_edges,
        "n_scc_gt1":        n_scc_gt1,
    }


# ── 34. Chain Completeness ────────────────────────────────────────────────────

def _chain_completeness(survivors: list[dict], dir_edges: list[tuple],
                         slots: list[dict]) -> dict:
    """
    For every CONDITION node in the subgraph, check whether it has BOTH:
      - an incoming TRIGGERS edge (the premise exists)
      - an outgoing TRIGGERS edge to an answer-type node (the consequence exists)

    A "dangling" condition has a premise but no consequence in the subgraph —
    retrieval got halfway through the chain but missed the terminal answer node.

    Returns:
      n_conditions        — CONDITION nodes in subgraph
      complete_chains     — conditions with both ends present
      dangling_conditions — conditions missing their consequence
      chain_completion_frac — complete / total (1.0 = all chains resolved)
      required_type_present — is any answer-type node in the subgraph at all?
    """
    primary_slot = (slots[0].get("slot") if slots else None) or "GENERAL"
    target_types = _ANSWER_TARGET_TYPES.get(primary_slot, _ANSWER_TARGET_TYPES["GENERAL"])

    condition_ids = {i for i, nd in enumerate(survivors) if nd.get("type") == "CONDITION"}
    target_ids    = {i for i, nd in enumerate(survivors) if nd.get("type") in target_types}

    null = {
        "n_conditions": len(condition_ids), "complete_chains": 0,
        "dangling_conditions": 0, "chain_completion_frac": None,
        "required_type_present": len(target_ids) > 0,
    }

    if not condition_ids:
        return null

    # Build incoming/outgoing TRIGGERS sets per node
    triggers_in:  dict[int, set[int]] = defaultdict(set)
    triggers_out: dict[int, set[int]] = defaultdict(set)
    for (i, j, etype, _w) in dir_edges:
        if etype == "TRIGGERS":
            triggers_out[i].add(j)
            triggers_in[j].add(i)

    complete  = 0
    dangling  = 0
    for cid in condition_ids:
        has_premise     = bool(triggers_in[cid])   # something triggers this condition
        has_consequence = any(j in target_ids for j in triggers_out[cid])
        if has_premise and has_consequence:
            complete += 1
        elif has_premise and not has_consequence:
            dangling += 1

    frac = round(complete / len(condition_ids), 4)

    return {
        "n_conditions":          len(condition_ids),
        "complete_chains":       complete,
        "dangling_conditions":   dangling,
        "chain_completion_frac": frac,
        "required_type_present": len(target_ids) > 0,
    }


# ── 35. Anchor Type Match ─────────────────────────────────────────────────────

# Type graph: which types logically lead to which answer types
# Distance = hops in the type dependency graph
_TYPE_LEADS_TO: dict[str, list[str]] = {
    "CONDITION":  ["OBLIGATION", "RIGHT"],    # 1 hop via TRIGGERS
    "OBLIGATION": ["NUMERIC", "DEFINITION"],  # 1 hop via USES_TERM
    "RIGHT":      ["NUMERIC", "DEFINITION"],
    "DEFINITION": [],
    "NUMERIC":    [],
    "REFERENCE":  ["OBLIGATION", "RIGHT"],    # 1 hop via reference resolution
}

def _anchor_type_match(survivors: list[dict], node_weights: list[float],
                        slots: list[dict]) -> dict:
    """
    Are the anchor nodes (question-keyword matches) of a type that logically
    leads to the answer type?

    Direct match:   anchor type IS the answer type → score 1.0
    One-hop:        anchor type leads to answer type in one type-graph step → 0.6
    Two-hop:        two steps → 0.3
    No path:        anchor type has no known path to answer type → 0.0

    anchor_type_score = mean over all anchors of their lead-score to best answer type
    anchor_type_diversity = n distinct anchor types / n anchors (redundancy check)
    """
    primary_slot = (slots[0].get("slot") if slots else None) or "GENERAL"
    target_types = _ANSWER_TARGET_TYPES.get(primary_slot, _ANSWER_TARGET_TYPES["GENERAL"])

    anchor_ids = [i for i, w in enumerate(node_weights) if w >= 1.0]
    if not anchor_ids:
        return {"anchor_type_score": None, "anchor_type_diversity": None,
                "anchor_types": {}, "dominant_anchor_type": None}

    def lead_score(atype: str) -> float:
        if atype in target_types:
            return 1.0
        one_hop = set(_TYPE_LEADS_TO.get(atype, []))
        if one_hop & target_types:
            return 0.6
        two_hop = set(t for a in one_hop for t in _TYPE_LEADS_TO.get(a, []))
        if two_hop & target_types:
            return 0.3
        return 0.0

    scores = [lead_score(survivors[i].get("type", "")) for i in anchor_ids]
    anchor_type_score = round(float(np.mean(scores)), 4)

    type_counts: dict[str, int] = defaultdict(int)
    for i in anchor_ids:
        type_counts[survivors[i].get("type", "?")] += 1
    diversity = round(len(type_counts) / len(anchor_ids), 4)
    dominant  = max(type_counts, key=type_counts.get) if type_counts else None

    return {
        "anchor_type_score":     anchor_type_score,
        "anchor_type_diversity": diversity,
        "anchor_types":          dict(type_counts),
        "dominant_anchor_type":  dominant,
    }


# ── 36. Slot Mismatch Depth ───────────────────────────────────────────────────

# Type-graph hop distances between node types (for mismatch depth)
_TYPE_DISTANCE: dict[tuple[str,str], int] = {
    # Same type = 0
    ("OBLIGATION","OBLIGATION"): 0, ("RIGHT","RIGHT"): 0,
    ("CONDITION","CONDITION"): 0,   ("DEFINITION","DEFINITION"): 0,
    ("NUMERIC","NUMERIC"): 0,
    # Direct semantic neighbours = 1
    ("CONDITION","OBLIGATION"): 1, ("OBLIGATION","CONDITION"): 1,
    ("CONDITION","RIGHT"):      1, ("RIGHT","CONDITION"):      1,
    ("OBLIGATION","RIGHT"):     1, ("RIGHT","OBLIGATION"):     1,
    ("OBLIGATION","NUMERIC"):   1, ("NUMERIC","OBLIGATION"):   1,
    ("RIGHT","NUMERIC"):        1, ("NUMERIC","RIGHT"):        1,
    ("OBLIGATION","DEFINITION"):1, ("DEFINITION","OBLIGATION"):1,
    ("RIGHT","DEFINITION"):     1, ("DEFINITION","RIGHT"):     1,
    # Two hops
    ("CONDITION","NUMERIC"):    2, ("NUMERIC","CONDITION"):    2,
    ("CONDITION","DEFINITION"): 2, ("DEFINITION","CONDITION"): 2,
    ("REFERENCE","OBLIGATION"): 1, ("REFERENCE","RIGHT"):      1,
}

def _slot_mismatch_depth(survivors: list[dict], slots: list[dict]) -> dict:
    """
    How far are the retrieved node types from the required answer type?

    For each survivor node, compute the minimum type-graph distance to any
    target type. Average this over all survivors.

    mismatch_depth = 0    → all retrieved nodes are exactly the right type
    mismatch_depth = 1    → retrieved adjacent types (CONDITION instead of OBLIGATION)
    mismatch_depth = 2    → retrieved distant types (DEFINITION when needing NUMERIC)
    mismatch_depth = 3+   → completely wrong region of the type graph

    Also returns:
      exact_frac    — fraction of nodes that ARE the target type
      adjacent_frac — fraction that are 1 hop away
      distant_frac  — fraction that are 2+ hops away
    """
    primary_slot = (slots[0].get("slot") if slots else None) or "GENERAL"
    target_types = _ANSWER_TARGET_TYPES.get(primary_slot, _ANSWER_TARGET_TYPES["GENERAL"])

    if not survivors:
        return {"mismatch_depth": None, "exact_frac": None,
                "adjacent_frac": None, "distant_frac": None}

    def min_dist(ntype: str) -> int:
        if ntype in target_types:
            return 0
        return min(
            (_TYPE_DISTANCE.get((ntype, t), 3) for t in target_types),
            default=3
        )
    # Focus on nodes that are relevant to the asked subject/value form.
    # This avoids diluting exactness with unrelated surviving nodes.
    subject_tokens: set[str] = set()
    value_focus = ""
    for sl in slots or []:
        subj = str(sl.get("subject") or "").lower()
        for tok in re.findall(r"[a-z0-9]+", subj):
            if len(tok) >= 4:
                subject_tokens.add(tok)
        if sl.get("value_focus"):
            value_focus = str(sl.get("value_focus") or "")

    considered: list[dict] = []
    for nd in survivors:
        ntype = nd.get("type", "")
        text = " ".join(filter(None, [
            str(nd.get("source_text") or ""),
            str(nd.get("action") or ""),
            str(nd.get("right") or ""),
            str(nd.get("trigger") or ""),
            str(nd.get("consequence") or ""),
            str(nd.get("applies_to") or ""),
            str(nd.get("value") or ""),
            str(nd.get("unit") or ""),
        ])).lower()
        overlap = any(tok in text for tok in subject_tokens) if subject_tokens else False

        value_like = False
        if primary_slot == "VALUE":
            has_num = bool(re.search(r"\d", text))
            has_pct = bool(re.search(r"\b\d{1,3}(?:\.\d+)?\s*%", text))
            has_per = bool(re.search(r"\bper\s+(?:month|year|day|week|quarter)\b", text))
            has_freq = ("once per year" in text) or ("no more than once per year" in text)
            has_cap = ("liability" in text and "exceed" in text and "months" in text)
            value_like = has_num or has_pct or has_per or has_freq or has_cap
            if value_focus == "rate":
                value_like = value_like and (has_pct or has_per)
            elif value_focus == "frequency":
                value_like = value_like and (has_freq or "per year" in text or "per month" in text)
            elif value_focus == "price_rule":
                value_like = value_like and ("price" in text or "order form" in text)
            elif value_focus == "cap":
                value_like = value_like and has_cap

        if overlap or ntype in target_types or value_like:
            considered.append(nd)

    use_nodes = considered if considered else survivors
    dists = [min_dist(nd.get("type", "")) for nd in use_nodes]
    n = len(dists)

    return {
        "mismatch_depth": round(float(np.mean(dists)), 4),
        "exact_frac":     round(sum(1 for d in dists if d == 0) / n, 4),
        "adjacent_frac":  round(sum(1 for d in dists if d == 1) / n, 4),
        "distant_frac":   round(sum(1 for d in dists if d >= 2) / n, 4),
    }


# ── 37. Required Type Sequence ────────────────────────────────────────────────

# The directed type-path each slot type needs to be answerable
_REQUIRED_SEQUENCES: dict[str, list[tuple[str,str,str]]] = {
    "REQUIREMENT": [("CONDITION", "TRIGGERS", "OBLIGATION")],
    "CONSEQUENCE": [("CONDITION", "TRIGGERS", "OBLIGATION"),
                    ("CONDITION", "TRIGGERS", "RIGHT")],
    "PERMISSION":  [("CONDITION", "TRIGGERS", "RIGHT"),
                    ("OBLIGATION", "PARTY_HAS", "RIGHT")],
    "VALUE":       [("CONDITION", "TRIGGERS", "NUMERIC"),
                    ("OBLIGATION", "USES_TERM", "NUMERIC")],
    "MEANING":     [("OBLIGATION", "USES_TERM", "DEFINITION"),
                    ("RIGHT", "USES_TERM", "DEFINITION")],
    "ACTOR":       [("CONDITION", "PARTY_HAS", "OBLIGATION")],
    "GENERAL":     [("CONDITION", "TRIGGERS", "OBLIGATION"),
                    ("CONDITION", "TRIGGERS", "RIGHT")],
}

def _required_type_sequence(survivors: list[dict], dir_edges: list[tuple],
                              slots: list[dict]) -> dict:
    """
    Does the exact directed type-sequence the question needs exist in the subgraph?

    For a REQUIREMENT question we need: CONDITION →[TRIGGERS]→ OBLIGATION
    Checks whether that specific (from_type, edge_type, to_type) triple exists
    as a real directed edge in the retrieved subgraph.

    Returns:
      sequences_present  — n required sequences found
      sequences_total    — n required sequences for this slot type
      sequence_frac      — present / total
      sequences_found    — list of which sequences were found
    """
    primary_slot = (slots[0].get("slot") if slots else None) or "GENERAL"
    required = _REQUIRED_SEQUENCES.get(primary_slot, _REQUIRED_SEQUENCES["GENERAL"])

    if not required:
        return {"sequences_present": 0, "sequences_total": 0,
                "sequence_frac": None, "sequences_found": []}

    # Index: type → node indices
    by_type: dict[str, set[int]] = defaultdict(set)
    for i, nd in enumerate(survivors):
        by_type[nd.get("type","")].add(i)

    # Index: (src_idx, etype) → set of dst_idx
    out_edges: dict[tuple[int,str], set[int]] = defaultdict(set)
    for (i, j, etype, _w) in dir_edges:
        out_edges[(i, etype)].add(j)

    found = []
    for (ft, et, tt) in required:
        seq_found = False
        for src in by_type.get(ft, set()):
            for dst in out_edges.get((src, et), set()):
                if dst in by_type.get(tt, set()):
                    seq_found = True
                    break
            if seq_found:
                break
        if seq_found:
            found.append(f"{ft}→{et}→{tt}")

    present = len(found)
    total   = len(required)
    return {
        "sequences_present": present,
        "sequences_total":   total,
        "sequence_frac":     round(present / total, 4) if total > 0 else None,
        "sequences_found":   found,
    }


# ── 38. Cross-question Contrastive Signal ─────────────────────────────────────

# Module-level cache: slot_type → list of slot_scores from prior questions this session
_SLOT_SCORE_HISTORY: dict[str, list[float]] = defaultdict(list)

def _contrastive_signal(slot_score: float | None, slots: list[dict],
                         chain_completion_frac: float | None) -> dict:
    """
    Compare this question's slot_score and chain_completion_frac against the
    running session average for the same slot type.

    slot_score_gap  = this_score - session_mean  (negative = below average → risky)
    chain_gap       = this_chain - session_mean_chain

    Populated after the first question of each slot type is seen.
    Returns None gaps until enough history exists (n < 2).
    """
    primary_slot = (slots[0].get("slot") if slots else None) or "GENERAL"

    # Update history
    if slot_score is not None:
        _SLOT_SCORE_HISTORY[primary_slot].append(slot_score)

    history = _SLOT_SCORE_HISTORY[primary_slot]
    if len(history) < 2:
        return {
            "slot_score_gap":   None,
            "session_mean":     None,
            "session_n":        len(history),
            "slot_type":        primary_slot,
        }

    # Exclude current value when computing mean (leave-one-out)
    others = history[:-1]
    session_mean = round(float(np.mean(others)), 4)
    gap = round((slot_score or 0.0) - session_mean, 4)

    return {
        "slot_score_gap": gap,
        "session_mean":   session_mean,
        "session_n":      len(others),
        "slot_type":      primary_slot,
    }


# ── 24. Keyword Trap Detection ────────────────────────────────────────────────

def _keyword_trap(node_weights: list[float], ricci: dict) -> dict:
    """
    Compound signal: topic found (high anchor_frac) AND graph is bridge-heavy
    (very negative Ricci mean) → likely a keyword match without logical support.

    Calibration from 40-question eval:
      Correct answerable:  anchor_frac=0.43, ricci_mean=-0.549
      Wrong answerable:    anchor_frac=0.55, ricci_mean=-0.649  ← trap zone
      Unanswerable:        anchor_frac=0.57, ricci_mean=-0.580

    FLAG if anchor_frac > 0.50  (topic clearly present)
         AND ricci_mean  < -0.55 (graph is bridge-heavy, lacks triangulated support)

    Wrong answers fall in this zone: topic IS in document, but the retrieved
    graph has no triangulated corroboration — thin bridges only.
    """
    n = len(node_weights)
    if n == 0:
        return {"flagged": None, "confidence": None, "anchor_frac": None, "ricci_mean": None}

    anchor_frac = sum(1 for w in node_weights if w >= 1.0) / n
    ricci_mean  = (ricci or {}).get("mean")

    if ricci_mean is None:
        return {"flagged": None, "confidence": None,
                "anchor_frac": round(anchor_frac, 4), "ricci_mean": None}

    ANCHOR_THRESH = 0.50
    RICCI_THRESH  = -0.55   # flag if ricci more negative than this

    flagged = bool(anchor_frac > ANCHOR_THRESH and ricci_mean < RICCI_THRESH)

    # Soft confidence: how deep into the trap zone are we?
    anchor_excess = max(0.0, anchor_frac - ANCHOR_THRESH) / (1.0 - ANCHOR_THRESH + 1e-6)
    ricci_excess  = max(0.0, -ricci_mean  - abs(RICCI_THRESH)) / (1.0 + 1e-6)
    confidence    = round(float(min(1.0, (anchor_excess + ricci_excess) / 2.0)), 4)

    return {
        "flagged":     flagged,
        "confidence":  confidence,
        "anchor_frac": round(anchor_frac, 4),
        "ricci_mean":  round(ricci_mean,  4),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def _llm_intent_enabled() -> bool:
    """
    Optional LLM intent typing gate.
    Enabled by default; disable with TOPO_USE_LLM_INTENT=0.
    """
    return os.environ.get("TOPO_USE_LLM_INTENT", "1").strip().lower() in {"1", "true", "yes", "on"}


def _llm_intent_blend() -> float:
    """
    Blend weight for LLM intent fields.
    0.0 = deterministic only, 1.0 = LLM-only intent.
    """
    raw = os.environ.get("TOPO_LLM_INTENT_BLEND", "0.80").strip()
    try:
        v = float(raw)
    except Exception:
        v = 0.80
    return max(0.0, min(1.0, v))


def _llm_intent_mode() -> str:
    """
    Intent routing mode:
      - hybrid   : deterministic + LLM blend (default)
      - llm_only : LLM-primary fields/flags when available
    """
    mode = os.environ.get("TOPO_LLM_INTENT_MODE", "hybrid").strip().lower()
    return mode if mode in {"hybrid", "llm_only"} else "hybrid"


def _llm_question_lane(question: str, primary_slot: str) -> dict:
    """
    LLM-assisted question typing. Returns deterministic-shaped fields so scorer
    can route by lane while preserving fallback behavior.
    """
    q = (question or "").strip()
    if not q or not _llm_intent_enabled():
        return {}
    # Reuse intent extracted during parse_question_node to avoid a second LLM call.
    try:
        from core_pipeline import get_cached_question_intent as _get_cached_intent
        cached_int = _get_cached_intent(q)
        if isinstance(cached_int, dict) and cached_int:
            out_cached = dict(cached_int)
            out_cached["llm_used"] = True
            return out_cached
    except Exception:
        pass
    cache_key = (q.lower(), (primary_slot or "GENERAL").upper())
    if cache_key in _LLM_INTENT_CACHE:
        return dict(_LLM_INTENT_CACHE[cache_key])
    try:
        from core_pipeline import ask as _ask  # local import to avoid module cycle at import time
    except Exception:
        return {}

    prompt = f"""
Classify this legal contract question into structured routing fields.
Return ONLY strict JSON object with keys:
- proof_type: one of [direct_rule_clause, exact_value, definition, enumeration, customer_instance_field, external_artifact_contents, mixed]
- answer_form: one of [yes_no, date, number, percent, name, list, clause_text, mixed]
- lane: one of [direct_rule, exact_value, dependency_risk, speculative_outside, mixed]
- exact_intent, simple_extract_intent, speculative_intent, outside_document_intent, missing_artifact_intent, instance_missing_field, external_reference_intent: floats in [0,1]
- expects_direct_rule_clause: float in [0,1]
- outside_doc_needed: float in [0,1]
- asks_for_external_contents: float in [0,1]
- external_doc_type: one of [none, dpa, security_measures, pricing_page, service_specific_terms, healthcare_addendum, audit_reports, supported_countries, order_form, exhibit, schedule, appendix, annex, other]
- answer_source_type: one of [base_clause, customer_instance, external_artifact, runtime_fact, mixed]
- source_sufficiency_required: one of [clause_quote_enough, explicit_value_required, artifact_contents_required, runtime_lookup_required, mixed]
- missing_if_not_present: boolean
- customer_instance_dependency: float in [0,1]
- external_artifact_dependency: float in [0,1]
- runtime_dependency: float in [0,1]

Question: {q}
Primary slot: {(primary_slot or "GENERAL").upper()}
"""
    try:
        raw = _ask(prompt, temperature=0.0, max_tokens=260, json_mode=True)
        data = json.loads(raw)
    except Exception:
        return {}

    def _clip01(v, default=0.0):
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return float(default)

    allowed_proof = {
        "direct_rule_clause", "exact_value", "definition", "enumeration",
        "customer_instance_field", "external_artifact_contents", "mixed",
    }
    allowed_form = {"yes_no", "date", "number", "percent", "name", "list", "clause_text", "mixed"}
    allowed_lane = {"direct_rule", "exact_value", "dependency_risk", "speculative_outside", "mixed"}
    allowed_source = {"base_clause", "customer_instance", "external_artifact", "runtime_fact", "mixed"}
    allowed_suff = {"clause_quote_enough", "explicit_value_required", "artifact_contents_required", "runtime_lookup_required", "mixed"}

    out = {
        "proof_type": data.get("proof_type") if str(data.get("proof_type")) in allowed_proof else "mixed",
        "answer_form": data.get("answer_form") if str(data.get("answer_form")) in allowed_form else "mixed",
        "lane": data.get("lane") if str(data.get("lane")) in allowed_lane else "mixed",
        "exact_intent": _clip01(data.get("exact_intent")),
        "simple_extract_intent": _clip01(data.get("simple_extract_intent")),
        "speculative_intent": _clip01(data.get("speculative_intent")),
        "outside_document_intent": _clip01(data.get("outside_document_intent")),
        "missing_artifact_intent": _clip01(data.get("missing_artifact_intent")),
        "instance_missing_field": _clip01(data.get("instance_missing_field")),
        "external_reference_intent": _clip01(data.get("external_reference_intent")),
        "expects_direct_rule_clause": _clip01(data.get("expects_direct_rule_clause")),
        "outside_doc_needed": _clip01(data.get("outside_doc_needed")),
        "asks_for_external_contents": _clip01(data.get("asks_for_external_contents")),
        "external_doc_type": str(data.get("external_doc_type") or "none").strip().lower(),
        "answer_source_type": data.get("answer_source_type") if str(data.get("answer_source_type")) in allowed_source else "mixed",
        "source_sufficiency_required": data.get("source_sufficiency_required") if str(data.get("source_sufficiency_required")) in allowed_suff else "mixed",
        "missing_if_not_present": bool(data.get("missing_if_not_present", False)),
        "customer_instance_dependency": _clip01(data.get("customer_instance_dependency")),
        "external_artifact_dependency": _clip01(data.get("external_artifact_dependency")),
        "runtime_dependency": _clip01(data.get("runtime_dependency")),
        "llm_used": True,
    }
    _LLM_INTENT_CACHE[cache_key] = dict(out)
    return out


def _llm_evidence_sufficiency(question: str, survivors: list[dict], intent: dict) -> dict:
    """
    LLM pass over retrieved clauses to decide if required proof source is present.
    """
    q = (question or "").strip()
    if not q or not _llm_intent_enabled():
        return {}
    mode = str(intent.get("llm_intent_mode") or _llm_intent_mode())
    cache_key = (q.lower(), _node_key({"type": "DEFINITION", "term": f"mode::{mode}::{len(survivors)}"}))
    if cache_key in _LLM_EVIDENCE_CACHE:
        return dict(_LLM_EVIDENCE_CACHE[cache_key])
    try:
        from core_pipeline import ask as _ask
    except Exception:
        return {}

    snippets = []
    for i, n in enumerate((survivors or [])[:16], 1):
        t = str(n.get("type") or "")
        txt = str(n.get("source_text") or "")[:320].replace("\n", " ").strip()
        to = str(n.get("to") or "").strip()
        desc = str(n.get("describes") or "").strip()
        extra = f" to={to}; describes={desc}" if t == "REFERENCE" else ""
        snippets.append(f"[{i}] type={t}; text={txt}{extra}")
    context = "\n".join(snippets) if snippets else "(no retrieved clauses)"

    prompt = f"""
You are checking if retrieved clauses contain the required proof source for a legal question.
Return ONLY strict JSON with keys:
- answer_source_type: one of [base_clause, customer_instance, external_artifact, runtime_fact, mixed]
- source_sufficiency_required: one of [clause_quote_enough, explicit_value_required, artifact_contents_required, runtime_lookup_required, mixed]
- inline_answer_present: float 0..1  (is required answer source actually present in retrieved clauses?)
- external_reference_only: float 0..1 (clauses mostly reference external docs without their contents)
- customer_instance_missing: float 0..1 (needs instance-specific fields not present)
- runtime_missing: float 0..1 (needs live/current/runtime data not present)
- evidence_support_reason: short string

Question: {q}
Question intent summary: {json.dumps({
    "proof_type": intent.get("proof_type"),
    "answer_form": intent.get("answer_form"),
    "external_doc_type": intent.get("external_doc_type"),
    "outside_doc_needed": intent.get("outside_doc_needed"),
    "asks_for_external_contents": intent.get("asks_for_external_contents"),
    "answer_source_type": intent.get("answer_source_type"),
    "source_sufficiency_required": intent.get("source_sufficiency_required"),
}, ensure_ascii=True)}

Retrieved clauses:
{context}
"""
    try:
        raw = _ask(prompt, temperature=0.0, max_tokens=260, json_mode=True)
        data = json.loads(raw)
    except Exception:
        return {}

    def _clip01(v, default=0.0):
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return float(default)

    allowed_source = {"base_clause", "customer_instance", "external_artifact", "runtime_fact", "mixed"}
    allowed_suff = {"clause_quote_enough", "explicit_value_required", "artifact_contents_required", "runtime_lookup_required", "mixed"}
    out = {
        "answer_source_type": data.get("answer_source_type") if str(data.get("answer_source_type")) in allowed_source else "mixed",
        "source_sufficiency_required": data.get("source_sufficiency_required") if str(data.get("source_sufficiency_required")) in allowed_suff else "mixed",
        "inline_answer_present": _clip01(data.get("inline_answer_present")),
        "external_reference_only": _clip01(data.get("external_reference_only")),
        "customer_instance_missing": _clip01(data.get("customer_instance_missing")),
        "runtime_missing": _clip01(data.get("runtime_missing")),
        "evidence_support_reason": str(data.get("evidence_support_reason") or "").strip(),
        "llm_used": True,
    }
    _LLM_EVIDENCE_CACHE[cache_key] = dict(out)
    return out

def _question_intent(question: str, slots: list[dict]) -> dict:
    """
    Deterministic question intent features used by the answerability scorer.
    """
    q = (question or "").strip().lower()
    primary_slot = (slots[0].get("slot") if slots else None) or "GENERAL"
    primary_slot = str(primary_slot).upper()

    def _has_any(tokens: tuple[str, ...]) -> bool:
        return any(tok in q for tok in tokens)

    asks_exact_wording = _has_any((" exact ", " exactly ", " specific ", " precisely ", " detailed "))
    asks_date_like = _has_any((" date", "when ", " end date", "start date", "effective date", "term end"))
    asks_name_like = _has_any((" name of", " agreement name", "title of", "called ", "what is the name"))
    asks_list_like = _has_any((" list ", " which ", " what are ", " all ", " each ", " every "))
    asks_value_like = _has_any((" value", " amount", " total", " market share", " percentage", " price", " fee", "minutes", "share"))
    asks_percent_like = _has_any((" percent", " percentage", "%", "share"))
    asks_yesno_like = _has_any((" do ", " does ", " is ", " are ", " can ", " may ", " must ", " shall "))
    asks_definition_like = _has_any((" define ", " definition", " means ", " refer to as", " referred to as"))
    asks_exhibit_like = _has_any((" exhibit ", " schedule ", " appendix ", " annex "))
    asks_report_like = _has_any((" report", " reports", " statement", " filing", " disclosure"))
    asks_contents_like = _has_any((" contents", " contain", " included", " include", " listed", " stations"))
    asks_internal_comm = _has_any(("emails", "email", "communications", "internal", "negotiation", "discussion", "strategy", "confidential"))
    asks_future_like = _has_any((" will ", " future", " going forward", " in the future", " would ", " could ", " might "))
    asks_opinion_like = _has_any((" opinion", " think", " likely", " expect", " success", " best", " should"))
    asks_hypo_like = _has_any((" if ", " hypothetic", " suppose", " assume", " post-renegotiation", "renegotiation"))
    asks_event_like = _has_any((" occurred", " happened", " already", " previous", " past ", " disputes", " lawsuits", " legal cases"))
    asks_instance_like = _has_any((
        "this customer", "specific customer", "this agreement instance", "this account",
        "authorized purchaser", "order form", "legal name", "start date", "minimum commitment",
        "discount", "workspace id", "organizational id"
    ))
    asks_external_reference_like = _has_any((
        "pricing page", "dpa", "data processing addendum", "security measures",
        "service-specific terms", "service specific terms", "subprocessor", "sub-processors",
        "supported countries", "territories list", "healthcare addendum", "audit reports",
        "external document", "appendix", "annex"
    ))
    asks_for_external_contents = _has_any((
        "full text", "exact obligations", "exact terms", "exact language", "contents of",
        "what is in", "what are in", "listed in", "inside the", "in the full", "full dpa"
    ))

    simple_extract_intent = 0.0
    if asks_date_like:
        simple_extract_intent += 0.35
    if asks_percent_like:
        simple_extract_intent += 0.25
    if asks_yesno_like:
        simple_extract_intent += 0.20
    if primary_slot in {"REQUIREMENT", "PERMISSION", "CONSEQUENCE"}:
        simple_extract_intent += 0.15
    if asks_name_like:
        simple_extract_intent += 0.15
    simple_extract_intent = max(0.0, min(1.0, simple_extract_intent))

    # Narrow exact intent: dangerous missing-field asks, not simple fact lookup.
    exact_intent = 0.0
    if asks_exact_wording:
        exact_intent += 0.18
    if asks_exhibit_like:
        exact_intent += 0.32
    if asks_report_like:
        exact_intent += 0.24
    if asks_contents_like and asks_list_like:
        exact_intent += 0.22
    if "market share" in q:
        exact_intent += 0.30
    if "total value" in q or "exact value" in q:
        exact_intent += 0.28
    if asks_internal_comm:
        exact_intent += 0.26
    if primary_slot == "VALUE":
        exact_intent += 0.10
    exact_intent -= 0.45 * simple_extract_intent
    exact_intent = max(0.0, min(1.0, exact_intent))

    speculative_intent = 0.0
    if asks_future_like:
        speculative_intent += 0.55
    if _has_any(("plan", "planned", "planning", "roadmap", "next year", "next quarter", "forecast", "projected")):
        speculative_intent += 0.40
    if _has_any(("update", "updates", "change", "changes", "modify", "modifications")) and _has_any(("plan", "planned", "planning", "future", "next year", "next quarter")):
        speculative_intent += 0.20
    if asks_opinion_like:
        speculative_intent += 0.35
    if asks_hypo_like:
        speculative_intent += 0.30
    if asks_event_like and asks_future_like:
        speculative_intent += 0.20
    # Conditional failure / near-future hypotheticals: "if it fails", "next month", "if X happens"
    if _has_any(("next month", "next week", "next day", "next quarter")) and asks_future_like:
        speculative_intent += 0.40
    if _has_any(("if it fails", "if it doesn't", "if they fail", "if he fails", "if she fails")):
        speculative_intent += 0.45
    if q.startswith("what will ") and asks_future_like and _has_any(("if ", "when ")):
        speculative_intent += 0.35
    speculative_intent = max(0.0, min(1.0, speculative_intent))

    outside_document_intent = 0.0
    if asks_internal_comm:
        outside_document_intent += 0.45
    if _has_any(("unwritten", "not written", "off the record", "oral", "confidential")):
        outside_document_intent += 0.40
    if _has_any(("replacement technology", "new technology", "future renegotiation", "post-renegotiation")):
        outside_document_intent += 0.30
    # Named-exec questions ("who is the current CEO/president/chairman") ask about
    # runtime org facts that are never in contract documents.
    if _has_any(("current ceo", "current president", "current chairman", "current founder",
                 "current cto", "current coo", "current cfo", "who is the ceo",
                 "who is the president", "who is the chairman", "who is the founder")):
        outside_document_intent += 0.70
    outside_document_intent = max(0.0, min(1.0, outside_document_intent))

    missing_artifact_intent = 0.0
    if asks_exhibit_like:
        missing_artifact_intent += 0.70
    if asks_report_like:
        missing_artifact_intent += 0.45
    if asks_contents_like:
        missing_artifact_intent += 0.20
    missing_artifact_intent = max(0.0, min(1.0, missing_artifact_intent))

    instance_missing_field = 0.0
    if asks_instance_like:
        instance_missing_field += 0.55
    if _has_any(("this customer", "specific customer", "this account", "authorized purchaser", "order form")):
        instance_missing_field += 0.25
    if asks_date_like or asks_name_like or asks_value_like or asks_percent_like:
        instance_missing_field += 0.15
    if asks_exact_wording:
        instance_missing_field += 0.10
    instance_missing_field = max(0.0, min(1.0, instance_missing_field))

    external_reference_intent = 0.0
    if asks_external_reference_like:
        external_reference_intent += 0.55
    if asks_contents_like:
        external_reference_intent += 0.20
    if asks_exact_wording or asks_list_like:
        external_reference_intent += 0.10
    if asks_exhibit_like or asks_report_like:
        external_reference_intent += 0.15
    external_reference_intent = max(0.0, min(1.0, external_reference_intent))
    outside_doc_needed = max(
        0.0,
        min(
            1.0,
            0.55 * external_reference_intent
            + 0.35 * missing_artifact_intent
            + (0.10 if asks_for_external_contents else 0.0),
        ),
    )

    external_doc_type = "none"
    if "dpa" in q or "data processing addendum" in q:
        external_doc_type = "dpa"
    elif "security measures" in q:
        external_doc_type = "security_measures"
    elif "pricing page" in q:
        external_doc_type = "pricing_page"
    elif "service-specific terms" in q or "service specific terms" in q:
        external_doc_type = "service_specific_terms"
    elif "healthcare addendum" in q:
        external_doc_type = "healthcare_addendum"
    elif "audit report" in q or "audit reports" in q:
        external_doc_type = "audit_reports"
    elif "supported countries" in q or "territories list" in q:
        external_doc_type = "supported_countries"
    elif "order form" in q:
        external_doc_type = "order_form"
    elif "exhibit" in q:
        external_doc_type = "exhibit"
    elif "schedule" in q:
        external_doc_type = "schedule"
    elif "appendix" in q:
        external_doc_type = "appendix"
    elif "annex" in q:
        external_doc_type = "annex"
    elif asks_external_reference_like:
        external_doc_type = "other"

    # Deterministic, question-only priors for dependency typing.
    customer_specific_dependency_prior = instance_missing_field
    external_doc_dependency_prior = external_reference_intent

    # Structured answer-form typing.
    answer_form_votes = []
    if asks_yesno_like:
        answer_form_votes.append("yes_no")
    if asks_date_like:
        answer_form_votes.append("date")
    if asks_percent_like:
        answer_form_votes.append("percent")
    if asks_value_like:
        answer_form_votes.append("number")
    if asks_name_like:
        answer_form_votes.append("name")
    if asks_list_like or asks_contents_like:
        answer_form_votes.append("list")
    if asks_definition_like:
        answer_form_votes.append("clause_text")
    if not answer_form_votes:
        answer_form_votes.append("clause_text")
    answer_form_unique = sorted(set(answer_form_votes))
    answer_form = "mixed" if len(answer_form_unique) > 1 else answer_form_unique[0]

    # Structured proof typing.
    proof_votes = []
    if asks_external_reference_like or asks_exhibit_like or asks_report_like:
        proof_votes.append("external_artifact_contents")
    if asks_instance_like:
        proof_votes.append("customer_instance_field")
    if asks_definition_like:
        proof_votes.append("definition")
    if asks_list_like or asks_contents_like:
        proof_votes.append("enumeration")
    if asks_date_like or asks_value_like or asks_percent_like or asks_name_like:
        proof_votes.append("exact_value")
    if asks_yesno_like or primary_slot in {"REQUIREMENT", "PERMISSION", "CONSEQUENCE"}:
        proof_votes.append("direct_rule_clause")
    if not proof_votes:
        proof_votes.append("direct_rule_clause" if primary_slot in {"REQUIREMENT", "PERMISSION", "CONSEQUENCE"} else "mixed")
    proof_unique = sorted(set(proof_votes))
    proof_type = "mixed" if len(proof_unique) > 1 else proof_unique[0]

    expects_direct_rule_clause = 0.0
    if asks_yesno_like:
        expects_direct_rule_clause += 0.45
    if primary_slot in {"REQUIREMENT", "PERMISSION", "CONSEQUENCE"}:
        expects_direct_rule_clause += 0.35
    if _has_any(("must ", " shall ", " may ", " may not", " allowed", " not allowed", " required to", " liable if", "without ")):
        expects_direct_rule_clause += 0.25
    if asks_external_reference_like or asks_exhibit_like or asks_report_like:
        expects_direct_rule_clause -= 0.20
    expects_direct_rule_clause = max(0.0, min(1.0, expects_direct_rule_clause))

    intent = {
        "primary_slot": primary_slot,
        "proof_type": proof_type,
        "answer_form": answer_form,
        "expects_direct_rule_clause": round(float(expects_direct_rule_clause), 4),
        "customer_specific_dependency_prior": round(float(customer_specific_dependency_prior), 4),
        "external_doc_dependency_prior": round(float(external_doc_dependency_prior), 4),
        "exact_intent": round(float(exact_intent), 4),
        "simple_extract_intent": round(float(simple_extract_intent), 4),
        "speculative_intent": round(float(speculative_intent), 4),
        "outside_document_intent": round(float(outside_document_intent), 4),
        "missing_artifact_intent": round(float(missing_artifact_intent), 4),
        "instance_missing_field": round(float(instance_missing_field), 4),
        "external_reference_intent": round(float(external_reference_intent), 4),
        "outside_doc_needed": round(float(outside_doc_needed), 4),
        "asks_for_external_contents": round(1.0 if asks_for_external_contents else 0.0, 4),
        "external_doc_type": external_doc_type,
        "answer_source_type": "mixed",
        "source_sufficiency_required": "mixed",
        "missing_if_not_present": False,
        "customer_instance_dependency": round(float(instance_missing_field), 4),
        "external_artifact_dependency": round(float(external_reference_intent), 4),
        "runtime_dependency": 0.0,
        "asks_exact_wording": asks_exact_wording,
        "asks_date_like": asks_date_like,
        "asks_name_like": asks_name_like,
        "asks_list_like": asks_list_like,
        "asks_value_like": asks_value_like,
        "asks_percent_like": asks_percent_like,
        "asks_yesno_like": asks_yesno_like,
        "asks_definition_like": asks_definition_like,
        "asks_exhibit_like": asks_exhibit_like,
        "asks_report_like": asks_report_like,
        "asks_contents_like": asks_contents_like,
        "asks_internal_comm": asks_internal_comm,
        "asks_future_like": asks_future_like,
        "asks_opinion_like": asks_opinion_like,
        "asks_hypo_like": asks_hypo_like,
        "asks_event_like": asks_event_like,
        "asks_instance_like": asks_instance_like,
        "asks_external_reference_like": asks_external_reference_like,
    }
    # Preserve deterministic baseline for diagnostics / fallback.
    intent["deterministic_only"] = {
        "proof_type": intent.get("proof_type"),
        "answer_form": intent.get("answer_form"),
        "lane": "mixed",
        "exact_intent": intent.get("exact_intent"),
        "simple_extract_intent": intent.get("simple_extract_intent"),
        "speculative_intent": intent.get("speculative_intent"),
        "outside_document_intent": intent.get("outside_document_intent"),
        "missing_artifact_intent": intent.get("missing_artifact_intent"),
        "instance_missing_field": intent.get("instance_missing_field"),
        "external_reference_intent": intent.get("external_reference_intent"),
        "expects_direct_rule_clause": intent.get("expects_direct_rule_clause"),
        "outside_doc_needed": intent.get("outside_doc_needed"),
        "asks_for_external_contents": intent.get("asks_for_external_contents"),
        "external_doc_type": intent.get("external_doc_type"),
        "answer_source_type": intent.get("answer_source_type"),
        "source_sufficiency_required": intent.get("source_sufficiency_required"),
        "missing_if_not_present": intent.get("missing_if_not_present"),
        "customer_instance_dependency": intent.get("customer_instance_dependency"),
        "external_artifact_dependency": intent.get("external_artifact_dependency"),
        "runtime_dependency": intent.get("runtime_dependency"),
    }

    llm_int = _llm_question_lane(question, primary_slot)
    if llm_int:
        mode = _llm_intent_mode()
        if mode == "llm_only":
            # Pure LLM intent mode (aggressive/high-recall profile).
            def _llm01(key: str, default: float = 0.0) -> float:
                try:
                    v = llm_int.get(key, default)
                    return round(max(0.0, min(1.0, float(v))), 4)
                except Exception:
                    return round(float(default), 4)

            intent["exact_intent"] = _llm01("exact_intent", intent.get("exact_intent", 0.0))
            intent["simple_extract_intent"] = _llm01("simple_extract_intent", intent.get("simple_extract_intent", 0.0))
            intent["speculative_intent"] = _llm01("speculative_intent", intent.get("speculative_intent", 0.0))
            intent["outside_document_intent"] = _llm01("outside_document_intent", intent.get("outside_document_intent", 0.0))
            intent["missing_artifact_intent"] = _llm01("missing_artifact_intent", intent.get("missing_artifact_intent", 0.0))
            intent["instance_missing_field"] = _llm01("instance_missing_field", intent.get("instance_missing_field", 0.0))
            intent["external_reference_intent"] = _llm01("external_reference_intent", intent.get("external_reference_intent", 0.0))
            intent["expects_direct_rule_clause"] = _llm01("expects_direct_rule_clause", intent.get("expects_direct_rule_clause", 0.0))
            intent["outside_doc_needed"] = _llm01("outside_doc_needed", intent.get("outside_doc_needed", 0.0))
            intent["asks_for_external_contents"] = _llm01("asks_for_external_contents", intent.get("asks_for_external_contents", 0.0))

            proof_type = str(llm_int.get("proof_type") or intent.get("proof_type") or "mixed").strip()
            answer_form = str(llm_int.get("answer_form") or intent.get("answer_form") or "mixed").strip()
            lane = str(llm_int.get("lane") or "mixed").strip()
            ext_type = str(llm_int.get("external_doc_type") or intent.get("external_doc_type") or "none").strip().lower()
            intent["proof_type"] = proof_type
            intent["answer_form"] = answer_form
            intent["external_doc_type"] = ext_type
            intent["lane"] = lane

            # LLM-derived structured booleans used downstream.
            intent["asks_yesno_like"] = answer_form == "yes_no" or proof_type == "direct_rule_clause"
            intent["asks_date_like"] = answer_form == "date"
            intent["asks_percent_like"] = answer_form == "percent"
            intent["asks_name_like"] = answer_form == "name"
            intent["asks_list_like"] = answer_form == "list"
            intent["asks_value_like"] = answer_form in {"number", "percent", "date"}
            intent["asks_definition_like"] = proof_type == "definition" or primary_slot == "MEANING"
            intent["asks_contents_like"] = bool(intent["asks_for_external_contents"] >= 0.5 or proof_type in {"enumeration", "external_artifact_contents"})
            intent["asks_exhibit_like"] = ext_type in {"exhibit", "schedule", "appendix", "annex"}
            intent["asks_report_like"] = ext_type == "audit_reports"
            intent["asks_external_reference_like"] = bool(ext_type != "none" or proof_type == "external_artifact_contents" or intent["outside_doc_needed"] >= 0.35)
            intent["asks_instance_like"] = bool(proof_type == "customer_instance_field" or intent["instance_missing_field"] >= 0.50)
            intent["asks_exact_wording"] = bool(intent["exact_intent"] >= 0.50)
            intent["asks_future_like"] = bool(intent["speculative_intent"] >= 0.45)
            intent["asks_hypo_like"] = bool(intent["speculative_intent"] >= 0.60)
            intent["asks_opinion_like"] = bool(intent["outside_document_intent"] >= 0.40 and intent["speculative_intent"] >= 0.35)
            intent["asks_event_like"] = bool(intent["outside_document_intent"] >= 0.35 and intent["speculative_intent"] >= 0.20)
            intent["asks_internal_comm"] = bool(intent["outside_document_intent"] >= 0.45)
            intent["customer_specific_dependency_prior"] = round(float(intent["instance_missing_field"]), 4)
            intent["external_doc_dependency_prior"] = round(float(intent["external_reference_intent"]), 4)
            intent["llm_blend_weight"] = 1.0
        else:
            # LLM-led routing with deterministic guardrails as fallback.
            blend = _llm_intent_blend()
            for k in (
                "exact_intent", "simple_extract_intent", "speculative_intent",
                "outside_document_intent", "missing_artifact_intent",
                "instance_missing_field", "external_reference_intent",
                "expects_direct_rule_clause", "outside_doc_needed",
                "asks_for_external_contents", "customer_instance_dependency",
                "external_artifact_dependency", "runtime_dependency",
            ):
                try:
                    b = float(intent.get(k, 0.0))
                    l = float(llm_int.get(k, b))
                    intent[k] = round(max(0.0, min(1.0, (1.0 - blend) * b + blend * l)), 4)
                except Exception:
                    pass
            if llm_int.get("proof_type"):
                intent["proof_type"] = llm_int["proof_type"]
            if llm_int.get("answer_form"):
                intent["answer_form"] = llm_int["answer_form"]
            if llm_int.get("external_doc_type"):
                intent["external_doc_type"] = llm_int["external_doc_type"]
            if llm_int.get("answer_source_type"):
                intent["answer_source_type"] = llm_int["answer_source_type"]
            if llm_int.get("source_sufficiency_required"):
                intent["source_sufficiency_required"] = llm_int["source_sufficiency_required"]
            intent["missing_if_not_present"] = bool(llm_int.get("missing_if_not_present", intent.get("missing_if_not_present", False)))
            intent["lane"] = llm_int.get("lane", "mixed")
            intent["llm_blend_weight"] = blend
        intent["llm_intent_mode"] = mode
        intent["llm_intent_used"] = True
    else:
        intent["lane"] = "mixed"
        intent["llm_blend_weight"] = 0.0
        intent["llm_intent_mode"] = "disabled"
        intent["llm_intent_used"] = False
    intent["question_text"] = q
    return intent


def _direct_explicit_support(question: str, survivors: list[dict], slots: list[dict], intent: dict) -> dict:
    """
    Deterministic clause-level support score in [0,1].
    Higher means a single clause directly states the requested answer type.
    """
    import re

    q = (question or "").lower()
    stop = {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "by", "with",
        "what", "which", "is", "are", "was", "were", "does", "do", "did", "under",
        "this", "that", "it", "as", "at", "from",
    }
    q_tokens = [t for t in re.findall(r"[a-z0-9]+", q) if len(t) >= 3 and t not in stop]
    q_set = set(q_tokens)

    asks_date = bool(intent.get("asks_date_like"))
    asks_percent = bool(intent.get("asks_percent_like"))
    asks_value = bool(intent.get("asks_value_like"))
    asks_name = bool(intent.get("asks_name_like"))
    asks_list = bool(intent.get("asks_list_like")) or bool(intent.get("asks_contents_like"))
    asks_yesno = bool(intent.get("asks_yesno_like")) or str(intent.get("primary_slot") or "") in {"REQUIREMENT", "PERMISSION", "CONSEQUENCE"}

    date_re = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}|\d{1,2}/\d{1,2}/\d{2,4}", re.I)
    pct_re = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%")
    num_re = re.compile(r"[\$£€]?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|usd|dollars?|days?|months?|years?|minutes?|hours?))?", re.I)

    best = 0.0
    best_idx = None
    best_trace = {}

    for idx, node in enumerate(survivors):
        raw = " ".join(
            str(node.get(k) or "")
            for k in ("source_text", "action", "party", "term", "right", "definition", "trigger", "consequence", "applies_to", "value", "unit")
        ).strip()
        if not raw:
            continue
        txt = raw.lower()
        tset = set(t for t in re.findall(r"[a-z0-9]+", txt) if len(t) >= 3 and t not in stop)
        overlap = len(q_set & tset) / max(1, len(q_set)) if q_set else 0.0
        R = max(0.0, min(1.0, overlap))

        type_match = 0.0
        if asks_date:
            type_match = max(type_match, 1.0 if date_re.search(raw) else 0.0)
        if asks_percent:
            type_match = max(type_match, 1.0 if pct_re.search(raw) else 0.0)
        if asks_value:
            type_match = max(type_match, 0.9 if num_re.search(raw) else 0.0)
        if asks_name:
            if ("this agreement" in txt) or ("means" in txt) or ("defined as" in txt):
                type_match = max(type_match, 0.8)
        if asks_list:
            itemish = raw.count(";") + raw.count(",")
            if itemish >= 2:
                type_match = max(type_match, 0.8)
        if asks_yesno:
            if any(k in txt for k in ("shall", "must", "may not", "shall not", "no fees", "not pay", "agrees to")):
                type_match = max(type_match, 0.85)
        if type_match == 0.0:
            type_match = min(0.75, 0.65 * R)

        explicitness = 1.0 if any(k in txt for k in (" shall ", " must ", " is ", " means ", " will ", " shall not ", " no ")) else 0.35
        p_score = max(0.0, min(1.0, float(node.get("_score", 0.5))))
        clause_support = max(0.0, min(1.0, 0.45 * R + 0.35 * type_match + 0.15 * explicitness + 0.05 * p_score))

        if clause_support > best:
            best = clause_support
            best_idx = idx
            best_trace = {
                "relevance_R": round(R, 4),
                "type_match_T": round(type_match, 4),
                "explicitness_F": round(explicitness, 4),
                "provenance_P": round(p_score, 4),
                "text": raw[:220],
                "node_type": node.get("type"),
            }

    return {
        "score": round(float(best), 4),
        "best_node_index": best_idx,
        "trace": best_trace,
    }


def _consensus_vote_stats(survivors: list[dict]) -> dict:
    """
    Summarize consensus strength across query variants from survivor vote fields.
    Expects nodes to carry `_vote_frac` and `_votes` from consensus retrieval.
    """
    fracs = []
    votes = []
    for n in survivors or []:
        try:
            vf = n.get("_vote_frac")
            if vf is not None:
                fracs.append(max(0.0, min(1.0, float(vf))))
        except Exception:
            pass
        try:
            vv = n.get("_votes")
            if vv is not None:
                votes.append(max(0.0, float(vv)))
        except Exception:
            pass
    if not fracs:
        return {
            "n_with_vote_frac": 0,
            "mean_vote_frac": None,
            "p75_vote_frac": None,
            "high_vote_frac": None,
            "max_vote_frac": None,
            "n_vote_ge2": int(sum(1 for v in votes if v >= 2.0)),
            "n_vote_ge3": int(sum(1 for v in votes if v >= 3.0)),
            "mean_votes": (round(float(np.mean(votes)), 4) if votes else None),
        }
    arr = np.array(fracs, dtype=float)
    return {
        "n_with_vote_frac": int(arr.size),
        "mean_vote_frac": round(float(arr.mean()), 4),
        "p75_vote_frac": round(float(np.percentile(arr, 75)), 4),
        "high_vote_frac": round(float((arr >= 0.60).mean()), 4),
        "max_vote_frac": round(float(arr.max()), 4),
        "n_vote_ge2": int(sum(1 for v in votes if v >= 2.0)),
        "n_vote_ge3": int(sum(1 for v in votes if v >= 3.0)),
        "mean_votes": (round(float(np.mean(votes)), 4) if votes else None),
    }


def _external_dependency_evidence(survivors: list[dict], question: str, intent: dict | None = None) -> dict:
    """
    Retrieval-aware evidence for external-document dependency.
    Distinguishes "reference-only mentions" from "inline substantive content".
    """
    q = (question or "").lower()
    intent = intent or {}
    q_doc = str(intent.get("external_doc_type") or "none").strip().lower()

    doc_aliases = {
        "dpa": ("dpa", "data processing addendum"),
        "security_measures": ("security measures",),
        "pricing_page": ("pricing page",),
        "service_specific_terms": ("service-specific terms", "service specific terms"),
        "healthcare_addendum": ("healthcare addendum",),
        "audit_reports": ("audit report", "audit reports"),
        "supported_countries": ("supported countries", "territories"),
        "order_form": ("order form",),
        "exhibit": ("exhibit",),
        "schedule": ("schedule",),
        "appendix": ("appendix",),
        "annex": ("annex",),
    }
    all_aliases = tuple(sorted({a for vals in doc_aliases.values() for a in vals}))

    pointer_phrases = (
        "incorporated by reference", "available at", "located at", "at:", "see ",
        "as set forth in", "according to", "will comply with", "comply with the",
        "provided by", "provided in",
    )
    inline_detail_markers = (
        " must ", " shall ", " may ", " may not ", " cannot ", " prohibited ",
        " means ", " includes ", " no more than ", " at least ", "within ", "%", "$",
    )

    total = max(1, len(survivors))
    ext_related = 0
    ref_only = 0
    inline_substance = 0
    q_doc_hits = 0

    q_aliases = doc_aliases.get(q_doc, ()) if q_doc in doc_aliases else ()
    asks_external_contents = float(intent.get("asks_for_external_contents") or 0.0) >= 0.5

    for n in survivors:
        src = str(n.get("source_text") or "").lower()
        nt = str(n.get("type") or "").upper()
        to_txt = str(n.get("to") or "").lower()
        desc_txt = str(n.get("describes") or "").lower()
        joined = f"{src} {to_txt} {desc_txt}".strip()

        has_external_anchor = any(a in joined for a in all_aliases) or ("http" in joined and "openai.com" in joined)
        if not has_external_anchor:
            continue
        ext_related += 1

        if q_aliases and any(a in joined for a in q_aliases):
            q_doc_hits += 1

        is_reference_node = nt == "REFERENCE"
        has_pointer_language = any(p in joined for p in pointer_phrases)
        has_inline_detail = any(m in f" {joined} " for m in inline_detail_markers)
        long_enough = len(src.split()) >= 16

        looks_ref_only = is_reference_node or (has_pointer_language and not has_inline_detail)
        if asks_external_contents and ("comply with" in joined or "incorporated by reference" in joined):
            looks_ref_only = True

        if looks_ref_only:
            ref_only += 1
        elif (not is_reference_node) and (has_inline_detail or long_enough):
            inline_substance += 1

    ext_related_ratio = ext_related / total
    if ext_related <= 0:
        ref_only_ratio = 0.0
        inline_substance_ratio = 0.0
    else:
        ref_only_ratio = ref_only / ext_related
        inline_substance_ratio = inline_substance / ext_related

    q_doc_match_ratio = (q_doc_hits / ext_related) if ext_related > 0 else 0.0
    return {
        "external_related_nodes": int(ext_related),
        "external_related_ratio": round(float(ext_related_ratio), 4),
        "external_ref_only_ratio": round(float(max(0.0, min(1.0, ref_only_ratio))), 4),
        "external_inline_substance_ratio": round(float(max(0.0, min(1.0, inline_substance_ratio))), 4),
        "question_doc_match_ratio": round(float(max(0.0, min(1.0, q_doc_match_ratio))), 4),
    }


def compute_all(survivors: list[dict], graph_index,
                question: str, q_node: dict, slots: list[dict],
                llm_evidence_sufficiency: dict | None = None) -> dict:
    """
    Compute all topology metrics over the surviving node subgraph.
    Returns raw numbers only — no verdicts or conclusions applied.
    """
    if not survivors or graph_index is None:
        return {
            "question": question,
            "n_nodes": len(survivors),
            "n_edges": 0,
            "typed_path_coverage": None,
            "betti":               None,
            "persistent_homology": None,
            "hodge":               None,
            "sheaf":               None,
            "conductance":         None,
            "ricci":               None,
            "spectral":            None,
            "kcore":               None,
            "path_entropy":        None,
            "bipartite":           None,
            "weighted": {
                "n_anchors":    0,
                "anchor_frac":  None,
                "mean_weight":  None,
                "node_weights": [],
                "weighted_consistent_frac": None,
                "anchor_weighted_edges":    0,
                "weighted_gradient_frac":   None,
                "weighted_harmonic_frac":   None,
                "weighted_mean":            None,
                "weighted_min":             None,
                "anchor_bridge_score":      None,
            },
            "direction": {
                "q_party_found":       False,
                "party_match_frac":    None,
                "party_mismatch_frac": None,
                "edge_role_inversion": None,
            },
            "chain": {
                "n_condition_nodes":    0,
                "reachable_conditions": 0,
                "chain_strength":       None,
                "trigger_edge_weight":  None,
                "conditions_required":  False,
            },
            "efficiency_lcc":    {"efficiency": None, "n_lcc": 0, "lcc_frac": None},
            "type_metrics":      {"obl_frac": None, "type_entropy": None, "type_counts": {}},
            "clustering":        {"mean_local_cc": None, "global_cc": None, "n_triangles": 0},
            "keyword_trap":      {"flagged": None, "confidence": None, "anchor_frac": None, "ricci_mean": None},
            "eff_resistance":    {"mean_resistance": None, "max_resistance": None, "n_anchor_pairs": 0, "resistances": []},
            "anchor_betweenness":{"mean_anchor_bc": None, "max_anchor_bc": None, "anchor_bc": []},
            "forman_ricci":      {"mean": None, "min": None, "max": None, "n_edges": 0},
            "transition_entropy":{"entropy": None, "n_pairs": 0, "top_transitions": []},
            "weighted_fiedler":  {"lambda2_weighted": None, "spectral_gap_weighted": None},
            "anchor_jaccard":    {"mean_jaccard": None, "min_jaccard": None, "n_anchor_pairs": 0, "jaccards": []},
            "von_neumann":       {"entropy": None, "normalised_entropy": None, "n_nonzero": 0},
            "nonbacktracking":   {"spectral_radius": None, "ramanujan_ratio": None, "n_dir_edges": 0},
            "magnetic":          {"frustration": None, "optimal_q": None, "lambda_min_q0": None},
            "persistent_laplacian": {"lambda2_profile": [], "beta0_profile": [], "delta_lambda2": None, "stability": None},
            "heat_kernel":       {"k_vals": None, "k_normalised": None, "decay_rate": None, "t_half": None},
            "answer_path": {
                "path_exists": False, "path_length": None, "path_bottleneck": None,
                "path_score": None, "path_edge_types": [], "same_section_frac": None,
                "n_targets": 0, "n_well_connected": 0,
                "target_weight_mean": None, "target_weight_max": None, "slot_type": "GENERAL",
            },
            "target_centrality": {
                "target_mean_degree": None, "target_min_degree": None,
                "target_orphan_frac": None, "target_type_frac": None, "n_targets": 0,
            },
            "causal_reachability": {
                "causal_reachable": 0, "causal_frac": None,
                "semantic_edge_frac": None, "anchor_trigger_count": 0,
            },
            "anchor_convergence": {
                "max_anchors_to_target": 0, "anchor_agreement_score": None,
                "best_target_type": None, "n_reachable_targets": 0,
            },
            "chain_completeness": {
                "n_conditions": 0, "complete_chains": 0, "dangling_conditions": 0,
                "chain_completion_frac": None, "required_type_present": False,
            },
            "anchor_type_match": {
                "anchor_type_score": None, "anchor_type_diversity": None,
                "anchor_types": {}, "dominant_anchor_type": None,
            },
            "slot_mismatch_depth": {
                "mismatch_depth": None, "exact_frac": None,
                "adjacent_frac": None, "distant_frac": None,
            },
            "required_sequence": {
                "sequences_present": 0, "sequences_total": 0,
                "sequence_frac": None, "sequences_found": [],
            },
            "contrastive": {
                "slot_score_gap": None, "session_mean": None,
                "session_n": 0, "slot_type": "GENERAL",
            },
            "hitting_time":      {"mean_hitting_time": None, "min_hitting_time": None, "n_transient": 0},
            "answer_pagerank":   {"answer_pr_mean": None, "answer_pr_max": None, "answer_pr_rank": None, "n_targets": 0},
            "feedforward_loops": {"n_ffl": 0, "ffl_per_edge": 0.0, "typed_ffl": 0},
            "a2a_resistance":    {"mean_a2a_resistance": None, "min_a2a_resistance": None, "n_pairs": 0},
            "deg_entropy":       {"deg_entropy": None, "hub_frac": None, "isolated_frac": None, "gini": None},
            "acyclicity":        {"acyclicity_score": 1.0, "n_back_edges": 0, "n_scc_gt1": 0},
            "question_intent":   _question_intent(question, slots),
            "llm_evidence_sufficiency": {
                "answer_source_type": "mixed",
                "source_sufficiency_required": "mixed",
                "inline_answer_present": 0.0,
                "external_reference_only": 0.0,
                "customer_instance_missing": 0.0,
                "runtime_missing": 0.0,
                "evidence_support_reason": "",
                "llm_used": False,
            },
            "consensus_vote_stats": {
                "n_with_vote_frac": 0,
                "mean_vote_frac": None,
                "p75_vote_frac": None,
                "high_vote_frac": None,
                "mean_votes": None,
            },
            "external_dependency_evidence": {
                "external_related_nodes": 0,
                "external_related_ratio": 0.0,
                "external_ref_only_ratio": 0.0,
                "external_inline_substance_ratio": 0.0,
                "question_doc_match_ratio": 0.0,
            },
            "direct_explicit_support": {"score": 0.0, "best_node_index": None, "trace": {}},
            "value_tag_match":   {"has_tags": False, "match_score": None, "best_overlap": 0, "required_tags": []},
            "placeholder_values":{"numeric_count": 0, "placeholder_count": 0, "real_count": 0, "placeholder_frac": None, "all_placeholders": False},
            "topo_pred":         {"predicted": None, "score": None, "confidence": "low", "signals": {}},
        }

    n, dir_edges, undir_edges, keys, key_to_idx = _build_subgraph(survivors, graph_index)

    # Centrality weights: 1.0 at anchors, decaying outward
    node_weights = _anchor_weights(survivors, q_node, slots, undir_edges)

    # Compute once — shared by multiple metrics
    ricci_result  = _ollivier_ricci(n, undir_edges)
    tpc_result    = _typed_path_coverage(survivors, keys, key_to_idx, dir_edges, slots)
    chain_result  = _chain_completeness(survivors, dir_edges, slots)
    intent_result = _question_intent(question, slots)
    des_result    = _direct_explicit_support(question, survivors, slots, intent_result)
    llm_ev_result = (llm_evidence_sufficiency or {}) if isinstance(llm_evidence_sufficiency, dict) else {}
    if not llm_ev_result:
        llm_ev_result = _llm_evidence_sufficiency(question, survivors, intent_result)
    vote_stats    = _consensus_vote_stats(survivors)
    extdep_result = _external_dependency_evidence(survivors, question, intent_result)

    result = {
        "question": question,
        "n_nodes": n,
        "n_edges": len(undir_edges),
        "typed_path_coverage": tpc_result,
        "betti":               _betti(n, undir_edges),
        "persistent_homology": _persistent_homology(n, undir_edges),
        "hodge":               _hodge(n, dir_edges, survivors),
        "sheaf":               _sheaf_consistency(survivors, dir_edges),
        "conductance":         _conductance(survivors, graph_index),
        "ricci":               ricci_result,
        "spectral":            _spectral_gap(n, undir_edges),
        "kcore":               _kcore(n, undir_edges, survivors),
        "path_entropy":        _path_entropy(n, undir_edges),
        "bipartite":           _bipartite_projection(survivors, dir_edges),
        # New structural metrics
        "efficiency_lcc":      _global_efficiency_lcc(n, undir_edges),
        "type_metrics":        _type_metrics(survivors),
        "clustering":          _clustering_coefficient(n, undir_edges),
        "keyword_trap":        _keyword_trap(node_weights, ricci_result),
        # New structural metrics (batch 2)
        "eff_resistance":      _effective_resistance(n, undir_edges, node_weights),
        "anchor_betweenness":  _anchor_betweenness(n, undir_edges, node_weights),
        "forman_ricci":        _forman_ricci(n, undir_edges),
        "transition_entropy":  _type_transition_entropy(survivors, dir_edges),
        "weighted_fiedler":    _weighted_fiedler(n, undir_edges, node_weights),
        "anchor_jaccard":      _anchor_jaccard(n, undir_edges, node_weights),
        # Novel quantum/spectral metrics (batch 3)
        "von_neumann":         _von_neumann_entropy(n, undir_edges),
        "nonbacktracking":     _nonbacktracking_radius(n, undir_edges),
        "magnetic":            _magnetic_frustration(n, undir_edges, dir_edges),
        "persistent_laplacian":_persistent_laplacian(n, undir_edges, node_weights),
        "heat_kernel":         _heat_kernel_trace(n, undir_edges),
        "answer_path":         _answer_path_topology(n, survivors, q_node, slots,
                                                      undir_edges, node_weights),
        # NEW: targeted answer-quality signals
        "target_centrality":   _target_centrality(survivors, node_weights, undir_edges, slots),
        "causal_reachability": _causal_reachability(survivors, node_weights, dir_edges, slots),
        "anchor_convergence":  _anchor_convergence(survivors, node_weights, undir_edges, slots),
        # NEW: chain / type reasoning signals
        "chain_completeness":  chain_result,
        "anchor_type_match":   _anchor_type_match(survivors, node_weights, slots),
        "slot_mismatch_depth": _slot_mismatch_depth(survivors, slots),
        "required_sequence":   _required_type_sequence(survivors, dir_edges, slots),
        "contrastive":         _contrastive_signal(
                                   tpc_result.get("score"),
                                   slots,
                                   chain_result.get("chain_completion_frac"),
                               ),
        # NEW: advanced math signals
        "hitting_time":        _hitting_time(n, survivors, node_weights, undir_edges, slots),
        "answer_pagerank":     _answer_pagerank(n, survivors, dir_edges, slots),
        "feedforward_loops":   _feedforward_loops(n, survivors, dir_edges),
        "a2a_resistance":      _anchor_to_answer_resistance(n, survivors, undir_edges, node_weights, slots),
        "deg_entropy":         _degree_sequence_entropy(n, undir_edges),
        "acyclicity":          _directed_acyclicity(n, dir_edges),
        "question_intent":     intent_result,
        "llm_evidence_sufficiency": llm_ev_result,
        "consensus_vote_stats": vote_stats,
        "external_dependency_evidence": extdep_result,
        "direct_explicit_support": des_result,
        # Flow math: metrics re-computed weighted by distance from question anchors
        "weighted": {
            "n_anchors":      sum(1 for w in node_weights if w >= 1.0),
            "anchor_frac":    round(sum(1 for w in node_weights if w >= 1.0) / n, 4) if n > 0 else None,
            "mean_weight":    round(float(np.mean(node_weights)), 4) if node_weights else None,
            # Per-node weights for frontend heatmap (parallel to survivors list)
            "node_weights":   [round(w, 4) for w in node_weights],
            **_weighted_sheaf(survivors, dir_edges, node_weights),
            **_weighted_hodge(n, dir_edges, node_weights),
            **_weighted_ricci(n, undir_edges, node_weights),
        },
        # C: party-role directional mismatch
        "direction": _directional_mismatch(survivors, q_node, node_weights, dir_edges),
        # B: condition chain strength (path persistence)
        "chain":     _condition_chain_strength(survivors, node_weights, dir_edges, undir_edges),
        # Value tag match: do retrieved nodes actually contain what the question needs?
        "value_tag_match":      _value_tag_match(survivors, question, slots),
        # Placeholder detection: are NUMERIC nodes variables (x,y,I) vs real values?
        "placeholder_values":   _placeholder_value_ratio(survivors, slots),
    }
    result["topo_pred"] = _predict_answerability(result)
    return result


def _placeholder_value_ratio(survivors: list[dict], slots: list[dict]) -> dict:
    """
    Detect when retrieved NUMERIC nodes contain placeholder variables rather
    than actual stated values.

    Documents sometimes define formulas using variables (x, y, I, n) instead
    of real numbers. A question asking for a specific quantity is unanswerable
    if the relevant NUMERIC nodes all hold variables, not values.

    Placeholder patterns: single letters (x, y, I, n), variable expressions
    ("x number", "y percentage", "I amounts"), or formula symbols.

    Returns:
        numeric_count     — NUMERIC nodes in survivors
        placeholder_count — nodes with placeholder values
        real_count        — nodes with actual numeric values
        placeholder_frac  — fraction that are placeholders (high → unanswerable)
        all_placeholders  — True if every NUMERIC node is a placeholder
    """
    import re

    # Matches real numeric values: digits, decimals, percentages, currencies
    _REAL_VALUE = re.compile(
        r'^[\$£€]?\s*\d[\d,\.]*\s*(%|k|m|bn|million|billion|days?|months?|years?|mbps|usd)?$',
        re.IGNORECASE
    )
    # Placeholder: single letter, or starts with a letter followed by words/spaces
    _PLACEHOLDER = re.compile(
        r'^[a-zA-Z](\s+\w+)?$'   # "x", "y", "I", "n", "x number", "y percentage"
    )

    numeric_nodes = [s for s in survivors if s.get("type") == "NUMERIC"]
    if not numeric_nodes:
        return {"numeric_count": 0, "placeholder_count": 0, "real_count": 0,
                "placeholder_frac": None, "all_placeholders": False}

    placeholder_count = 0
    real_count = 0
    for node in numeric_nodes:
        val = (node.get("value") or "").strip()
        if _PLACEHOLDER.match(val) and not _REAL_VALUE.match(val):
            placeholder_count += 1
        else:
            real_count += 1

    frac = round(placeholder_count / len(numeric_nodes), 4) if numeric_nodes else None

    return {
        "numeric_count":     len(numeric_nodes),
        "placeholder_count": placeholder_count,
        "real_count":        real_count,
        "placeholder_frac":  frac,
        "all_placeholders":  placeholder_count == len(numeric_nodes),
    }


def _value_tag_match(survivors: list[dict], question: str, slots: list[dict]) -> dict:
    """
    Check whether the retrieved NUMERIC/DEFINITION nodes actually contain
    value_tags that match what the question is asking for.

    A VALUE/NUMERIC question asking for "total revenue" needs a NUMERIC node
    tagged ["revenue","currency","total"]. If no such node exists in survivors,
    the answer is structurally absent — not just hard to find.

    Returns:
        has_tags      — True if any survivor has value_tags at all
        match_score   — fraction of question concept words found in any node's tags
        best_overlap  — highest single-node tag overlap count
        required_tags — tags extracted from question + slot
    """
    # ── Extract concept words from question ─────────────────────────────
    q_lower = question.lower()

    # Semantic concept map: question keywords → value_tag concepts
    _CONCEPT_MAP = {
        "revenue":    ["revenue", "income", "earnings", "currency"],
        "profit":     ["profit", "earnings", "income", "currency"],
        "income":     ["income", "revenue", "earnings", "currency"],
        "cost":       ["cost", "currency", "payment", "fee"],
        "price":      ["price", "currency", "cost", "payment"],
        "fee":        ["fee", "payment", "currency"],
        "salary":     ["salary", "compensation", "currency"],
        "payment":    ["payment", "currency", "fee"],
        "viewer":     ["viewer", "subscriber", "count", "audience"],
        "subscriber": ["subscriber", "viewer", "count", "household"],
        "audience":   ["audience", "viewer", "subscriber", "count"],
        "market":     ["market", "share", "percent"],
        "share":      ["share", "percent", "market"],
        "rate":       ["rate", "percent", "interest"],
        "percent":    ["percent", "rate", "share"],
        "number":     ["count", "quantity", "number"],
        "count":      ["count", "number", "quantity"],
        "total":      ["total", "aggregate", "sum"],
        "aggregate":  ["aggregate", "total", "sum"],
        "mean":       ["definition", "term"],
        "define":     ["definition", "term"],
        "definition": ["definition", "term"],
        "means":      ["definition", "term"],
    }

    required: set[str] = set()
    for word, concepts in _CONCEPT_MAP.items():
        if word in q_lower:
            required.update(concepts)

    # Only fire on question keywords — slot type is too broad and causes
    # false negatives on legitimate VALUE questions (e.g. "notice period").
    if not required:
        return {"has_tags": False, "match_score": None,
                "best_overlap": 0, "required_tags": []}

    # ── Check survivor nodes for tag overlap ─────────────────────────────
    tagged_nodes = [
        s for s in survivors
        if s.get("type") in ("NUMERIC", "DEFINITION") and s.get("value_tags")
    ]

    if not tagged_nodes:
        # No tags at all — document was extracted before value_tags were added
        return {"has_tags": False, "match_score": None,
                "best_overlap": 0, "required_tags": sorted(required)}

    best_overlap = 0
    for node in tagged_nodes:
        tags = set(node["value_tags"])
        overlap = len(tags & required)
        if overlap > best_overlap:
            best_overlap = overlap

    match_score = round(min(best_overlap / max(len(required), 1), 1.0), 4)

    return {
        "has_tags":     True,
        "match_score":  match_score,
        "best_overlap": best_overlap,
        "required_tags": sorted(required),
    }


def _predict_answerability(topo: dict) -> dict:
    """Answerability scorer with explicit completeness bottleneck and escape-hatch gating."""
    import math

    def _get(d: dict | None, *keys, default=None):
        v = d
        for k in keys:
            if not isinstance(v, dict):
                return default
            v = v.get(k)
        return v if v is not None else default

    def _clean01(v):
        if v is None:
            return None
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return None

    def _clean_range(v, lo: float, hi: float):
        if v is None:
            return None
        try:
            x = float(v)
        except Exception:
            return None
        return max(lo, min(hi, x))

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _cal_pos(x, center, scale):
        if x is None:
            return None
        return _sigmoid((x - center) / max(scale, 1e-6))

    def _cal_risk(x, center, scale):
        if x is None:
            return None
        return _sigmoid((x - center) / max(scale, 1e-6))

    def _wavg(items: list[tuple[float | None, float]], fallback: float) -> float:
        used = [(x, w) for (x, w) in items if x is not None]
        if not used:
            return fallback
        den = sum(w for _, w in used)
        num = sum(float(x) * w for x, w in used)
        return num / max(den, 1e-9)

    # ---- Inputs ----
    slot_type = (_get(topo, "answer_path", "slot_type", default="GENERAL") or "GENERAL").upper()
    question_text = str(topo.get("question") or topo.get("q") or "").strip()
    slots_hint = [{"slot": slot_type}] if slot_type else []

    intent = _get(topo, "question_intent", default={}) or {}
    if not intent:
        try:
            intent = _question_intent(question_text, slots_hint)
        except Exception:
            intent = {}

    exact_intent = _clean01(intent.get("exact_intent")) or 0.0
    simple_extract_intent = _clean01(intent.get("simple_extract_intent")) or 0.0
    proof_type_initial = str(intent.get("proof_type") or "mixed")
    answer_form = str(intent.get("answer_form") or "mixed")
    lane = str(intent.get("lane") or "mixed")
    expects_direct_rule_clause = _clean01(intent.get("expects_direct_rule_clause")) or 0.0
    speculative_intent = _clean01(intent.get("speculative_intent")) or 0.0
    outside_document_intent = _clean01(intent.get("outside_document_intent")) or 0.0
    missing_artifact_intent = _clean01(intent.get("missing_artifact_intent")) or 0.0
    outside_doc_needed = _clean01(intent.get("outside_doc_needed")) or 0.0
    asks_for_external_contents = _clean01(intent.get("asks_for_external_contents")) or 0.0
    external_doc_type = str(intent.get("external_doc_type") or "none").strip().lower()
    answer_source_type = str(intent.get("answer_source_type") or "mixed").strip().lower()
    source_sufficiency_required = str(intent.get("source_sufficiency_required") or "mixed").strip().lower()
    missing_if_not_present = bool(intent.get("missing_if_not_present", False))
    customer_instance_dependency = _clean01(intent.get("customer_instance_dependency")) or 0.0
    external_artifact_dependency = _clean01(intent.get("external_artifact_dependency")) or 0.0
    runtime_dependency = _clean01(intent.get("runtime_dependency")) or 0.0
    customer_specific_dependency_prior = _clean01(intent.get("customer_specific_dependency_prior"))
    if customer_specific_dependency_prior is None:
        customer_specific_dependency_prior = _clean01(intent.get("instance_missing_field")) or 0.0
    external_doc_dependency_prior = _clean01(intent.get("external_doc_dependency_prior"))
    if external_doc_dependency_prior is None:
        external_doc_dependency_prior = _clean01(intent.get("external_reference_intent")) or 0.0
    instance_missing_field = _clean01(intent.get("instance_missing_field")) or 0.0
    external_reference_intent = _clean01(intent.get("external_reference_intent")) or 0.0
    ql = question_text.lower()
    asks_name_like_early = bool(intent.get("asks_name_like"))
    asks_instance_like_early = bool(intent.get("asks_instance_like"))
    asks_value_like_early = bool(intent.get("asks_value_like"))
    asks_percent_like_early = bool(intent.get("asks_percent_like"))
    asks_date_like_early = bool(intent.get("asks_date_like"))
    asks_list_like_early = bool(intent.get("asks_list_like"))
    asks_yesno_like_early = bool(intent.get("asks_yesno_like"))
    asks_external_reference_like_early = bool(intent.get("asks_external_reference_like"))
    exact_form_like_early = bool(
        asks_value_like_early
        or asks_percent_like_early
        or asks_date_like_early
        or asks_name_like_early
        or asks_list_like_early
        or answer_form in {"date", "number", "percent", "name", "list"}
        or proof_type_initial in {"exact_value", "enumeration", "customer_instance_field"}
    )
    customer_instance_markers = any(
        tok in ql for tok in (
            "this customer", "for this customer", "specific customer", "this customer's",
            "current monthly", "current invoice", "authorized purchaser by name",
            "customer legal name", "customer organization",
        )
    )
    external_artifact_markers = any(
        tok in ql for tok in (
            "pricing page", "dpa", "security measures", "service-specific terms",
            "healthcare addendum", "supported countries", "exhibit", "appendix",
            "annex", "schedule", "audit reports",
        )
    )
    external_contents_shape = bool(
        not asks_yesno_like_early
        and (
            "contents" in ql
            or "listed in" in ql
            or "which specific" in ql
            or "what are the exact" in ql
            or "exact terms" in ql
            or "exact clauses" in ql
            or "table" in ql
            or "schedule values" in ql
        )
    )
    audit_frequency_rule_like = bool(
        "audit reports" in ql
        and (
            "how often" in ql
            or "request" in ql
            or "frequency" in ql
            or "once per year" in ql
        )
    )
    # "where is the DPA located/found?" — asks for a URL or pointer stated inline in
    # the document, not for the contents of the external document itself.
    url_location_rule_like = bool(
        (
            any(tok in ql for tok in ("where is ", "where can ", "where are ", "where do "))
            and any(tok in ql for tok in ("located", "found", "available", "accessed",
                                          "access it", "find it", "find the", "access the"))
            or any(tok in ql for tok in ("what url ", "which url ", "what is the url",
                                          "url is the primary", "what is the primary url"))
        )
        and not any(tok in ql for tok in ("what does", "what are the terms",
                                          "what are the obligations",
                                          "what does it say", "describe the contents"))
    )
    # "What is MLX's principal address?" / "What is HCI's principal place of business?" —
    # Party addresses are stated once in the contract preamble. The graph scores them low
    # because preamble boilerplate is sparse; route deterministically to bypass topo gate.
    address_question_rule_like = bool(
        any(tok in ql for tok in (
            "principal address", "principal place of business", "place of business address",
            "registered address", "headquarters address", "principal office address",
            "mailing address", "principal place of",
        ))
    )
    # "what can X do" / "what right/remedy does X have" — CONSEQUENCE questions about
    # contractual rights. Even if the subject (e.g. "security measures") appears in
    # external_artifact_markers, these questions ask about contract terms, not artifact
    # contents. Route them away from the external_artifact deterministic path.
    consequence_right_like = bool(
        not external_contents_shape
        and (
            "what can " in ql
            or "what right" in ql
            or "what remedy" in ql
            or "what recourse" in ql
            or "what option" in ql
        )
    )
    # Named-exec runtime questions: "who is the current CEO/president" — asks for a
    # live org fact that is never in a contract document.
    named_exec_rule_like = bool(
        any(tok in ql for tok in (
            "current ceo", "current president", "current chairman", "current founder",
            "current cto", "current cfo", "current coo", "current vp",
            "who is the ceo", "who is the president", "who is the chairman",
            "who is the founder", "who is the cto", "who is the cfo",
        ))
    )
    # Asking what an external *policy* document says — "what does X's Privacy/AI Policy
    # say about Y?" — the policy is referenced in the contract but its contents are external.
    policy_contents_rule_like = bool(
        "policy" in ql
        and any(tok in ql for tok in ("say about", "say regarding", "say on ", "state about",
                                       "what does the", "what do the", "describe the",
                                       "what does it say", "what does cyberark"))
        and not asks_yesno_like_early
    )
    # Speculative conditional questions: "if it fails next month" / "what will X be if Y?"
    # Also: amendment-history instance facts — whether the agreement has been amended is a
    # runtime state not present in the static document.
    speculative_conditional_rule_like = bool(
        any(tok in ql for tok in ("next month", "next week", "if it fails", "if they fail",
                                   "if it doesn't", "if he fails", "if she fails",
                                   "been amended since", "has this agreement been amended",
                                   "have any amendments been", "any amendment been",
                                   "been amended or modified"))
        or (
            ql.startswith("what will ")
            and any(tok in ql for tok in ("next month", "next week", "if it fails",
                                           "if it doesn't", "if they fail"))
        )
    )
    # "generated since the Effective Date" / "since inception" / "to date" — runtime count facts.
    runtime_count_markers = any(tok in ql for tok in (
        "since the effective date", "since inception", "since signing", "since execution",
        "since the launch date", "since launch", "since the launch",
        "generated since", "accrued since", "earned to date", "earned since",
        "to date", "so far", "thus far", "currently have", "currently enrolled",
    ))
    runtime_markers = any(tok in ql for tok in ("current ", "currently", "as of now", "today")) \
        or runtime_count_markers
    deterministic_source_route = "none"
    if runtime_count_markers and exact_form_like_early:
        # "how many X have been Y since the Effective Date" — always a runtime count.
        deterministic_source_route = "runtime_fact"
        answer_source_type = "runtime_fact"
        source_sufficiency_required = "runtime_lookup_required"
        missing_if_not_present = True
        runtime_dependency = max(runtime_dependency, 0.82)
    elif runtime_markers and customer_instance_markers and exact_form_like_early:
        deterministic_source_route = "runtime_fact"
        answer_source_type = "runtime_fact"
        source_sufficiency_required = "runtime_lookup_required"
        missing_if_not_present = True
        runtime_dependency = max(runtime_dependency, 0.82)
    elif external_artifact_markers and exact_form_like_early and (
        external_contents_shape
        or asks_for_external_contents >= 0.55
        or missing_artifact_intent >= 0.55
        or asks_external_reference_like_early
    ) and not audit_frequency_rule_like and not consequence_right_like and not url_location_rule_like \
      and not named_exec_rule_like and not speculative_conditional_rule_like \
      and not address_question_rule_like:
        deterministic_source_route = "external_artifact"
        answer_source_type = "external_artifact"
        source_sufficiency_required = "artifact_contents_required"
        missing_if_not_present = True
        external_artifact_dependency = max(external_artifact_dependency, 0.78)
        external_doc_dependency_prior = max(external_doc_dependency_prior, 0.72)
        external_reference_intent = max(external_reference_intent, 0.68)
        asks_for_external_contents = max(asks_for_external_contents, 0.72)
        missing_artifact_intent = max(missing_artifact_intent, 0.70)
    elif audit_frequency_rule_like:
        deterministic_source_route = "audit_base_clause"
        answer_source_type = "base_clause"
        source_sufficiency_required = "clause_quote_enough"
        missing_if_not_present = False
        asks_for_external_contents = min(asks_for_external_contents, 0.20)
        missing_artifact_intent = min(missing_artifact_intent, 0.20)
        external_reference_intent = min(external_reference_intent, 0.25)
        external_doc_dependency_prior = min(external_doc_dependency_prior, 0.25)
    elif url_location_rule_like:
        # Asking where a document/policy/URL is located — the answer is the inline URL
        # or pointer stated in this contract, not the contents of the external doc.
        deterministic_source_route = "url_base_clause"
        answer_source_type = "base_clause"
        source_sufficiency_required = "clause_quote_enough"
        missing_if_not_present = False
        asks_for_external_contents = min(asks_for_external_contents, 0.20)
        missing_artifact_intent = min(missing_artifact_intent, 0.20)
        external_reference_intent = min(external_reference_intent, 0.25)
        external_doc_dependency_prior = min(external_doc_dependency_prior, 0.25)
    elif address_question_rule_like:
        # Party address / place-of-business questions — the answer is stated once in the
        # preamble. The graph rarely connects preamble boilerplate strongly so topology
        # scores low; route deterministically to base_clause so the pipeline override fires.
        deterministic_source_route = "address_base_clause"
        answer_source_type = "base_clause"
        source_sufficiency_required = "clause_quote_enough"
        missing_if_not_present = False
        asks_for_external_contents = min(asks_for_external_contents, 0.20)
        missing_artifact_intent = min(missing_artifact_intent, 0.20)
        external_reference_intent = min(external_reference_intent, 0.25)
        external_doc_dependency_prior = min(external_doc_dependency_prior, 0.25)
    elif named_exec_rule_like:
        # Named-exec: asking for a live organizational fact (CEO, president) — never in document.
        deterministic_source_route = "runtime_fact"
        answer_source_type = "runtime_fact"
        source_sufficiency_required = "runtime_lookup_required"
        missing_if_not_present = True
        runtime_dependency = max(runtime_dependency, 0.88)
        outside_document_intent = max(outside_document_intent, 0.80)
    elif policy_contents_rule_like:
        # Asking what an external policy says — the policy is referenced but external.
        deterministic_source_route = "external_artifact"
        answer_source_type = "external_artifact"
        source_sufficiency_required = "artifact_contents_required"
        missing_if_not_present = True
        external_artifact_dependency = max(external_artifact_dependency, 0.78)
        external_doc_dependency_prior = max(external_doc_dependency_prior, 0.72)
        asks_for_external_contents = max(asks_for_external_contents, 0.72)
        missing_artifact_intent = max(missing_artifact_intent, 0.70)
    elif speculative_conditional_rule_like:
        # Speculative/conditional future questions — not answerable from static document.
        deterministic_source_route = "speculative"
        answer_source_type = "runtime_fact"
        source_sufficiency_required = "runtime_lookup_required"
        missing_if_not_present = True
        runtime_dependency = max(runtime_dependency, 0.80)
        outside_document_intent = max(outside_document_intent, 0.75)
    elif customer_instance_markers and exact_form_like_early:
        deterministic_source_route = "customer_instance"
        answer_source_type = "customer_instance"
        if source_sufficiency_required == "mixed":
            source_sufficiency_required = "explicit_value_required"
        missing_if_not_present = True
        customer_instance_dependency = max(customer_instance_dependency, 0.78)
        customer_specific_dependency_prior = max(customer_specific_dependency_prior, 0.72)
        instance_missing_field = max(instance_missing_field, 0.70)
    llm_ev = topo.get("llm_evidence_sufficiency", {}) or {}
    if llm_ev:
        answer_source_type = str(llm_ev.get("answer_source_type") or answer_source_type).strip().lower()
        source_sufficiency_required = str(llm_ev.get("source_sufficiency_required") or source_sufficiency_required).strip().lower()
    if deterministic_source_route == "runtime_fact":
        answer_source_type = "runtime_fact"
        source_sufficiency_required = "runtime_lookup_required"
    elif deterministic_source_route == "external_artifact":
        answer_source_type = "external_artifact"
        source_sufficiency_required = "artifact_contents_required"
    elif deterministic_source_route == "audit_base_clause":
        answer_source_type = "base_clause"
        source_sufficiency_required = "clause_quote_enough"
    elif deterministic_source_route == "customer_instance":
        answer_source_type = "customer_instance"
        if source_sufficiency_required == "mixed":
            source_sufficiency_required = "explicit_value_required"
    elif deterministic_source_route == "url_base_clause":
        answer_source_type = "base_clause"
        source_sufficiency_required = "clause_quote_enough"
    elif deterministic_source_route == "address_base_clause":
        answer_source_type = "base_clause"
        source_sufficiency_required = "clause_quote_enough"
    elif deterministic_source_route == "speculative":
        answer_source_type = "runtime_fact"
        source_sufficiency_required = "runtime_lookup_required"
    llm_inline_answer_present = _clean01(llm_ev.get("inline_answer_present")) or 0.0
    llm_external_reference_only = _clean01(llm_ev.get("external_reference_only")) or 0.0
    llm_customer_instance_missing = _clean01(llm_ev.get("customer_instance_missing")) or 0.0
    llm_runtime_missing = _clean01(llm_ev.get("runtime_missing")) or 0.0
    # Yes/no questions ask whether a rule applies — never whether an external
    # artifact's contents are present. Prevent the hardening cascade from treating
    # them as artifact-content requests regardless of what the LLM intent said.
    if (
        asks_yesno_like_early
        and source_sufficiency_required == "artifact_contents_required"
        and asks_for_external_contents < 0.40
    ):
        source_sufficiency_required = "clause_quote_enough"
        missing_if_not_present = False
    # If the question names no external artifact AND the document graph shows zero
    # artifact dependency, a LLM-only "external_artifact" classification is almost
    # certainly a misfire (e.g. LLM sees a nearby Exhibit A reference and assumes
    # all nearby values come from it). Override back to base_clause before hardening.
    if (
        answer_source_type == "external_artifact"
        and deterministic_source_route == "none"
        and not external_artifact_markers
        and external_artifact_dependency == 0.0
    ):
        answer_source_type = "base_clause"
        source_sufficiency_required = "explicit_value_required"
        missing_if_not_present = False
        asks_for_external_contents = min(asks_for_external_contents, 0.20)
        missing_artifact_intent = min(missing_artifact_intent, 0.20)
    # Source/sufficiency-aware prior hardening.
    if answer_source_type == "customer_instance":
        customer_specific_dependency_prior = max(
            customer_specific_dependency_prior,
            0.52 + 0.28 * max(customer_instance_dependency, llm_customer_instance_missing),
        )
        instance_missing_field = max(instance_missing_field, customer_specific_dependency_prior)
        if source_sufficiency_required in {"explicit_value_required", "mixed"}:
            missing_if_not_present = True
    if answer_source_type == "external_artifact":
        external_doc_dependency_prior = max(
            external_doc_dependency_prior,
            0.58 + 0.24 * max(external_artifact_dependency, llm_external_reference_only),
        )
        missing_artifact_intent = max(missing_artifact_intent, 0.52)
    if source_sufficiency_required == "artifact_contents_required":
        external_doc_dependency_prior = max(
            external_doc_dependency_prior,
            0.68 + 0.22 * max(external_artifact_dependency, llm_external_reference_only),
        )
        missing_artifact_intent = max(missing_artifact_intent, 0.66)
        asks_for_external_contents = max(asks_for_external_contents, 0.72)
    if answer_source_type == "runtime_fact" or source_sufficiency_required == "runtime_lookup_required":
        runtime_dependency = max(runtime_dependency, 0.72)
        outside_doc_needed = max(outside_doc_needed, 0.68)
        outside_document_intent = max(outside_document_intent, 0.48)

    vote_stats = topo.get("consensus_vote_stats", {}) or {}
    mean_vote_frac = _clean01(vote_stats.get("mean_vote_frac"))
    p75_vote_frac = _clean01(vote_stats.get("p75_vote_frac"))
    high_vote_frac = _clean01(vote_stats.get("high_vote_frac"))
    max_vote_frac = _clean01(vote_stats.get("max_vote_frac"))
    n_vote_ge2 = int(vote_stats.get("n_vote_ge2") or 0)
    n_vote_ge3 = int(vote_stats.get("n_vote_ge3") or 0)

    tpc_details = _get(topo, "typed_path_coverage", "details", default=[]) or []
    tpc_total = int(_get(topo, "typed_path_coverage", "total", default=0) or 0)
    tpc_matched = int(_get(topo, "typed_path_coverage", "matched", default=0) or 0)

    slot = _clean01(_get(topo, "typed_path_coverage", "score"))
    path = _clean01(_get(topo, "answer_path", "path_bottleneck"))
    h0 = _clean01(_get(topo, "persistent_homology", "h0_mean_lifetime"))
    trig = _clean01(_get(topo, "chain", "trigger_edge_weight"))
    sheaf = _clean01(_get(topo, "sheaf", "consistent_frac"))
    w_sheaf = _clean01(_get(topo, "weighted", "weighted_consistent_frac"))
    w_ricci = _clean_range(_get(topo, "weighted", "weighted_mean"), -1.0, 1.0)
    bridge = _clean_range(_get(topo, "weighted", "anchor_bridge_score"), -1.0, 1.0)
    exact = _clean01(_get(topo, "slot_mismatch_depth", "exact_frac"))
    adjacent = _clean01(_get(topo, "slot_mismatch_depth", "adjacent_frac"))
    seq = _clean01(_get(topo, "required_sequence", "sequence_frac"))

    vtm = topo.get("value_tag_match", {}) or {}
    tag = _clean01(vtm.get("match_score")) if vtm.get("has_tags", False) else None
    n_targets = int(_get(topo, "answer_path", "n_targets", default=0) or 0)
    n_well = int(_get(topo, "answer_path", "n_well_connected", default=0) or 0)
    well_conn = _clean01((n_well / n_targets) if n_targets > 0 else None)
    n_nodes_total = int(_get(topo, "n_nodes", default=0) or 0)
    target_frac = _clean01((n_targets / n_nodes_total) if n_nodes_total > 0 else None)
    path_exists = bool(_get(topo, "answer_path", "path_exists", default=False))
    path_bottleneck = path

    chain = _clean01(_get(topo, "chain", "chain_strength"))
    type_entropy = _clean_range(_get(topo, "type_metrics", "type_entropy"), 0.0, 4.0)
    ph = _clean01(_get(topo, "placeholder_values", "placeholder_frac"))
    distant = _clean01(_get(topo, "slot_mismatch_depth", "distant_frac"))
    party_mismatch = _clean01(_get(topo, "direction", "party_mismatch_frac"))
    role_inv = _clean01(_get(topo, "direction", "edge_role_inversion"))

    kw_flagged = _get(topo, "keyword_trap", "flagged")
    if kw_flagged is True:
        kw_risk = _clean01(_get(topo, "keyword_trap", "confidence")) or 0.60
    elif kw_flagged is False:
        kw_risk = 0.0
    else:
        kw_risk = None

    # ---- Topic evidence T ----
    topic_terms = [
        (_cal_pos(path, 0.260, 0.100), 0.30),
        (_cal_pos(h0, 0.370, 0.060), 0.15),
        (_cal_pos(trig, 0.420, 0.050), 0.12),
        (_cal_pos(sheaf, 0.700, 0.100), 0.11),
        (_cal_pos(w_sheaf, 0.720, 0.100), 0.08),
        (_cal_pos(w_ricci, -0.400, 0.120), 0.06),
        (_cal_pos(bridge, -0.500, 0.100), 0.05),
        (_cal_pos(seq, 0.450, 0.220), 0.05),
        (_cal_pos(well_conn, 0.500, 0.250), 0.08),
    ]
    T = _wavg(topic_terms, fallback=0.35)
    topic_used = sum(1 for x, _ in topic_terms if x is not None)

    # ---- Risk ----
    risk_terms = [
        (_cal_risk(chain, 0.880, 0.200), 0.25),
        (_cal_risk(type_entropy, 2.200, 0.200), 0.22),
        (ph, 0.38),
        (distant, 0.20),
        (party_mismatch, 0.20),
        (role_inv, 0.18),
        (kw_risk, 0.16),
    ]
    risk = _wavg(risk_terms, fallback=0.40)
    risk_used = sum(1 for x, _ in risk_terms if x is not None)

    # ---- Slot/answer completeness features ----
    req_slots = tpc_total if tpc_total > 0 else len(tpc_details)
    slot_fill_ratio = _clean01((tpc_matched / req_slots) if req_slots > 0 else slot)
    direct_count = sum(1 for d in tpc_details if d.get("direct"))
    direct_fill_ratio = _clean01((direct_count / req_slots) if req_slots > 0 else None)

    slot_specificity = {
        "GENERAL": 0.25,
        "ACTOR": 0.45,
        "PERMISSION": 0.55,
        "MEANING": 0.55,
        "VALUE": 0.90,
        "REQUIREMENT": 0.80,
        "CONSEQUENCE": 0.80,
    }.get(slot_type, 0.50)

    S = max(slot_specificity, exact_intent)

    # Partial-fill fallback: when typed path counters are missing/noisy, preserve a
    # weak structured fill signal from typed_path score and direct clause support.
    if slot_fill_ratio is None:
        slot_fill_ratio = _clean01(slot)
    sf = slot_fill_ratio if slot_fill_ratio is not None else (slot if slot is not None else 0.0)
    ex = exact if exact is not None else (slot if slot is not None else 0.0)
    df = direct_fill_ratio if direct_fill_ratio is not None else sf
    dist_good = (1.0 - distant) if distant is not None else 0.5
    wc = well_conn if well_conn is not None else 0.5

    A_raw = max(0.0, min(1.0, 0.44 * sf + 0.30 * ex + 0.12 * df + 0.08 * wc + 0.06 * dist_good))

    # ---- Direct explicit support Q ----
    des = topo.get("direct_explicit_support", {}) or {}
    Q = _clean01(des.get("score"))
    if Q is None:
        # Fallback proxy for offline CSV evals lacking clause-level traces.
        path_proxy = path if path is not None else 0.5
        Q = max(0.0, min(1.0, 0.55 * ex + 0.25 * sf + 0.20 * path_proxy))
    Q_source = "direct_explicit_support" if _clean01(des.get("score")) is not None else "proxy"
    if slot_fill_ratio is None:
        slot_fill_ratio = max(0.0, min(1.0, 0.60 * Q + 0.40 * (slot if slot is not None else 0.0)))
        sf = slot_fill_ratio

    # ---- Proof routing override (diagnostic only; no decision impact) ----
    des_trace = des.get("trace", {}) if isinstance(des.get("trace"), dict) else {}
    des_explicit = _clean01(des_trace.get("explicitness_F"))
    if des_explicit is None:
        des_explicit = 0.0
    des_node_type = str(des_trace.get("node_type") or "").upper()
    des_text = str(des_trace.get("text") or "").lower()
    rule_language_hit = any(tok in des_text for tok in (" shall ", " must ", " may ", " may not", " shall not ", " required to ", " liable ", " liability ", " charge ", " cap "))
    asks_yesno_like = bool(intent.get("asks_yesno_like"))
    finance_rule_like = bool(intent.get("asks_value_like")) and any(tok in question_text.lower() for tok in ("fee", "charge", "cap", "liability", "finance", "interest", "overdue"))
    rule_shape = (
        asks_yesno_like
        or slot_type in {"PERMISSION", "REQUIREMENT", "CONSEQUENCE"}
        or finance_rule_like
    )
    direct_support_rule_like = (
        Q >= 0.68
        and des_explicit >= 0.80
        and (
            rule_language_hit
            or des_node_type in {"OBLIGATION", "RIGHT", "CONDITION"}
        )
    )
    low_spec_outside = (speculative_intent <= 0.20 and outside_document_intent <= 0.20)
    direct_rule_clause_override = bool(
        proof_type_initial == "mixed"
        and rule_shape
        and expects_direct_rule_clause >= 0.45
        and direct_support_rule_like
        and low_spec_outside
    )
    proof_type_routed = "direct_rule_clause" if direct_rule_clause_override else proof_type_initial

    # Effective strictness drops when direct support is strong for simple extractive asks.
    S_eff = max(0.0, min(1.0, S * (1.0 - 0.65 * Q * simple_extract_intent)))

    # Answer evidence A is completeness-bottlenecked and lightly coupled to topic relevance.
    # A_cap raised from 0.56+0.44*Q to 0.64+0.36*Q: same ceiling at Q=1.0 (both ≈1.0)
    # but a higher floor at low Q, reducing score sensitivity to intent-LLM Q variance.
    A_cap = 0.64 + 0.36 * Q
    if S_eff >= 0.62:
        A = min(A_raw, A_cap)
    else:
        A = min(1.0, 0.88 * A_raw + 0.12 * min(T, A_raw))

    # Agreement/disagreement D: metric disagreement + topic/answer mismatch.
    core = [p for p in (
        _cal_pos(slot, 0.530, 0.090),
        _cal_pos(path, 0.260, 0.100),
        _cal_pos(h0, 0.370, 0.060),
        _cal_pos(trig, 0.420, 0.050),
    ) if p is not None]
    core_coverage = len(core) / 4.0
    if len(core) >= 2:
        core_for_disp = sorted(core)[1:-1] if len(core) >= 4 else core
        mean_core = sum(core_for_disp) / len(core_for_disp)
        var_core = sum((p - mean_core) ** 2 for p in core_for_disp) / len(core_for_disp)
        D_core = min(1.0, math.sqrt(var_core) / 0.30)
        D_core *= (0.55 + 0.45 * core_coverage)
    else:
        D_core = 0.0
    D = max(0.0, min(1.0, 0.65 * D_core + 0.35 * abs(T - A)))

    # Additional structured penalties.
    P = speculative_intent
    O = outside_document_intent
    M = missing_artifact_intent
    # LLM source/sufficiency routing corrections.
    if answer_source_type == "customer_instance":
        instance_missing_field = max(instance_missing_field, 0.55 * customer_instance_dependency + 0.45 * llm_customer_instance_missing)
    if answer_source_type == "runtime_fact":
        runtime_gap = max(runtime_dependency, llm_runtime_missing)
        O = max(O, 0.50 * runtime_gap)
        P = max(P, 0.25 * runtime_gap)
    if answer_source_type == "external_artifact" or source_sufficiency_required == "artifact_contents_required":
        ext_gap = max(external_artifact_dependency, llm_external_reference_only)
        M = max(M, 0.65 * ext_gap + 0.35 * asks_for_external_contents)
        external_reference_intent = max(external_reference_intent, 0.50 * ext_gap + 0.25 * asks_for_external_contents)
    if missing_if_not_present and llm_inline_answer_present < 0.45:
        M = max(M, 0.55 + 0.25 * (1.0 - llm_inline_answer_present))
    definition_base_clause_relief = bool(
        proof_type_routed == "definition"
        and answer_source_type == "base_clause"
        and llm_inline_answer_present >= 0.80
        and A_raw >= 0.70
        and source_sufficiency_required in {"clause_quote_enough", "explicit_value_required", "mixed"}
    )
    if definition_base_clause_relief:
        M = min(M, 0.20)
        missing_artifact_intent = min(missing_artifact_intent, 0.25)
        asks_for_external_contents = min(asks_for_external_contents, 0.25)
    rule_reference_base_relief = bool(
        answer_source_type == "base_clause"
        and proof_type_routed in {"direct_rule_clause", "definition"}
        and llm_inline_answer_present >= 0.80
        and (
            asks_for_external_contents < 0.55
            or asks_yesno_like_early  # yes/no answers can't require artifact contents; trust base_clause routing
        )
        and (
            asks_yesno_like_early
            or slot_type in {"PERMISSION", "REQUIREMENT", "CONSEQUENCE"}
        )
    )
    if rule_reference_base_relief:
        M *= 0.65
        external_reference_intent *= 0.70
    # Yes/no questions answered from the base contract can never require an external
    # artifact — the answer is "yes" or "no" drawn from the contract's own terms.
    # Cap M regardless of llm_inline_answer_present to prevent misfire cascade.
    if asks_yesno_like_early and answer_source_type == "base_clause":
        M = min(M, 0.35)
        missing_artifact_intent = min(missing_artifact_intent, 0.40)
    # Lane-aware shaping: small local adjustments only.
    if lane == "dependency_risk":
        M = min(1.0, M + 0.10)
        O = min(1.0, O + 0.08)
    elif lane == "speculative_outside":
        P = min(1.0, P + 0.10)
        O = min(1.0, O + 0.10)
    ql = question_text.lower()

    # Micro-calibration priors (scorer-only, narrow patterns) to prevent
    # post-plumbing false positives on instance/external commercial asks.
    asks_name_like = bool(intent.get("asks_name_like"))
    asks_instance_like = bool(intent.get("asks_instance_like"))
    asks_value_like = bool(intent.get("asks_value_like"))
    asks_percent_like = bool(intent.get("asks_percent_like"))
    asks_report_like = bool(intent.get("asks_report_like"))
    asks_yesno_like_intent = bool(intent.get("asks_yesno_like"))
    asks_external_reference_like_intent = bool(intent.get("asks_external_reference_like"))
    primary_slot_intent = str(intent.get("primary_slot") or slot_type or "GENERAL").upper()
    customer_identity_like = bool(
        asks_name_like
        and asks_instance_like
        and any(tok in ql for tok in ("customer", "organization", "organisation", "signed", "signatory"))
    )
    external_fee_schedule_like = (
        asks_value_like
        and any(tok in ql for tok in ("dollar amount", "amount", "how much", "fee amount", "filing fee"))
        and any(tok in ql for tok in ("arbitration", "nam", "schedule", "table"))
    )
    reseller_commercial_term_like = (
        any(tok in ql for tok in ("reseller", "revenue-sharing", "revenue sharing"))
        or ("discount" in ql and "percentage" in ql)
    )
    type_counts = _get(topo, "type_metrics", "type_counts", default={}) or {}
    ref_count = float(type_counts.get("REFERENCE", 0.0) or 0.0)
    blank_count = float(type_counts.get("BLANK", 0.0) or 0.0)
    content_count = float(type_counts.get("OBLIGATION", 0.0) or 0.0) + float(type_counts.get("RIGHT", 0.0) or 0.0) + float(type_counts.get("DEFINITION", 0.0) or 0.0) + float(type_counts.get("NUMERIC", 0.0) or 0.0) + float(type_counts.get("CONDITION", 0.0) or 0.0)
    denom_tc = max(1.0, ref_count + blank_count + content_count)
    ref_frac = ref_count / denom_tc
    blank_frac = blank_count / denom_tc
    ph_norm = ph if ph is not None else 0.0
    extdep = topo.get("external_dependency_evidence", {}) or {}
    ext_ref_only_ratio = _clean01(extdep.get("external_ref_only_ratio")) or 0.0
    ext_inline_substance_ratio = _clean01(extdep.get("external_inline_substance_ratio")) or 0.0
    ext_doc_match_ratio = _clean01(extdep.get("question_doc_match_ratio")) or 0.0
    ext_related_ratio = _clean01(extdep.get("external_related_ratio")) or 0.0
    # Require corroboration: missing_artifact_intent alone (without asks_for_external_contents
    # above 0.20) can be a LLM misfire (e.g. it sees a document name but the question is about
    # contract rules, not artifact contents). Trust asks_for_external_contents as the primary signal.
    asks_external_contents = bool(
        asks_for_external_contents >= 0.55
        or (missing_artifact_intent >= 0.62 and asks_for_external_contents >= 0.20)
    )
    mentions_external_doc = bool(
        asks_external_reference_like_intent
        or external_doc_type != "none"
        or external_doc_dependency_prior >= 0.45
        or ext_related_ratio >= 0.20
    )

    # Retrieval-aware refinement of customer-instance dependency.
    customer_specific_dependency_refined = max(
        0.0,
        min(
            1.0,
            customer_specific_dependency_prior
            + 0.24 * max(blank_frac, ph_norm)
            + 0.14 * (1.0 - min(1.0, Q))
            + 0.12 * max(0.0, 0.70 - sf)
            + 0.08 * max(0.0, 0.65 - ex),
        ),
    )
    if customer_identity_like:
        customer_specific_dependency_refined = max(
            customer_specific_dependency_refined,
            min(1.0, customer_specific_dependency_refined + 0.28),
        )
    if Q >= 0.90 and sf >= 0.88 and ex >= 0.72 and path_exists and (path_bottleneck or 0.0) >= 0.38:
        customer_specific_dependency_refined = max(0.0, customer_specific_dependency_refined - 0.18)
    # Back-compat key used in existing logic/UI.
    instance_missing_field = customer_specific_dependency_refined

    # Retrieval-aware refinement of external-document dependency.
    external_doc_dependency_refined = max(
        0.0,
        min(
            1.0,
            external_doc_dependency_prior
            + 0.30 * ref_frac
            + 0.18 * M
            + 0.16 * ext_ref_only_ratio
            + 0.10 * max(0.0, ext_ref_only_ratio - ext_inline_substance_ratio)
            + 0.08 * ext_doc_match_ratio
            + 0.08 * asks_for_external_contents * max(0.0, ext_ref_only_ratio - 0.20)
            + 0.06 * outside_doc_needed
            + 0.12 * (1.0 - min(1.0, Q))
            + 0.10 * max(0.0, 0.68 - ex)
            + (0.08 if not bool(vtm.get("has_tags", False)) else 0.0),
        ),
    )
    # Policy-level distinction: "mentions external doc" vs "asks for its contents".
    # If direct base-clause support is strong and the question is not asking for
    # external artifact contents, cap external dependency pressure.
    base_clause_reference_safe = bool(
        mentions_external_doc
        and not asks_external_contents
        and O <= 0.15
        and P <= 0.15
        and not customer_identity_like
        and not external_fee_schedule_like
        and not reseller_commercial_term_like
        and Q >= 0.66
        and A_raw >= 0.62
        and sf >= 0.72
        and path_exists
        and (path_bottleneck or 0.0) >= 0.32
        and (
            primary_slot_intent in {"REQUIREMENT", "PERMISSION", "CONSEQUENCE", "MEANING"}
            or asks_yesno_like_intent
            or proof_type_routed in {"direct_rule_clause", "definition"}
        )
    )
    if base_clause_reference_safe:
        external_doc_dependency_refined = min(external_doc_dependency_refined, 0.42)
        M *= 0.70
    if external_fee_schedule_like:
        external_doc_dependency_refined = max(
            external_doc_dependency_refined,
            min(1.0, external_doc_dependency_refined + 0.30),
        )
    if Q >= 0.92 and sf >= 0.90 and ex >= 0.75 and ref_frac < 0.15:
        external_doc_dependency_refined = max(0.0, external_doc_dependency_refined - 0.20)
    # If the question itself doesn't mention or ask about any external document,
    # REFERENCE-type nodes in the survivor set are cross-reference context rather
    # than evidence the answer requires an external artifact. Cap the dependency
    # so their presence doesn't inflate Q2 on purely base-clause questions.
    # O and P are intentionally NOT used as guards here: they are LLM-derived
    # and fluctuate across runs. Using them as guards causes the cap to toggle
    # stochastically, making Q2 variance unpredictable. The structural guards
    # (mentions_external_doc, asks_external_contents) are more stable because
    # they aggregate multiple signals including lexical patterns.
    # Exception: external fee schedule queries genuinely require an external
    # document (the fee schedule), so the cap must not suppress them.
    if (
        not mentions_external_doc
        and not asks_external_contents
        and not external_fee_schedule_like
    ):
        external_doc_dependency_refined = min(external_doc_dependency_refined, 0.25)
    # Back-compat key used in existing logic/UI.
    external_reference_intent = external_doc_dependency_refined

    if reseller_commercial_term_like:
        O = max(O, min(1.0, O + 0.22))
        instance_missing_field = max(instance_missing_field, min(1.0, instance_missing_field + 0.18))

    direct_clause_sufficiency = bool(
        not customer_identity_like
        and not external_fee_schedule_like
        and not reseller_commercial_term_like
        and O <= 0.10
        and P <= 0.10
        and sf >= 0.95
        and Q >= 0.54
        and A >= 0.75
        and path_exists
        and (path_bottleneck or 0.0) >= 0.38
        and (
            primary_slot_intent in {"REQUIREMENT", "PERMISSION", "CONSEQUENCE"}
            or (asks_report_like and "how often" in ql)
            or (asks_value_like and "priced" in ql and "during" in ql and "term" in ql)
            or ("must be included" in ql and "order form" in ql)
            or (asks_yesno_like_intent and "workspace" in ql and "organizational id" in ql)
        )
    )
    direct_yesno_clause_relief = bool(
        not customer_identity_like
        and not external_fee_schedule_like
        and not reseller_commercial_term_like
        and asks_yesno_like_intent
        and Q >= 0.82
        and A >= 0.90
        and sf >= 0.95
        and O <= 0.10
        and P <= 0.10
        and (
            ("workspace" in ql and "organizational id" in ql)
            or ("same workspace" in ql)
            or ("same organizational id" in ql)
        )
    )
    base_clause_strong_support = bool(
        answer_source_type == "base_clause"
        and source_sufficiency_required in {"clause_quote_enough", "explicit_value_required", "mixed"}
        and llm_inline_answer_present >= 0.80
        and A >= 0.78
        and sf >= 0.90
        and (
            Q >= 0.50                    # relaxed from 0.60 — intent-LLM Q variance must
                                         # not gate a clear base-clause inline answer
            or (proof_type_routed in {"direct_rule_clause", "definition", "exact_value"} and Q >= 0.42)
        )
        and M <= 0.35
        and O <= 0.15
        and P <= 0.15
    )
    if direct_clause_sufficiency:
        instance_missing_field *= 0.55
        external_reference_intent *= (0.60 if (asks_report_like and "how often" in ql) else 0.75)
        if asks_report_like and "how often" in ql:
            M *= 0.45
    Q2 = max(P, O, instance_missing_field, external_reference_intent)
    Q2_extra = max(0.0, Q2 - max(P, O))
    instantiation_gap = max(
        0.0,
        min(
            1.0,
            (0.62 * instance_missing_field + 0.38 * external_reference_intent) * (1.0 - min(1.0, Q)),
        ),
    )
    instance_sensitive_question = bool(
        customer_identity_like
        or external_fee_schedule_like
        or reseller_commercial_term_like
    )

    X = max(0.0, 0.50 - ex)
    G = max(0.0, 0.62 - A_raw)
    B = max(0.0, (target_frac - 0.70) if target_frac is not None else 0.0)

    # ---- Main score ----
    sx_penalty_scale = max(0.60, 1.0 - 0.35 * simple_extract_intent * Q)
    # Narrow carveout: base-agreement rule despite addendum/reference language.
    des_type_match = _clean01(des_trace.get("type_match_T"))
    if des_type_match is None:
        des_type_match = 0.0
    des_score = _clean01(des.get("score")) or 0.0
    finance_rule_like = bool(intent.get("asks_value_like")) and any(
        tok in question_text.lower() for tok in ("cap", "liability", "charge", "fee", "interest", "finance")
    )
    carveout_q_floor = 0.62 if finance_rule_like else 0.70
    base_rule_despite_reference = bool(
        (
            slot_type in {"PERMISSION", "REQUIREMENT", "CONSEQUENCE"}
            or finance_rule_like
        )
        and expects_direct_rule_clause >= 0.40
        and Q >= carveout_q_floor
        and des_explicit >= 0.85
        and des_node_type in {"OBLIGATION", "RIGHT", "CONDITION"}
        and P == 0.0
        and O == 0.0
        and instance_missing_field <= 0.30
        and 0.35 <= external_reference_intent <= 0.78
        and (
            D >= 0.18
            or des_type_match >= 0.78
        )
        and (
            des_score >= (0.70 if finance_rule_like else 0.72)
            or des_type_match >= 0.82
        )
        and (
            (not finance_rule_like)
            or des_type_match >= 0.84
            or des_explicit >= 0.90
        )
    )
    # Continuous Q2 dampening: the double-sided Q2 penalty on both score and
    # threshold is proportionally reduced when direct evidence is strong. This
    # replaces the binary base_clause_strong_support + base_rule_despite_reference
    # checks and generalizes across the full evidence-strength continuum, removing
    # the structural problem where a single intent misfire causes both score to
    # collapse and threshold to rise simultaneously.
    _q2_inline_s = max(0.0, min(1.0, (llm_inline_answer_present - 0.50) * 2.5))
    _q2_direct_s = max(0.0, min(1.0, (Q - 0.50) * 2.5))
    _q2_clean_s  = max(0.0, 1.0 - 2.0 * max(M, O, P))
    # Q2 dampening is scoped to base-clause and undefined (mixed) questions.
    # For external_artifact, customer_instance, and runtime_fact questions, Q2_extra
    # represents genuine unanswerable signal and must not be dampened — even if Q
    # or llm_inline appear high from a misfire (reducing dampening here would let
    # LLM variance flip external-content questions to answerable).
    if answer_source_type not in {"base_clause", "mixed"}:
        q2_evidence_dampen = 0.0
    else:
        # Dampen ∈ [0, 1]: 0 = full penalty, 1 = maximum reduction
        q2_evidence_dampen = min(1.0, _q2_inline_s * _q2_direct_s * _q2_clean_s * 2.0)
        # Floor: confirmed base-clause answers always get at least partial dampening.
        # Without this, low Q from intent-LLM variance can produce near-zero dampening
        # even when sf/A/inline are all high.
        if base_clause_strong_support and llm_inline_answer_present >= 0.85:
            q2_evidence_dampen = max(q2_evidence_dampen, 0.70)   # coef ≤ 1.44
    q2_score_penalty_coef = 2.00 - 0.80 * q2_evidence_dampen  # range [1.20, 2.00]
    q2_threshold_scale    = 1.00 - 0.50 * q2_evidence_dampen  # range [0.50, 1.00]
    q2_local_scale = 1.0
    # base_rule_despite_reference: stronger dampening for direct rule-clause support
    if base_rule_despite_reference:
        q2_local_scale     = min(q2_local_scale, 0.30)
        q2_threshold_scale = min(q2_threshold_scale, 0.65)
    sxx_soften = 1.0
    # Soften the graph-exactness penalty for simple name/date/number extractions
    # when the QA LLM confirmed the answer is present inline. Low exact_frac on
    # these questions often reflects slot-match granularity, not a real gap.
    if (
        proof_type_routed == "exact_value"
        and answer_form in {"name", "date", "number"}
        and llm_inline_answer_present >= 0.90
        and M == 0.0
        and O == 0.0
        and P == 0.0
    ):
        sxx_soften = 0.60
    safe_direct_support_relief = bool(
        Q >= 0.62
        and A >= 0.78
        and sf >= 0.95
        and path_exists
        and (path_bottleneck or 0.0) >= 0.38
        and O <= 0.10
        and P <= 0.10
        and des_explicit >= 0.78
    )

    z = (
        2.65 * (A - 0.52)
        + 1.05 * (min(T, A) - 0.50)
        + 0.70 * max(0.0, Q - 0.65)            # direct support bonus: Q only enters
                                                # z through A's cap otherwise; strong
                                                # direct evidence deserves direct credit
        - 1.75 * (risk - 0.45)
        - 0.65 * D
        - 1.75 * G
        - 1.35 * sxx_soften * sx_penalty_scale * S_eff * X
        - 1.20 * B
        - 1.20 * M
        - 2.10 * P
        - 2.30 * O
        - q2_score_penalty_coef * q2_local_scale * Q2_extra
        - (0.60 * instantiation_gap if instance_sensitive_question else 0.0)
        - 0.10
    )

    if not path_exists:
        z -= 0.30
    elif path_bottleneck is None or path_bottleneck < 0.30:
        z -= 0.15
    if n_targets == 0:
        z -= 0.20
    if direct_clause_sufficiency:
        z += 0.16
    if direct_yesno_clause_relief:
        z += 0.14
    if safe_direct_support_relief:
        z += 0.10
    # Global safeguard: asking for external artifact contents requires stronger proof.
    asks_external_contents_guard = bool(
        asks_external_contents
        and mentions_external_doc
        and ext_ref_only_ratio >= 0.28
    )
    if asks_external_contents_guard and Q < 0.90:
        z -= 0.10

    score = max(0.0, min(1.0, _sigmoid(z)))

    # ---- Dynamic threshold ----
    evidence_coverage = topic_used / len(topic_terms)
    risk_coverage = risk_used / len(risk_terms)

    slot_threshold_offset = {
        "GENERAL": 0.03,
        "VALUE": 0.06,
        "REQUIREMENT": 0.05,
        "CONSEQUENCE": 0.05,
        "PERMISSION": 0.04,
        "ACTOR": 0.04,
        "MEANING": 0.04,
    }.get(slot_type, 0.04)

    threshold = (
        0.43
        + 0.08 * (1.0 - core_coverage)
        + 0.07 * (1.0 - evidence_coverage)
        + 0.07 * D
        + 0.06 * max(0.0, risk - 0.56)
        + slot_threshold_offset
        + 0.06 * S_eff
        + 0.11 * G
        + 0.09 * S_eff * X
        + 0.08 * B
        + 0.09 * M
        + 0.08 * P
        + 0.10 * O
        + 0.12 * q2_threshold_scale * Q2_extra
        + (0.06 * instantiation_gap if instance_sensitive_question else 0.0)
    )
    if lane == "direct_rule" and Q >= 0.70 and M <= 0.20 and O <= 0.20:
        threshold -= 0.015
    elif lane == "exact_value":
        threshold += 0.012

    # Targeted FN relief: when support is strong and there is no missing-artifact
    # or outside-doc pressure, avoid over-penalizing high disagreement cases.
    thr_rebate_disagreement = (
        0.030
        * max(0.0, D - 0.45)
        * (0.70 + 0.30 * Q)
        * (1.0 - M)
        * (1.0 - O)
    )
    # Small support rebate for clause-supported cases.
    thr_rebate_support = (
        0.012
        * max(0.0, Q - 0.78)
        * max(0.0, A - 0.70)
        * (1.0 - M)
        * (1.0 - O)
    )
    # Clean extractive rebates (small, bounded):
    # recover tiny misses without opening speculative/outside-document cases.
    thr_rebate_clean_extract = 0.0
    thr_rebate_nearmiss = 0.0
    thr_rebate_safe_recover = 0.0
    thr_rebate_rule_nearmiss = 0.0
    thr_rebate_base_clause = 0.0
    clean_extractive = (simple_extract_intent >= 0.20) or (slot_type in {"VALUE", "REQUIREMENT", "CONSEQUENCE"})
    q_safe = (Q <= 0.05) or (Q >= 0.75)
    if (
        clean_extractive
        and q_safe
        and M <= 0.05
        and O <= 0.05
        and P <= 0.05
        and D >= 0.32
        and D <= 0.40
        and A >= 0.70
    ):
        thr_rebate_clean_extract = 0.010 * min(1.0, ((D - 0.32) / 0.25) + 0.35)
        margin_preview = threshold - score
        if 0.0 < margin_preview <= 0.03:
            thr_rebate_nearmiss = min(0.015, 0.85 * margin_preview)

    # Conservative FN recovery: only when direct support is strong and
    # missing/external/speculative pressure is low.
    if (
        Q >= 0.80
        and sf >= 0.75
        and ex >= 0.55
        and M <= 0.20
        and instance_missing_field <= 0.35
        and external_reference_intent <= 0.35
        and P <= 0.30
        and O <= 0.30
    ):
        thr_rebate_safe_recover = min(0.03, 0.02 + 0.01 * max(0.0, Q - 0.90))

    # Lane 1: safe recovery for direct rule/extractive near-misses.
    rule_extract_shape = (
        slot_type in {"PERMISSION", "REQUIREMENT", "CONSEQUENCE"}
        or finance_rule_like
        or (slot_type == "VALUE" and simple_extract_intent >= 0.24)
    )
    lowrisk_nearmiss = (
        rule_extract_shape
        and Q >= 0.72
        and A >= 0.62
        and D >= 0.14
        and P <= 0.20
        and O <= 0.20
        and M <= 0.20
        and instance_missing_field <= 0.35
        and external_reference_intent <= 0.55
    )
    if lowrisk_nearmiss:
        margin_preview = threshold - score
        base_rebate = 0.008 + 0.010 * max(0.0, min(1.0, (Q - 0.72) / 0.20))
        near_rebate = 0.0
        if 0.0 < margin_preview <= 0.10:
            near_rebate = min(0.015, 0.60 * margin_preview)
        thr_rebate_rule_nearmiss = min(0.025, base_rebate + near_rebate)
    if (
        answer_source_type == "base_clause"
        and deterministic_source_route in {"none", "audit_base_clause"}
        and not customer_instance_markers
        and llm_inline_answer_present >= 0.85
        and A >= 0.70
        and sf >= 0.90
        and Q >= 0.52
        and M <= 0.30
        and O <= 0.10
        and P <= 0.10
        and proof_type_routed in {"direct_rule_clause", "definition", "exact_value"}
    ):
        thr_rebate_base_clause = 0.018 if proof_type_routed != "exact_value" else 0.015
        if proof_type_routed == "exact_value" and answer_form in {"name", "date", "number", "percent"}:
            thr_rebate_base_clause = min(0.022, thr_rebate_base_clause + 0.004)



    threshold -= (
        thr_rebate_disagreement
        + thr_rebate_support
        + thr_rebate_clean_extract
        + thr_rebate_nearmiss
        + thr_rebate_safe_recover
        + thr_rebate_rule_nearmiss
        + thr_rebate_base_clause
    )

    if not path_exists:
        threshold += 0.04
    elif path_bottleneck is None or path_bottleneck < 0.30:
        threshold += 0.03
    if n_targets == 0:
        threshold += 0.03
    if direct_clause_sufficiency:
        threshold -= 0.045
    if direct_yesno_clause_relief:
        threshold -= 0.050
    if safe_direct_support_relief:
        threshold -= 0.02
    if base_clause_strong_support:
        threshold -= 0.018
    if asks_external_contents_guard:
        threshold += 0.03
    # Slot-heavy asks need stronger variant-consensus.
    exact_slot_heavy = (
        proof_type_routed in {"exact_value", "customer_instance_field", "enumeration"}
        or slot_type in {"VALUE", "MEANING"}
    )
    weak_slot_channels = (sf < 0.45) or (ex < 0.45)
    high_quality_exact_support = bool(
        sf >= 0.90
        and ex >= 0.58
        and Q >= 0.70
        and path_exists
        and (path_bottleneck or 0.0) >= 0.35
    )
    # Safer consistency rule: at least 2 multi-variant-supported nodes.
    weak_variant_consensus = bool(n_vote_ge2 < 2 and not high_quality_exact_support)
    slot_variant_reject = bool(
        exact_slot_heavy
        and n_vote_ge2 < 2
        and Q < 0.70
        and ex < 0.45
        and not high_quality_exact_support
    )
    if slot_variant_reject:
        threshold += 0.035

    if evidence_coverage > 0.92 and risk < 0.35 and A > 0.78 and path_exists and (path_bottleneck or 0.0) >= 0.35:
        threshold -= 0.02

    threshold = max(0.40, min(0.86, threshold))

    # ---- Hard gates with direct-support escape hatch ----
    hard_reasons: list[str] = []

    # Escape hatch for straightforward extractive questions with direct clause support.
    escape_hatch = (
        Q >= 0.72
        and simple_extract_intent >= 0.45
        and A_raw >= 0.58
        and sf >= 0.70
        and path_exists
        and (path_bottleneck or 0.0) >= 0.30
        and O < 0.35
        and P < 0.45
    )

    # Strong contrary evidence can still block even with escape hatch.
    strong_contrary = (
        O >= 0.75
        or (P >= 0.82 and Q < 0.78)
        or (risk > 0.82 and D > 0.70)
    )

    if O >= 0.60:
        hard_reasons.append("outside_document_intent")
    if P >= 0.78 and Q < 0.78:
        hard_reasons.append("speculative_intent")
    if (
        Q2 >= 0.78
        and D < 0.90
        and Q < 0.88
        and not direct_clause_sufficiency
        and not safe_direct_support_relief
        and not base_clause_reference_safe
        and not (
            escape_hatch
            and Q >= 0.88
            and sf >= 0.82
            and ex >= 0.62
            and instance_missing_field < 0.65
            and external_reference_intent < 0.65
        )
    ):
        hard_reasons.append("q2_missing_or_external")
    if (
        asks_external_contents_guard
        and external_reference_intent >= 0.72
        and Q < 0.90
        and not escape_hatch
    ):
        hard_reasons.append("missing_artifact_contents")
    if (
        answer_source_type == "customer_instance"
        and source_sufficiency_required in {"explicit_value_required", "mixed"}
        and llm_inline_answer_present < 0.45
        and Q < 0.90
        and not escape_hatch
    ):
        hard_reasons.append("customer_instance_value_missing")
    if (
        source_sufficiency_required == "artifact_contents_required"
        and llm_inline_answer_present < 0.50
        and Q < 0.92
        and not escape_hatch
    ):
        hard_reasons.append("external_artifact_contents_missing")
    if (
        answer_source_type == "runtime_fact"
        and llm_inline_answer_present < 0.50
        and Q < 0.92
        and not escape_hatch
    ):
        hard_reasons.append("runtime_lookup_missing")
    if slot_variant_reject:
        hard_reasons.append("slot_incomplete_across_variants")
    if (
        answer_source_type in {"customer_instance", "runtime_fact"}
        and llm_inline_answer_present < 0.42
        and Q < 0.88
        and not escape_hatch
    ):
        hard_reasons.append("instance_or_runtime_missing")
    if (
        instance_sensitive_question
        and instantiation_gap >= 0.45
        and Q < 0.88
        and not (
            escape_hatch
            and sf >= 0.84
            and ex >= 0.62
        )
    ):
        hard_reasons.append("instance_specific_not_instantiated")
    if (
        customer_identity_like
        and instance_missing_field >= 0.40
        and Q < 0.92
    ):
        hard_reasons.append("customer_identity_not_instantiated")
    if (
        external_fee_schedule_like
        and external_reference_intent >= 0.45
        and Q < 0.92
    ):
        hard_reasons.append("external_fee_schedule_not_included")
    if (
        reseller_commercial_term_like
        and (asks_percent_like or asks_value_like)
        and (outside_document_intent >= 0.10 or instance_missing_field >= 0.18)
        and Q < 0.92
    ):
        hard_reasons.append("commercial_terms_not_instantiated")

    if M >= 0.65 and S_eff >= 0.55 and Q < 0.80:
        if (sf < 0.90 or ex < 0.62) and not base_clause_reference_safe:
            hard_reasons.append("missing_artifact_contents")

    if S_eff >= 0.78 and Q < 0.70:
        if A_raw < 0.55:
            hard_reasons.append("low_answer_evidence_exact")
        if ex < 0.38 and not safe_direct_support_relief and not direct_clause_sufficiency:
            hard_reasons.append("low_exactness")
        if sf < 0.68:
            hard_reasons.append("low_slot_fill")

    no_answer_path_escape = bool(
        A >= 0.78
        and Q >= 0.68
        and sf >= 0.90
        and llm_inline_answer_present >= 0.80
    ) or bool(base_clause_strong_support and Q >= 0.50 and llm_inline_answer_present >= 0.80)
    if (
        not path_exists
        and S_eff >= 0.60
        and not direct_yesno_clause_relief
        and not no_answer_path_escape
    ):
        hard_reasons.append("no_answer_path")
    if n_targets == 0 and S_eff >= 0.55:
        hard_reasons.append("no_targets")

    escape_triggered = False
    # Secondary escape hatch: avoid over-blocking straightforward VALUE extracts
    # where exact_frac is noisy but clause-level support is still strong.
    soft_exactness_escape = (
        "low_exactness" in hard_reasons
        and slot_type == "VALUE"
        and simple_extract_intent >= 0.18
        and exact_intent <= 0.15
        and missing_artifact_intent < 0.20
        and Q >= 0.48
        and A_raw >= 0.70
        and sf >= 0.80
        and path_exists
        and (path_bottleneck or 0.0) >= 0.45
        and O < 0.25
        and P < 0.35
    )
    if soft_exactness_escape and not strong_contrary:
        kept = [r for r in hard_reasons if r != "low_exactness"]
        if len(kept) < len(hard_reasons):
            escape_triggered = True
        hard_reasons = kept

    if escape_hatch and not strong_contrary:
        relaxable = {
            "missing_artifact_contents",
            "low_answer_evidence_exact",
            "low_exactness",
            "low_slot_fill",
            "no_answer_path",
            "no_targets",
        }
        kept = [r for r in hard_reasons if r not in relaxable]
        if len(kept) < len(hard_reasons):
            escape_triggered = True
        hard_reasons = kept

    if hard_reasons:
        predicted = "unanswerable"
        gate_reason = ",".join(sorted(set(hard_reasons)))
    else:
        predicted = "answerable" if score >= threshold else "unanswerable"
        gate_reason = "none"

    # ---- Confidence from uncertainty, not only margin ----
    margin = abs(score - threshold)
    uncertainty = min(
        1.0,
        0.30 * D
        + 0.15 * (1.0 - core_coverage)
        + 0.12 * (1.0 - evidence_coverage)
        + 0.12 * (1.0 - risk_coverage)
        + 0.10 * G
        + 0.08 * S_eff * X
        + 0.06 * B
        + 0.07 * P
        + 0.09 * O
        + 0.10 * Q2_extra
        + 0.05 * (1.0 - Q),
    )
    conf_score = max(0.0, min(1.0, (margin / 0.30) * (1.0 - uncertainty)))
    if conf_score >= 0.72:
        confidence = "high"
    elif conf_score >= 0.42:
        confidence = "medium"
    else:
        confidence = "low"
    if hard_reasons and confidence == "low":
        confidence = "medium"

    return {
        "predicted": predicted,
        "score": round(score, 4),
        "confidence": confidence,
        "threshold": round(threshold, 4),
        "signals": {
            "slot": round(_cal_pos(slot, 0.530, 0.090), 4) if slot is not None else None,
            "path": round(_cal_pos(path, 0.260, 0.100), 4) if path is not None else None,
            "h0": round(_cal_pos(h0, 0.370, 0.060), 4) if h0 is not None else None,
            "trig": round(_cal_pos(trig, 0.420, 0.050), 4) if trig is not None else None,
            "chain": round(_cal_risk(chain, 0.880, 0.200), 4) if chain is not None else None,
            "sheaf": round(_cal_pos(sheaf, 0.700, 0.100), 4) if sheaf is not None else None,
            "w_sheaf": round(_cal_pos(w_sheaf, 0.720, 0.100), 4) if w_sheaf is not None else None,
            "w_ricci": round(_cal_pos(w_ricci, -0.400, 0.120), 4) if w_ricci is not None else None,
            "bridge": round(_cal_pos(bridge, -0.500, 0.100), 4) if bridge is not None else None,
            "type_entropy": round(_cal_risk(type_entropy, 2.200, 0.200), 4) if type_entropy is not None else None,
            "distant": round(distant, 4) if distant is not None else None,
            "placeholder": round(ph, 4) if ph is not None else None,
            "tag": round(_cal_pos(tag, 0.350, 0.120), 4) if tag is not None else None,
            "topic_evidence": round(T, 4),
            "answer_evidence": round(A, 4),
            "answer_evidence_raw": round(A_raw, 4),
            "slot_fill_ratio": round(sf, 4),
            "direct_fill_ratio": round(df, 4),
            "exact_frac": round(ex, 4),
            "adjacent_frac": round(adjacent, 4) if adjacent is not None else None,
            "target_frac": round(target_frac, 4) if target_frac is not None else None,
            "direct_explicit_support_Q": round(Q, 4),
            "outside_doc_needed": round(outside_doc_needed, 4),
            "asks_for_external_contents": round(asks_for_external_contents, 4),
            "external_ref_only_ratio": round(ext_ref_only_ratio, 4),
            "external_inline_substance_ratio": round(ext_inline_substance_ratio, 4),
        },
        "meta": {
            "slot_type": slot_type,
            "question": question_text,
            "question_intent": intent,
            "proof_type_initial": proof_type_initial,
            "proof_type_routed": proof_type_routed,
            "lane": lane,
            "answer_form": answer_form,
            "answer_source_type": answer_source_type,
            "source_sufficiency_required": source_sufficiency_required,
            "missing_if_not_present": bool(missing_if_not_present),
            "customer_instance_dependency": round(customer_instance_dependency, 4),
            "external_artifact_dependency": round(external_artifact_dependency, 4),
            "runtime_dependency": round(runtime_dependency, 4),
            "llm_evidence_sufficiency": llm_ev,
            "llm_inline_answer_present": round(llm_inline_answer_present, 4),
            "llm_external_reference_only": round(llm_external_reference_only, 4),
            "llm_customer_instance_missing": round(llm_customer_instance_missing, 4),
            "llm_runtime_missing": round(llm_runtime_missing, 4),
            "outside_doc_needed": round(outside_doc_needed, 4),
            "asks_for_external_contents": round(asks_for_external_contents, 4),
            "external_doc_type": external_doc_type,
            "expects_direct_rule_clause": round(expects_direct_rule_clause, 4),
            "direct_rule_clause_override": bool(direct_rule_clause_override),
            "direct_explicit_support": des,
            "direct_support_source": Q_source,
            "required_slots": req_slots,
            "matched_slots": tpc_matched,
            "path_exists": path_exists,
            "path_bottleneck": round(path_bottleneck, 4) if path_bottleneck is not None else None,
            "n_targets": n_targets,
            "n_nodes": n_nodes_total,
            "target_frac": round(target_frac, 4) if target_frac is not None else None,

            # Requested full feature logging
            "T": round(T, 4),
            "A": round(A, 4),
            "D": round(D, 4),
            "S": round(S, 4),
            "S_eff": round(S_eff, 4),
            "P": round(P, 4),
            "O": round(O, 4),
            "Q": round(Q, 4),
            "M": round(M, 4),
            "X": round(X, 4),
            "G": round(G, 4),
            "B": round(B, 4),
            "score": round(score, 4),
            "threshold": round(threshold, 4),
            "gate_reason": gate_reason,
            "hard_gate_reasons": hard_reasons,
            "hard_gate_triggered": bool(hard_reasons),
            "escape_hatch_triggered": bool(escape_triggered),
            "escape_hatch_eligible": bool(escape_hatch),
            "soft_exactness_escape": bool(soft_exactness_escape),
            "strong_contrary": bool(strong_contrary),

            # Compatibility keys used by existing eval tooling/UI
            "evidence_score": round(A, 4),
            "graph_evidence": round(T, 4),
            "answer_evidence": round(A, 4),
            "answer_evidence_raw": round(A_raw, 4),
            "slot_fill_ratio": round(sf, 4),
            "risk_score": round(risk, 4),
            "core_coverage": round(core_coverage, 4),
            "evidence_coverage": round(evidence_coverage, 4),
            "risk_coverage": round(risk_coverage, 4),
            "disagreement": round(D, 4),
            "uncertainty": round(uncertainty, 4),
            "z_raw": round(z, 4),
            "exact_intent": round(exact_intent, 4),
            "simple_extract_intent": round(simple_extract_intent, 4),
            "speculative_intent": round(speculative_intent, 4),
            "outside_document_intent": round(outside_document_intent, 4),
            "missing_artifact_intent": round(missing_artifact_intent, 4),
            "customer_specific_dependency_prior": round(customer_specific_dependency_prior, 4),
            "external_doc_dependency_prior": round(external_doc_dependency_prior, 4),
            "instance_missing_field": round(instance_missing_field, 4),
            "external_reference_intent": round(external_reference_intent, 4),
            "consensus_vote_stats": vote_stats,
            "mean_vote_frac": (round(mean_vote_frac, 4) if mean_vote_frac is not None else None),
            "p75_vote_frac": (round(p75_vote_frac, 4) if p75_vote_frac is not None else None),
            "high_vote_frac": (round(high_vote_frac, 4) if high_vote_frac is not None else None),
            "max_vote_frac": (round(max_vote_frac, 4) if max_vote_frac is not None else None),
            "n_vote_ge2": int(n_vote_ge2),
            "n_vote_ge3": int(n_vote_ge3),
            "exact_slot_heavy": bool(exact_slot_heavy),
            "weak_slot_channels": bool(weak_slot_channels),
            "weak_variant_consensus": bool(weak_variant_consensus),
            "slot_variant_reject": bool(slot_variant_reject),
            "no_answer_path_escape": bool(no_answer_path_escape),
            "customer_specific_dependency_refined": round(customer_specific_dependency_refined, 4),
            "external_doc_dependency_refined": round(external_doc_dependency_refined, 4),
            "external_dependency_evidence": extdep,
            "external_related_ratio": round(ext_related_ratio, 4),
            "external_ref_only_ratio": round(ext_ref_only_ratio, 4),
            "external_inline_substance_ratio": round(ext_inline_substance_ratio, 4),
            "external_doc_match_ratio": round(ext_doc_match_ratio, 4),
            "Q2": round(Q2, 4),
            "Q2_extra": round(Q2_extra, 4),
            "instantiation_gap": round(instantiation_gap, 4),
            "instance_sensitive_question": bool(instance_sensitive_question),
            "customer_identity_like": bool(customer_identity_like),
            "external_fee_schedule_like": bool(external_fee_schedule_like),
            "reseller_commercial_term_like": bool(reseller_commercial_term_like),
            "deterministic_source_route": deterministic_source_route,
            "customer_instance_markers": bool(customer_instance_markers),
            "external_artifact_markers": bool(external_artifact_markers),
            "audit_frequency_rule_like": bool(audit_frequency_rule_like),
            "url_location_rule_like": bool(url_location_rule_like),
            "address_question_rule_like": bool(address_question_rule_like),
            "direct_clause_sufficiency": bool(direct_clause_sufficiency),
            "direct_yesno_clause_relief": bool(direct_yesno_clause_relief),
            "base_clause_strong_support": bool(base_clause_strong_support),
            "definition_base_clause_relief": bool(definition_base_clause_relief),
            "rule_reference_base_relief": bool(rule_reference_base_relief),
            "base_rule_despite_reference": bool(base_rule_despite_reference),
            "base_clause_reference_safe": bool(base_clause_reference_safe),
            "mentions_external_doc": bool(mentions_external_doc),
            "asks_external_contents": bool(asks_external_contents),
            "asks_external_contents_guard": bool(asks_external_contents_guard),
            "safe_direct_support_relief": bool(safe_direct_support_relief),
            "q2_local_scale": round(float(q2_local_scale), 4),
            "q2_threshold_scale": round(float(q2_threshold_scale), 4),
            "q2_score_penalty_coef": round(float(q2_score_penalty_coef), 4),
            "sxx_soften": round(float(sxx_soften), 4),
            "sx_penalty_scale": round(sx_penalty_scale, 4),
            "thr_rebate_disagreement": round(thr_rebate_disagreement, 4),
            "thr_rebate_support": round(thr_rebate_support, 4),
            "thr_rebate_clean_extract": round(thr_rebate_clean_extract, 4),
            "thr_rebate_nearmiss": round(thr_rebate_nearmiss, 4),
            "thr_rebate_safe_recover": round(thr_rebate_safe_recover, 4),
            "thr_rebate_rule_nearmiss": round(thr_rebate_rule_nearmiss, 4),
            "thr_rebate_base_clause": round(thr_rebate_base_clause, 4),
        },
    }
