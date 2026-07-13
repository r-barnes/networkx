"""
Algorithms for partitioning weighted trees into connected components.

Supports MIN-MAX and MAX-MIN objectives over four built-in additive weight
functions, selected via the ``weight_function`` keyword:
``"vertex_weight_sum"``, ``"edge_weight_sum"``, ``"mixed_sum"``, and
``"vertex_count"``.

Internally every weight function is *additive*: it assigns a weight to
every node and every edge, and a component's weight is the sum of its node
and edge weights.  Additivity is what the algorithms rely on — it makes
component weight monotone under merging (so the greedy feasibility oracles
are optimal, Kundu-Misra style) and puts every achievable component weight
on the bisection grid (so the threshold search is exact).  Non-additive
objectives (e.g. component diameter) are not supported.  Each built-in is
described by a small ``_WeightSpec`` in the ``_WEIGHT_FUNCTIONS`` table;
adding a new built-in means adding a table entry.  None of this machinery
is public API.

Complexity
----------
Let N = |V(T)| and let B be the number of feasibility probes.  The
threshold search bisects the grid of representable weight values with
achieved-value snapping and candidate-verification probes, so
B = O(64) when any weight is a float (IEEE-754 doubles) and B = O(log W)
when all weights are integers of total magnitude W (Python's
arbitrary-precision ``int``); in practice verification and the pigeonhole
seed usually terminate the search after a handful of probes.

- ``min_max_tree_partition``: O(N log N · B) — the feasibility oracle
  sorts children by residual weight at each node.
- ``max_min_tree_partition``: O(N · B) — feasibility is a single
  post-order pass with no per-node sort.
"""

from __future__ import annotations

import math
import numbers
import struct
import sys
from collections.abc import Hashable
from typing import Any, NamedTuple

import networkx as nx

__all__ = [
    "min_max_tree_partition",
    "max_min_tree_partition",
]


# ---------------------------------------------------------------------------
# Weight function interface and built-in implementations
# ---------------------------------------------------------------------------


class _WeightSpec(NamedTuple):
    """Additive weight specification: where node and edge weights come from.

    Component weight is the sum of per-node and per-edge weights.
    Additivity is what the algorithms rely on — it makes component weight
    monotone under merging (so the greedy feasibility oracles are optimal)
    and puts every achievable component weight on the bisection grid (so
    the threshold search is exact).  Non-additive objectives (e.g.
    component diameter) cannot be expressed.

    ``node_attr``/``edge_attr`` name the attribute to read (elements
    missing the attribute default to 1); ``None`` means every element
    contributes a constant instead — ``node_const`` for nodes, 0 for
    edges.  Integer weights are kept exact at arbitrary precision;
    everything else is coerced to float.
    """

    node_attr: str | None
    node_const: int
    edge_attr: str | None


# The four public weight_function names, each mapping the public
# (node_weight, edge_weight) attribute-name arguments to a spec:
# "vertex_weight_sum" reads node weights (validated > 0);
# "edge_weight_sum" reads edge weights (validated >= 0 — zero is allowed
# because a tree with all-zero edge weights still has a well-defined
# partition, and single-vertex components weigh 0);
# "mixed_sum" reads both; "vertex_count" reads neither and counts 1 per
# node.  Adding a built-in weight function means adding an entry here.
_WEIGHT_FUNCTIONS = {
    "vertex_weight_sum": lambda nw, ew: _WeightSpec(nw, 1, None),
    "edge_weight_sum": lambda nw, ew: _WeightSpec(None, 0, ew),
    "mixed_sum": lambda nw, ew: _WeightSpec(nw, 1, ew),
    "vertex_count": lambda nw, ew: _WeightSpec(None, 1, None),
}


def _coerce_weight(w: Any) -> int | float:
    """Keep integers exact (arbitrary precision); coerce all else to float."""
    return w if isinstance(w, numbers.Integral) else float(w)


def _check_node_weight(v: Hashable, w: Any) -> None:
    """Shared node-weight validation: numeric, finite, > 0."""
    if not isinstance(w, numbers.Integral):
        # Integers are always finite (and may exceed float range, so must
        # not be converted); everything else must convert to a finite float.
        try:
            w = float(w)
        except (TypeError, ValueError) as err:
            raise nx.NetworkXError(
                f"Node {v!r} has non-numeric weight {w!r}; all weights must be numbers."
            ) from err
        if not math.isfinite(w):
            raise nx.NetworkXError(
                f"Node {v!r} has non-finite weight {w!r}; all weights must be finite."
            )
    if w <= 0:
        raise nx.NetworkXError(
            f"Node {v!r} has weight {w!r}; all node weights must be > 0."
        )


def _check_edge_weight(u: Hashable, v: Hashable, w: Any) -> None:
    """Shared edge-weight validation: numeric, finite, >= 0."""
    if not isinstance(w, numbers.Integral):
        try:
            w = float(w)
        except (TypeError, ValueError) as err:
            raise nx.NetworkXError(
                f"Edge ({u!r}, {v!r}) has non-numeric weight {w!r}; "
                "all weights must be numbers."
            ) from err
        if not math.isfinite(w):
            raise nx.NetworkXError(
                f"Edge ({u!r}, {v!r}) has non-finite weight {w!r}; "
                "all weights must be finite."
            )
    if w < 0:
        raise nx.NetworkXError(
            f"Edge ({u!r}, {v!r}) has weight {w!r}; all edge weights must be >= 0."
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _root_at_leaf(T: nx.Graph) -> Hashable:
    """Return a leaf node of T (degree 1, or the only node if |V|=1)."""
    if len(T) == 1:
        return next(iter(T.nodes()))
    return next(v for v, d in T.degree() if d == 1)


def _root_tree(
    T: nx.Graph, root: Hashable
) -> tuple[
    list[Hashable],
    dict[Hashable, Hashable | None],
    dict[Hashable, list[Hashable]],
]:
    """Root tree at *root*.

    Returns (post_order, parent, children) where:
    - post_order: list of nodes in post-order (leaves first)
    - parent: dict mapping node -> parent (root -> None)
    - children: dict mapping node -> list of child nodes
    """
    parent = {root: None}
    children = {v: [] for v in T}
    post_order = []
    stack = [(root, iter(T.adj[root]))]
    visited = {root}
    while stack:
        v, it = stack[-1]
        found = False
        for u in it:
            if u not in visited:
                visited.add(u)
                parent[u] = v
                children[v].append(u)
                stack.append((u, iter(T.adj[u])))
                found = True
                break
        if not found:
            post_order.append(v)
            stack.pop()
    return post_order, parent, children


def _weight_maps(
    T: nx.Graph, spec: _WeightSpec
) -> tuple[
    dict[Hashable, int | float],
    dict[tuple[Hashable, Hashable], int | float],
]:
    """Precompute every node and edge weight once per public call.

    Returns (node_w, edge_w).  ``edge_w`` stores *both* orientations
    ``(u, v)`` and ``(v, u)`` so hot-path lookups need no frozenset
    construction or orientation normalization.
    """
    if spec.node_attr is None:
        node_w = dict.fromkeys(T, spec.node_const)
    else:
        node_w = {v: _coerce_weight(T.nodes[v].get(spec.node_attr, 1)) for v in T}
    edge_w = {}
    if spec.edge_attr is None:
        for u, v in T.edges():
            edge_w[u, v] = 0
            edge_w[v, u] = 0
    else:
        for u, v, d in T.edges(data=True):
            w = _coerce_weight(d.get(spec.edge_attr, 1))
            edge_w[u, v] = w
            edge_w[v, u] = w
    return node_w, edge_w


def _weights_integral(
    node_w: dict[Hashable, int | float],
    edge_w: dict[tuple[Hashable, Hashable], int | float],
) -> bool:
    """True iff every weight is an integer, enabling exact integer bisection."""
    return all(isinstance(w, numbers.Integral) for w in node_w.values()) and all(
        isinstance(w, numbers.Integral) for w in edge_w.values()
    )


def _validate_partition_args(T: nx.Graph, q: int, spec: _WeightSpec) -> None:
    """Raise appropriate errors for invalid inputs."""
    if not nx.is_tree(T):
        raise nx.NotATree("input graph is not a tree")
    if not isinstance(q, numbers.Integral):
        raise nx.NetworkXError(f"q must be an integer, got {q!r}.")
    n = len(T)
    if q < 1 or q > n:
        raise nx.NetworkXError(
            f"q must satisfy 1 <= q <= number of nodes ({n}), got q={q}."
        )
    if spec.node_attr is not None:
        for v in T.nodes:
            _check_node_weight(v, T.nodes[v].get(spec.node_attr, 1))
    if spec.edge_attr is not None:
        for u, v, edata in T.edges(data=True):
            _check_edge_weight(u, v, edata.get(spec.edge_attr, 1))


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------


def _components_from_cut_edges(
    T: nx.Graph, cut_edges: set[frozenset[Hashable]]
) -> list[list[Hashable]]:
    """Connected components of T after removing cut_edges.

    *cut_edges* is a set of ``frozenset({u, v})`` keys.
    """
    comps = []
    visited = set()
    for start in T:
        if start in visited:
            continue
        comp = []
        stk = [start]
        visited.add(start)
        while stk:
            x = stk.pop()
            comp.append(x)
            for y in T.neighbors(x):
                if y in visited:
                    continue
                if frozenset((x, y)) in cut_edges:
                    continue
                visited.add(y)
                stk.append(y)
        comps.append(comp)
    return comps


def _component_weight(
    T: nx.Graph,
    comp_vertices: list[Hashable],
    node_w: dict[Hashable, int | float],
    edge_w: dict[tuple[Hashable, Hashable], int | float],
) -> int | float:
    """Weight of a connected component: node weights + internal edge weights."""
    total = 0
    for v in comp_vertices:
        total += node_w[v]
    comp_set = set(comp_vertices)
    counted = set()
    for v in comp_vertices:
        for u in T.adj[v]:
            if u in comp_set and u not in counted:
                total += edge_w[v, u]
        counted.add(v)
    return total


# ---------------------------------------------------------------------------
# Feasibility tests
# ---------------------------------------------------------------------------


def _feasible_minmax(
    lam: float,
    post_order: list[Hashable],
    children: dict[Hashable, list[Hashable]],
    node_w: dict[Hashable, int | float],
    ew_par: dict[Hashable, int | float],
    root: Hashable,
) -> tuple[bool, list[tuple[Hashable, Hashable]] | None, float | None]:
    """Greedy feasibility test for min-max at threshold *lam*.

    For each vertex v in post-order, start from v's own weight and greedily
    merge each child's residual whenever the merged weight stays <= lam.
    Otherwise cut the edge (the child becomes a finished part with weight
    <= lam).  Comparisons against *lam* are exact — the outer bisection
    over the grid of representable weights depends on that.

    Children are processed in ascending candidate-weight order (the running
    weight of v is a common additive offset, so ordering by residual + edge
    weight is the same order).  This is the Kundu-Misra greedy: it yields
    the minimum number of cuts at threshold lam and a residual that is
    monotone non-increasing in lam, which is the property the outer
    bisection needs.

    Returns (ok, cuts, achieved):
      ok       -- False iff some node's own weight > lam (infeasible).
      cuts     -- cut edges as (child, parent) tuples; only the winning
                  probe's cuts are converted to frozenset keys, once, by
                  the solver.
      achieved -- max weight over the finished parts (cut subtrees and the
                  root residual): the max component weight of the partition
                  the cuts induce, <= lam when ok.
    """
    state = {}
    cuts = []
    achieved = None
    for v in post_order:
        sv = node_w[v]
        if sv > lam:
            return False, None, None
        kids = children[v]
        if len(kids) > 1:
            kids = sorted(kids, key=lambda c: state[c] + ew_par[c])
        for c in kids:
            cand = sv + state[c] + ew_par[c]
            if cand <= lam:
                sv = cand
            else:
                cuts.append((c, v))
                wc = state[c]
                if achieved is None or wc > achieved:
                    achieved = wc
        state[v] = sv
    w_root = state[root]
    if achieved is None or w_root > achieved:
        achieved = w_root
    return True, cuts, achieved


def _feasible_maxmin(
    lam: float,
    post_order: list[Hashable],
    children: dict[Hashable, list[Hashable]],
    node_w: dict[Hashable, int | float],
    ew_par: dict[Hashable, int | float],
    root: Hashable,
) -> tuple[int, list[tuple[Hashable, Hashable]], float | None, float]:
    """Greedy feasibility test for max-min at threshold *lam*.

    For each vertex v in post-order, accumulate children's residuals; as soon
    as a child's residual weight is >= lam, cut the edge (that child becomes
    a finished "heavy" part with weight >= lam).  Otherwise merge it in.
    Comparisons against *lam* are exact — the outer bisection over the grid
    of representable weights depends on that.

    Returns (heavy_count, cuts, min_heavy, root_weight):
      heavy_count -- number of parts with weight >= lam (cut subtrees plus
                     the root residual if it qualifies).
      cuts        -- cut edges as (child, parent) tuples.
      min_heavy   -- smallest weight among the heavy parts (None if there
                     are none); >= lam by construction.
      root_weight -- weight of the root's final residual.
    """
    state = {}
    cuts = []
    min_heavy = None
    for v in post_order:
        sv = node_w[v]
        for c in children[v]:
            wc = state[c]
            if wc >= lam:
                cuts.append((c, v))
                if min_heavy is None or wc < min_heavy:
                    min_heavy = wc
            else:
                sv += wc + ew_par[c]
        state[v] = sv
    root_w = state[root]
    heavy_count = len(cuts)
    if root_w >= lam:
        heavy_count += 1
        if min_heavy is None or root_w < min_heavy:
            min_heavy = root_w
    return heavy_count, cuts, min_heavy, root_w


# ---------------------------------------------------------------------------
# Post-processing to reach exactly q parts
# ---------------------------------------------------------------------------


def _add_cuts_to_reach_q(
    T: nx.Graph, cut_edges: set[frozenset[Hashable]], q: int
) -> set[frozenset[Hashable]]:
    """Add arbitrary internal edges as cuts until exactly q parts.

    Every tree edge is a bridge, so each non-cut edge added as a cut
    increases the component count by exactly 1.  To reach q parts we
    need q - 1 total cuts; pick any non-cut edges to fill the deficit.

    Safe for monotone W: splitting can only decrease the max component weight.
    """
    need = (q - 1) - len(cut_edges)
    if need <= 0:
        return cut_edges
    for u, v in T.edges():
        if need <= 0:
            break
        key = frozenset((u, v))
        if key not in cut_edges:
            cut_edges.add(key)
            need -= 1
    return cut_edges


def _reduce_to_q_parts(
    T: nx.Graph,
    cut_edges: set[frozenset[Hashable]],
    q: int,
    node_w: dict[Hashable, int | float],
    edge_w: dict[tuple[Hashable, Hashable], int | float],
) -> list[tuple[frozenset, int | float]]:
    """Reduce parts to exactly q for max-min by merging the lightest component
    with a neighbor, returning the final partition as (nodes, weight) pairs.

    Uses union-find (merging vertex lists and weights in place, so the final
    partition needs no second components-and-weights pass).  Sort keys are
    computed once from the initial component weights and are not refreshed
    after merges.  This still preserves the max-min optimum because the
    lightest cut is always adjacent to the at-most-one sub-threshold
    component, which is absorbed into a heavy neighbor before any
    heavy-only cuts are considered; every resulting component therefore
    contains at least one heavy initial component (weight >= lambda*).  The
    reduction is optimal-valued though not necessarily optimal-shaped.
    """
    comps = _components_from_cut_edges(T, cut_edges)
    weights = [_component_weight(T, c, node_w, edge_w) for c in comps]
    if len(comps) <= q:
        return [(frozenset(c), w) for c, w in zip(comps, weights)]

    comp_of = {}
    for i, comp in enumerate(comps):
        for v in comp:
            comp_of[v] = i

    uf_parent = list(range(len(comps)))
    uf_weight = list(weights)
    uf_verts = comps  # merged in place; `comps` is not used afterwards

    def find(x):
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]
        return x

    # Deterministic tiebreaker: index each cut by its position in T.edges()
    # iteration, which follows adjacency-dict insertion order.  Using id(cut)
    # or sorted(cut) would be nondeterministic (memory addresses) or fail on
    # mixed-type node labels.
    edge_index = {frozenset(e): i for i, e in enumerate(T.edges())}
    cut_info = []
    for cut in cut_edges:
        a, b = tuple(cut)
        ca, cb = comp_of[a], comp_of[b]
        min_w = min(weights[ca], weights[cb])
        cut_info.append((min_w, edge_index[cut], cut, ca, cb))
    cut_info.sort()

    n_comps = len(comps)
    for _, _, cut, ca, cb in cut_info:
        if n_comps <= q:
            break
        ra, rb = find(ca), find(cb)
        if ra == rb:
            continue
        # Union by vertex-list size keeps list merging O(n log n); root
        # choice does not affect the resulting partition.
        if len(uf_verts[ra]) > len(uf_verts[rb]):
            ra, rb = rb, ra
        uf_parent[ra] = rb
        a, b = tuple(cut)
        # Un-cutting reconnects the edge, so its weight rejoins the part.
        uf_weight[rb] += uf_weight[ra] + edge_w[a, b]
        uf_verts[rb].extend(uf_verts[ra])
        uf_verts[ra] = []
        n_comps -= 1

    roots = {find(i) for i in range(len(uf_parent))}
    return [(frozenset(uf_verts[r]), uf_weight[r]) for r in roots]


# ---------------------------------------------------------------------------
# Bisection grid
# ---------------------------------------------------------------------------
#
# The threshold search bisects a *grid* of representable weight values rather
# than a real interval: the integers when every weight is an integer, the
# nonnegative IEEE-754 doubles (ordered by bit pattern) otherwise.  Every
# achievable component weight lies on the grid, so when the bracket contains
# no interior grid point the feasible endpoint is exactly optimal — no
# tolerance parameters are involved.


def _bits(x: float) -> int:
    """Ordered integer bit pattern of a nonnegative finite double."""
    return struct.unpack("<q", struct.pack("<d", x))[0]


def _from_bits(b: int) -> float:
    return struct.unpack("<d", struct.pack("<q", b))[0]


def _grid_mid(lo: int | float, hi: int | float, integral: bool) -> int | float:
    """A grid value strictly between lo and hi (callers ensure one exists)."""
    if integral:
        return (lo + hi) // 2
    return _from_bits((_bits(lo) + _bits(hi)) // 2)


def _grid_adjacent(lo: int | float, hi: int | float, integral: bool) -> bool:
    """True if no grid value lies strictly between lo and hi."""
    if integral:
        return hi - lo <= 1
    return _bits(hi) - _bits(lo) <= 1


def _grid_prev(x: int | float, integral: bool) -> int | float:
    """Largest grid value < x."""
    if integral:
        return x - 1
    return math.nextafter(x, -math.inf)


def _grid_next(x: int | float, integral: bool) -> int | float:
    """Smallest grid value > x."""
    if integral:
        return x + 1
    return math.nextafter(x, math.inf)


def _grid_up(w: int | float, integral: bool) -> int | float:
    """Smallest grid value >= w (w itself when already on the grid)."""
    if integral:
        return w
    try:
        f = float(w)
    except OverflowError:
        # An all-integer component weight beyond float range in a tree that
        # also has float weights; +inf is the smallest double >= w.
        return math.inf
    if f < w:
        f = math.nextafter(f, math.inf)
    return f


def _grid_down(w: int | float, integral: bool) -> int | float:
    """Largest grid value <= w (w itself when already on the grid)."""
    if integral:
        return w
    try:
        f = float(w)
    except OverflowError:
        return sys.float_info.max
    if f > w:
        f = math.nextafter(f, -math.inf)
    return f


# ---------------------------------------------------------------------------
# Bisection solvers
# ---------------------------------------------------------------------------


def _binary_search_minmax(
    T: nx.Graph,
    q: int,
    node_w: dict[Hashable, int | float],
    edge_w: dict[tuple[Hashable, Hashable], int | float],
    integral: bool,
) -> list[tuple[frozenset, int | float]]:
    """Min-max q-partition via exact grid bisection + greedy oracle.

    Bisects the grid of representable threshold values, snapping the
    feasible upper bound to the weight actually achieved by each feasible
    probe.  Two probe-count optimizations preserve exactness because every
    probe verifies which side of the optimum it lands on:

    - a pigeonhole seed (some part must weigh at least total/q, so the
      first probe lands near the average instead of bit-space middle);
    - candidate verification (after each snap, probe one grid step below
      the candidate — if that is infeasible the candidate is optimal and
      the search ends immediately; alternated with bisection steps so the
      worst case stays O(grid bisection)).

    Terminates when the bracket contains no interior grid point, at which
    point the upper bound is the exact optimum (with respect to the
    oracle's summation order in the float case).
    """
    n = len(T)
    verts = list(T)

    if q == 1:
        return [(frozenset(verts), _component_weight(T, verts, node_w, edge_w))]

    if q == n:
        return [(frozenset([v]), node_w[v]) for v in verts]

    root = _root_at_leaf(T)
    post_order, parent, children = _root_tree(T, root)
    ew_par = {v: edge_w[v, p] for v, p in parent.items() if p is not None}

    def probe(lam):
        ok, cuts, achieved = _feasible_minmax(
            lam, post_order, children, node_w, ew_par, root
        )
        return ok and len(cuts) + 1 <= q, cuts, achieved

    def cuts_to_partition(cuts):
        cut_keys = _add_cuts_to_reach_q(T, {frozenset(e) for e in cuts}, q)
        comps = _components_from_cut_edges(T, cut_keys)
        return [(frozenset(c), _component_weight(T, c, node_w, edge_w)) for c in comps]

    # No partition can beat the heaviest single vertex; if that lower bound
    # is already feasible it is optimal (covers degenerate cases such as
    # all-zero edge weights, where every threshold is feasible at 0).
    lo = max(node_w.values())
    ok, cuts, _ = probe(lo)
    if ok:
        return cuts_to_partition(cuts)

    # An unbounded threshold is always feasible with zero cuts; its achieved
    # weight (the whole-tree weight, in oracle summation order) seeds the
    # feasible upper bound.
    _, best_cuts, hi = probe(math.inf)
    lo = _grid_down(lo, integral)
    hi = _grid_up(hi, integral)

    # Pigeonhole seed: some part must weigh at least total/q.
    guess = -(-hi // q) if integral else hi / q
    verified = True
    if lo < guess < hi:
        ok, cuts, achieved = probe(guess)
        if ok:
            hi = _grid_up(achieved, integral)
            best_cuts = cuts
            verified = False
        else:
            lo = guess

    # Invariants: lo is infeasible, hi is feasible and achieved by
    # best_cuts, and the optimum lies in (lo, hi].  Every achievable
    # component weight is on the grid, so an empty open bracket means
    # hi is the optimum.
    while not _grid_adjacent(lo, hi, integral):
        verifying = not verified
        mid = _grid_prev(hi, integral) if verifying else _grid_mid(lo, hi, integral)
        ok, cuts, achieved = probe(mid)
        if ok:
            hi = _grid_up(achieved, integral)
            best_cuts = cuts
            verified = verifying
        else:
            lo = mid
            verified = True

    return cuts_to_partition(best_cuts)


def _binary_search_maxmin(
    T: nx.Graph,
    q: int,
    node_w: dict[Hashable, int | float],
    edge_w: dict[tuple[Hashable, Hashable], int | float],
    integral: bool,
) -> list[tuple[frozenset, int | float]]:
    """Max-min q-partition via exact grid bisection + greedy oracle.

    Mirror image of `_binary_search_minmax` (see there for the seeding and
    candidate-verification scheme): the feasible lower bound snaps up to
    the lightest heavy part achieved by each feasible probe, and the
    optimum is the final lower bound.
    """
    n = len(T)
    verts = list(T)

    if q == 1:
        return [(frozenset(verts), _component_weight(T, verts, node_w, edge_w))]

    if q == n:
        return [(frozenset([v]), node_w[v]) for v in verts]

    root = _root_at_leaf(T)
    post_order, parent, children = _root_tree(T, root)
    ew_par = {v: edge_w[v, p] for v, p in parent.items() if p is not None}

    def probe(lam):
        heavy_count, cuts, min_heavy, root_w = _feasible_maxmin(
            lam, post_order, children, node_w, ew_par, root
        )
        return heavy_count >= q, cuts, min_heavy, root_w

    # lam = 0 cuts every edge (all weights are nonnegative), so it is
    # feasible whenever q <= n; snap the feasible bound up to the lightest
    # part it actually produced.
    _, best_cuts, achieved, _ = probe(0)
    lo = _grid_down(achieved, integral)

    # No q >= 2 disjoint parts can each weigh as much as the whole tree
    # (unless the total is zero, in which case lo == hi already), so the
    # whole-tree weight — the unbounded probe's residual — is an infeasible
    # upper bound.
    _, _, _, total = probe(math.inf)
    hi = _grid_up(total, integral)

    # Pigeonhole seed: the lightest part can weigh at most total/q.
    guess = hi // q if integral else hi / q
    verified = True
    if lo < guess < hi:
        ok, cuts, achieved, _ = probe(guess)
        if ok:
            lo = _grid_down(achieved, integral)
            best_cuts = cuts
            verified = False
        else:
            hi = guess

    # Invariants: lo is feasible and achieved by best_cuts, hi is
    # infeasible, and the optimum lies in [lo, hi).  Every achievable
    # component weight is on the grid, so an empty open bracket means
    # lo is the optimum.
    while not _grid_adjacent(lo, hi, integral):
        verifying = not verified
        mid = _grid_next(lo, integral) if verifying else _grid_mid(lo, hi, integral)
        ok, cuts, achieved, _ = probe(mid)
        if ok:
            lo = _grid_down(achieved, integral)
            best_cuts = cuts
            verified = verifying
        else:
            hi = mid
            verified = True

    return _reduce_to_q_parts(T, {frozenset(e) for e in best_cuts}, q, node_w, edge_w)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sorted_components(
    partition: list[tuple[frozenset, float]], descending: bool
) -> list[frozenset]:
    """Sort (nodes, weight) pairs by weight and return just the node sets."""
    ordered = sorted(partition, key=lambda x: x[1], reverse=descending)
    return [nodes for nodes, _ in ordered]


def _resolve_weight_function(
    weight_function: str, node_weight: str, edge_weight: str
) -> _WeightSpec:
    """Map a weight-function name to its additive weight specification."""
    try:
        make_spec = _WEIGHT_FUNCTIONS[weight_function]
    except KeyError:
        raise nx.NetworkXError(
            f"weight_function must be one of {sorted(_WEIGHT_FUNCTIONS)}, "
            f"got {weight_function!r}."
        ) from None
    return make_spec(node_weight, edge_weight)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@nx.utils.not_implemented_for("directed")
@nx.utils.not_implemented_for("multigraph")
# node_attrs/edge_attrs are over-declared for weight_function="vertex_count"
# (which reads neither attribute); the dispatcher will pass attribute data
# through unused, which is benign and matches NetworkX convention for
# optional-attribute dispatch.
@nx._dispatchable(graphs="T", node_attrs="node_weight", edge_attrs="edge_weight")
def min_max_tree_partition(
    T: nx.Graph,
    q: int,
    node_weight: str = "weight",
    edge_weight: str = "weight",
    *,
    weight_function: str = "vertex_weight_sum",
) -> list[frozenset]:
    r"""Partition a weighted tree into ``q`` components minimizing the maximum
    component weight.

    Removes ``q - 1`` edges from a tree ``T`` to produce ``q`` connected
    subtrees (components) such that the weight of the heaviest component
    is as small as possible.

    By default, component weight is the sum of vertex weights, read from
    the node attribute ``weight`` (nodes missing the attribute are assigned
    weight 1).  To use a different notion of component weight, set
    *weight_function*.

    Parameters
    ----------
    T : NetworkX Graph
        An undirected tree (connected acyclic graph).

    q : int
        Number of components to partition ``T`` into. Must satisfy
        ``1 <= q <= len(T)``.

    node_weight : str, optional (default ``"weight"``)
        Node attribute key read by ``"vertex_weight_sum"`` and ``"mixed_sum"``.

    edge_weight : str, optional (default ``"weight"``)
        Edge attribute key read by ``"edge_weight_sum"`` and ``"mixed_sum"``.

    weight_function : str, optional (default ``"vertex_weight_sum"``)
        How component weight is defined.  One of:

        - ``"vertex_weight_sum"`` — sum of vertex weights (reads
          ``node_weight``; weights must be finite and > 0).
        - ``"edge_weight_sum"`` — sum of edge weights (reads
          ``edge_weight``; weights must be finite and >= 0).  Singleton
          components have weight 0.
        - ``"mixed_sum"`` — sum of vertex and edge weights (reads both;
          vertex weights > 0, edge weights >= 0, all finite).
        - ``"vertex_count"`` — number of vertices (reads neither).

    Returns
    -------
    partition : list of frozenset
        A list of ``q`` frozensets, each holding the node labels of one
        component.  The list is sorted in **descending** order of component
        weight (heaviest component first), because the heaviest component
        is the quantity being minimized.

    Raises
    ------
    NetworkXNotImplemented
        If ``T`` is directed or a multigraph.

    NetworkXError
        If ``q`` is not an integer, ``q`` is not in ``[1, len(T)]``,
        ``weight_function`` is unrecognised, a node or edge weight is
        invalid for the selected weight function (non-numeric, non-finite,
        or out of range), or float weights are mixed with integer weights
        too large to represent as floats.

    NetworkXPointlessConcept
        If ``T`` has no nodes.

    NotATree
        If ``T`` is not a tree.

    Notes
    -----
    The algorithm bisects over threshold values combined with a greedy
    post-order DFS feasibility oracle.  At each candidate threshold λ, the
    oracle roots the tree and greedily merges children in ascending
    residual-weight order, cutting an edge whenever merging would exceed λ.

    The bisection runs over the grid of representable weight values and
    snaps to weights actually achieved by feasible probes, so there are no
    tolerance parameters.  When every weight is an integer the grid is the
    integers (Python's arbitrary-precision ``int``) and the result is exact
    at any magnitude; otherwise the grid is the IEEE-754 doubles and the
    result is exact with respect to floating-point summation of component
    weights.  With B feasibility probes (B = O(64) for float weights,
    B = O(log W) for integer weights of total magnitude W; a pigeonhole
    seed and candidate-verification probes usually finish far sooner),
    each costing O(n log n) for the sort, the overall complexity is
    :math:`O(n \log n \cdot B)`.

    References
    ----------
    .. [1] S. Kundu and J. Misra,
       "A linear tree partitioning algorithm",
       *SIAM Journal on Computing*, vol. 6, no. 1, pp. 151–154, 1977.
       https://doi.org/10.1137/0206011

    See Also
    --------
    max_min_tree_partition

    Examples
    --------
    Partition a path graph of 6 nodes into 3 equal parts (all weights default
    to 1):

    >>> G = nx.path_graph(6)
    >>> parts = nx.tree.min_max_tree_partition(G, 3)
    >>> sorted(sorted(p) for p in parts)
    [[0, 1], [2, 3], [4, 5]]

    With explicit vertex weights:

    >>> G = nx.path_graph(4)
    >>> nx.set_node_attributes(G, {0: 3, 1: 1, 2: 1, 3: 3}, "weight")
    >>> parts = nx.tree.min_max_tree_partition(G, 2, node_weight="weight")
    >>> sorted(sorted(p) for p in parts)
    [[0, 1], [2, 3]]
    """
    spec = _resolve_weight_function(weight_function, node_weight, edge_weight)
    _validate_partition_args(T, q, spec)
    node_w, edge_w = _weight_maps(T, spec)
    integral = _weights_integral(node_w, edge_w)
    try:
        partition = _binary_search_minmax(T, q, node_w, edge_w, integral)
    except OverflowError as err:
        raise nx.NetworkXError(
            "cannot mix float weights with integer weights beyond float "
            "range; use all-integer weights for exact arbitrary-precision "
            "arithmetic."
        ) from err
    return _sorted_components(partition, descending=True)


@nx.utils.not_implemented_for("directed")
@nx.utils.not_implemented_for("multigraph")
# See min_max_tree_partition: node_attrs/edge_attrs are deliberately
# over-declared for the vertex_count weight function.
@nx._dispatchable(graphs="T", node_attrs="node_weight", edge_attrs="edge_weight")
def max_min_tree_partition(
    T: nx.Graph,
    q: int,
    node_weight: str = "weight",
    edge_weight: str = "weight",
    *,
    weight_function: str = "vertex_weight_sum",
) -> list[frozenset]:
    r"""Partition a weighted tree into ``q`` components maximizing the minimum
    component weight.

    Removes ``q - 1`` edges from a tree ``T`` to produce ``q`` connected
    subtrees (components) such that the weight of the lightest component
    is as large as possible.

    By default, component weight is the sum of vertex weights, read from
    the node attribute ``weight`` (nodes missing the attribute are assigned
    weight 1).  To use a different notion of component weight, set
    *weight_function*.

    Parameters
    ----------
    T : NetworkX Graph
        An undirected tree (connected acyclic graph).

    q : int
        Number of components to partition ``T`` into. Must satisfy
        ``1 <= q <= len(T)``.

    node_weight : str, optional (default ``"weight"``)
        Node attribute key read by ``"vertex_weight_sum"`` and ``"mixed_sum"``.

    edge_weight : str, optional (default ``"weight"``)
        Edge attribute key read by ``"edge_weight_sum"`` and ``"mixed_sum"``.

    weight_function : str, optional (default ``"vertex_weight_sum"``)
        How component weight is defined.  One of:

        - ``"vertex_weight_sum"`` — sum of vertex weights (reads
          ``node_weight``; weights must be finite and > 0).
        - ``"edge_weight_sum"`` — sum of edge weights (reads
          ``edge_weight``; weights must be finite and >= 0).  Singleton
          components have weight 0.
        - ``"mixed_sum"`` — sum of vertex and edge weights (reads both;
          vertex weights > 0, edge weights >= 0, all finite).
        - ``"vertex_count"`` — number of vertices (reads neither).

    Returns
    -------
    partition : list of frozenset
        A list of ``q`` frozensets, each holding the node labels of one
        component.  The list is sorted in **ascending** order of component
        weight (lightest component first), because the lightest component
        is the quantity being maximized.

    Raises
    ------
    NetworkXNotImplemented
        If ``T`` is directed or a multigraph.

    NetworkXError
        If ``q`` is not an integer, ``q`` is not in ``[1, len(T)]``,
        ``weight_function`` is unrecognised, a node or edge weight is
        invalid for the selected weight function (non-numeric, non-finite,
        or out of range), or float weights are mixed with integer weights
        too large to represent as floats.

    NetworkXPointlessConcept
        If ``T`` has no nodes.

    NotATree
        If ``T`` is not a tree.

    Notes
    -----
    The algorithm bisects over threshold values combined with a greedy
    post-order DFS feasibility oracle.  At each candidate threshold λ, the
    oracle roots the tree and greedily cuts each child subtree whose
    accumulated weight reaches λ, creating a "heavy" component.

    The bisection runs over the grid of representable weight values and
    snaps to weights actually achieved by feasible probes, so there are no
    tolerance parameters.  When every weight is an integer the grid is the
    integers (Python's arbitrary-precision ``int``) and the result is exact
    at any magnitude; otherwise the grid is the IEEE-754 doubles and the
    result is exact with respect to floating-point summation of component
    weights.  With B feasibility probes (B = O(64) for float weights,
    B = O(log W) for integer weights of total magnitude W; a pigeonhole
    seed and candidate-verification probes usually finish far sooner),
    each costing O(n), the overall complexity is :math:`O(n \cdot B)`.

    References
    ----------
    .. [1] Y. Perl and S. R. Schach,
       "Max-Min Tree Partitioning",
       *Journal of the ACM*, vol. 28, no. 1, pp. 5–15, 1981.
       https://doi.org/10.1145/322234.322236

    .. [2] S. Kundu and J. Misra,
       "A linear tree partitioning algorithm",
       *SIAM Journal on Computing*, vol. 6, no. 1, pp. 151–154, 1977.
       https://doi.org/10.1137/0206011

    See Also
    --------
    min_max_tree_partition

    Examples
    --------
    Partition a path graph of 7 nodes into 2 parts (all weights default to 1).
    The lightest component comes first:

    >>> G = nx.path_graph(7)
    >>> parts = nx.tree.max_min_tree_partition(G, 2)
    >>> [len(p) for p in parts]
    [3, 4]

    With explicit vertex weights:

    >>> G = nx.path_graph(4)
    >>> nx.set_node_attributes(G, {0: 1, 1: 5, 2: 5, 3: 1}, "weight")
    >>> parts = nx.tree.max_min_tree_partition(G, 2, node_weight="weight")
    >>> sorted(sorted(p) for p in parts)
    [[0, 1], [2, 3]]
    """
    spec = _resolve_weight_function(weight_function, node_weight, edge_weight)
    _validate_partition_args(T, q, spec)
    node_w, edge_w = _weight_maps(T, spec)
    integral = _weights_integral(node_w, edge_w)
    try:
        partition = _binary_search_maxmin(T, q, node_w, edge_w, integral)
    except OverflowError as err:
        raise nx.NetworkXError(
            "cannot mix float weights with integer weights beyond float "
            "range; use all-integer weights for exact arbitrary-precision "
            "arithmetic."
        ) from err
    return _sorted_components(partition, descending=False)
