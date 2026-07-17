"""
Min-max and max-min tree partitioning algorithms.

Implements ``min_max_tree_partition`` (Kundu-Misra 1977) and
``max_min_tree_partition`` (Perl-Schach 1981): given a weighted tree and an
integer `q`, partition it into exactly `q` connected
subtrees minimizing the heaviest component (min-max) or maximizing the lightest
component (max-min).

Both algorithms support the same four built-in additive weight functions,
selected via the ``weight_function`` keyword:
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
- ``max_min_tree_partition``: O(N · B + N log N) — feasibility is a
  single post-order pass with no per-node sort, but the final merge-down
  of surplus parts sorts the winning probe's cuts once.
"""

from __future__ import annotations

import math
import numbers
import struct
from collections.abc import Callable, Hashable
from typing import Any, NamedTuple

import networkx as nx

__all__ = [
    "min_max_tree_partition",
    "max_min_tree_partition",
    "tree_partition_weights",
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
_WEIGHT_FUNCTIONS: dict[str, Callable[[str, str], _WeightSpec]] = {
    "vertex_weight_sum": lambda nw, _: _WeightSpec(nw, 1, None),
    "edge_weight_sum": lambda _, ew: _WeightSpec(None, 0, ew),
    "mixed_sum": lambda nw, ew: _WeightSpec(nw, 1, ew),
    "vertex_count": lambda _1, _2: _WeightSpec(None, 1, None),
}


def _coerce_weight(w: Any) -> int | float:
    """Coerce to Python int (exact, arbitrary precision) or float.

    ``int(w)`` matters for fixed-width integer types such as numpy's: they
    pass ``isinstance(w, numbers.Integral)`` but silently wrap on overflow
    if used in arithmetic directly.
    """
    return int(w) if isinstance(w, numbers.Integral) else float(w)


def _check_weight(w: Any, elem: Any, is_node: bool) -> None:
    """Shared weight validation: numeric, finite, node > 0 / edge >= 0.

    *elem* is the node label or ``(u, v)`` edge tuple, used only in error
    messages (formatted lazily so the happy path pays no repr cost).
    """

    def label():
        return f"Node {elem!r}" if is_node else f"Edge {elem!r}"

    if isinstance(w, bool) or not isinstance(w, numbers.Number):
        # Rejects numeric-looking strings (float("5") parses) and bools;
        # both are near-certain data errors, not weights.
        raise nx.NetworkXError(
            f"{label()} has non-numeric weight {w!r}; all weights must be numbers."
        )
    if not isinstance(w, numbers.Integral):
        # Integers are always finite (and may exceed float range, so must
        # not be converted); everything else must convert to a finite float.
        try:
            wf = float(w)
        except (TypeError, ValueError) as err:  # e.g. complex
            raise nx.NetworkXError(
                f"{label()} has non-numeric weight {w!r}; all weights must be numbers."
            ) from err
        except OverflowError as err:  # e.g. Fraction(10**400)
            raise nx.NetworkXError(
                f"{label()} has weight {w!r} too large to represent as a "
                "float; all non-integer weights must fit in a float."
            ) from err
        if not math.isfinite(wf):
            raise nx.NetworkXError(
                f"{label()} has non-finite weight {w!r}; all weights must be finite."
            )
        w = wf
    if is_node:
        if w <= 0:
            raise nx.NetworkXError(
                f"{label()} has weight {w!r}; all node weights must be > 0."
            )
    elif w < 0:
        raise nx.NetworkXError(
            f"{label()} has weight {w!r}; all edge weights must be >= 0."
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _root_at_leaf(T: nx.Graph) -> Hashable:
    """Return a leaf node of T (degree 1, or the only node if |V|=1)."""
    if len(T) == 1:
        return nx.utils.arbitrary_element(T)
    return next(v for v, d in T.degree() if d == 1)


def _rooted_setup(
    T: nx.Graph, edge_w: dict[tuple[Hashable, Hashable], int | float]
) -> tuple[
    Hashable,
    list[Hashable],
    dict[Hashable, list[Hashable]],
    dict[Hashable, int | float],
]:
    """Root T at a leaf; shared setup for both bisection solvers.

    Returns (root, post_order, children, ew_par) where post_order lists
    nodes leaves-first, children maps every node to its (possibly empty)
    child list, and ew_par maps each non-root node to the weight of the
    edge to its parent.
    """
    root = _root_at_leaf(T)
    post_order = list(nx.dfs_postorder_nodes(T, root))
    parent = nx.dfs_predecessors(T, root)
    succ = nx.dfs_successors(T, root)
    children = {v: succ.get(v, []) for v in T}
    ew_par = {v: edge_w[v, p] for v, p in parent.items()}
    return root, post_order, children, ew_par


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
    return all(isinstance(w, int) for w in node_w.values()) and all(
        isinstance(w, int) for w in edge_w.values()
    )


def _weights_as_floats(
    node_w: dict[Hashable, int | float],
    edge_w: dict[tuple[Hashable, Hashable], int | float],
) -> tuple[
    dict[Hashable, float],
    dict[tuple[Hashable, Hashable], float],
]:
    """Convert mixed int/float weight maps to all-float.

    Called only when not every weight is an integer.  The bisection then
    runs on the grid of IEEE-754 doubles, which contains every achievable
    component weight only if every individual weight is itself an exact
    double — so an integer weight that would round on conversion raises
    instead of silently yielding a partition the grid cannot certify.
    """

    def as_float(w):
        if isinstance(w, int):
            try:
                f = float(w)
            except OverflowError:
                f = None
            if f is None or f != w:
                raise nx.NetworkXError(
                    "cannot mix float weights with integer weights that are "
                    "not exactly representable as floats; use all-integer "
                    "weights for exact arbitrary-precision arithmetic."
                )
            return f
        return w

    return (
        {k: as_float(w) for k, w in node_w.items()},
        {k: as_float(w) for k, w in edge_w.items()},
    )


def _validate_partition_args(T: nx.Graph, q: int, spec: _WeightSpec) -> None:
    """Raise appropriate errors for invalid inputs."""
    if not nx.is_tree(T):
        raise nx.NotATree("input graph is not a tree")
    if isinstance(q, bool) or not isinstance(q, numbers.Integral):
        raise nx.NetworkXError(f"q must be an integer, got {q!r}.")
    n = len(T)
    if q < 1 or q > n:
        raise nx.NetworkXError(
            f"q must satisfy 1 <= q <= number of nodes ({n}), got q={q}."
        )
    if spec.node_attr is not None:
        for v in T.nodes:
            _check_weight(T.nodes[v].get(spec.node_attr, 1), v, True)
    if spec.edge_attr is not None:
        for u, v, edata in T.edges(data=True):
            _check_weight(edata.get(spec.edge_attr, 1), (u, v), False)


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------


def _components_from_cut_edges(
    T: nx.Graph, cut_edges: set[frozenset[Hashable]]
) -> list[list[Hashable]]:
    """Connected components of T after removing cut_edges.

    *cut_edges* is a set of ``frozenset({u, v})`` keys.
    """
    view = nx.restricted_view(T, [], [tuple(e) for e in cut_edges])
    return [list(c) for c in nx.connected_components(view)]


def _component_weight(
    T: nx.Graph,
    comp_vertices: list[Hashable],
    node_w: dict[Hashable, int | float],
    edge_w: dict[tuple[Hashable, Hashable], int | float],
) -> int | float:
    """Weight of a connected component: node weights + internal edge weights."""
    total = sum(node_w[v] for v in comp_vertices)
    for u, v in T.subgraph(comp_vertices).edges():
        total += edge_w[u, v]
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


def _trivial_partition(
    T: nx.Graph,
    q: int,
    node_w: dict[Hashable, int | float],
    edge_w: dict[tuple[Hashable, Hashable], int | float],
) -> list[tuple[frozenset[Hashable], int | float]] | None:
    """The q == 1 (whole tree) and q == n (all singletons)
    partitions, common to both solvers; None when 1 < q < n."""
    verts = list(T)
    if q == 1:
        return [(frozenset(verts), _component_weight(T, verts, node_w, edge_w))]
    if q == len(T):
        return [(frozenset([v]), node_w[v]) for v in verts]
    return None


def _add_cuts_to_reach_q(
    T: nx.Graph, cut_edges: set[frozenset[Hashable]], q: int
) -> set[frozenset[Hashable]]:
    """Add arbitrary internal edges as cuts until exactly `q` parts.

    Every tree edge is a bridge, so each non-cut edge added as a cut
    increases the component count by exactly 1.  To reach `q` parts we
    need `q - 1` total cuts; pick any non-cut edges to fill the deficit.

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
) -> list[tuple[frozenset[Hashable], int | float]]:
    """Reduce parts to exactly `q` for max-min by merging the lightest
    component with a neighbor, returning the final partition as (nodes, weight) pairs.

    Uses ``nx.utils.UnionFind`` over component indices; vertices are
    regrouped in a single pass at the end.  Sort keys are computed once
    from the initial component weights and are not refreshed after merges.
    This still preserves the max-min optimum because the lightest cut is
    always adjacent to the at-most-one sub-threshold component, which is
    absorbed into a heavy neighbor before any heavy-only cuts are
    considered; every resulting component therefore contains at least one
    heavy initial component (weight >= lambda*).  The reduction is
    optimal-valued though not necessarily optimal-shaped.
    """
    comps = _components_from_cut_edges(T, cut_edges)
    weights = [_component_weight(T, c, node_w, edge_w) for c in comps]
    if len(comps) <= q:
        return [(frozenset(c), w) for c, w in zip(comps, weights)]

    comp_of = {}
    for i, comp in enumerate(comps):
        for v in comp:
            comp_of[v] = i

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

    uf = nx.utils.UnionFind(range(len(comps)))
    weight_of = dict(enumerate(weights))
    n_comps = len(comps)
    for _, _, cut, ca, cb in cut_info:
        if n_comps <= q:
            break
        ra, rb = uf[ca], uf[cb]
        if ra == rb:
            continue
        a, b = tuple(cut)
        # Un-cutting reconnects the edge, so its weight rejoins the part.
        merged_w = weight_of.pop(ra) + weight_of.pop(rb) + edge_w[a, b]
        uf.union(ra, rb)
        weight_of[uf[ra]] = merged_w
        n_comps -= 1

    merged_verts = {}
    for i, comp in enumerate(comps):
        merged_verts.setdefault(uf[i], []).extend(comp)
    return [(frozenset(vs), weight_of[r]) for r, vs in merged_verts.items()]


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


_Cuts = list[tuple[Hashable, Hashable]]


def _bisect_threshold(
    probe: Callable[[int | float], tuple[bool, _Cuts | None, int | float | None]],
    bad: int | float,
    good: int | float,
    best_cuts: _Cuts,
    guess: int | float,
    minimize: bool,
    integral: bool,
) -> _Cuts:
    """Verified grid bisection shared by both solvers.

    *good* is the feasible bound — the upper bound when *minimize* is true
    (min-max), the lower bound otherwise (max-min) — and always equals the
    weight achieved by *best_cuts*; *bad* is the infeasible bound on the
    other side.  *probe* maps a threshold to ``(ok, cuts, achieved)``,
    where *achieved* is on the bisection grid whenever *ok*.  Every
    achievable component weight is on the grid, so when no grid value lies
    strictly between the bounds, *good* is the exact optimum; returns the
    cuts achieving it.

    Two probe-count optimizations preserve exactness because every probe
    determines which side of the optimum it lands on:

    - the seed *guess* is probed first when it lies strictly inside the
      bracket, so the search starts near the caller's estimate instead of
      the bit-space middle;
    - candidate verification: after each snap of *good* to an achieved
      weight, probe one grid step past it — if that is infeasible the
      candidate is optimal and the search ends immediately; alternated
      with bisection steps so the worst case stays O(grid bisection).
    """
    step = _grid_prev if minimize else _grid_next

    def bracket():
        return (bad, good) if minimize else (good, bad)

    lo, hi = bracket()
    verified = True
    if lo < guess < hi:
        ok, cuts, achieved = probe(guess)
        if ok:
            good, best_cuts, verified = achieved, cuts, False
        else:
            bad = guess
        lo, hi = bracket()
    while not _grid_adjacent(lo, hi, integral):
        verifying = not verified
        mid = step(good, integral) if verifying else _grid_mid(lo, hi, integral)
        ok, cuts, achieved = probe(mid)
        if ok:
            good, best_cuts, verified = achieved, cuts, verifying
        else:
            bad, verified = mid, True
        lo, hi = bracket()
    return best_cuts


# ---------------------------------------------------------------------------
# Bisection solvers
# ---------------------------------------------------------------------------


def _binary_search_minmax(
    T: nx.Graph,
    q: int,
    node_w: dict[Hashable, int | float],
    edge_w: dict[tuple[Hashable, Hashable], int | float],
    integral: bool,
) -> list[tuple[frozenset[Hashable], int | float]]:
    """Partition into `q` parts via exact grid bisection + greedy oracle.

    Bisects the grid of representable threshold values (see
    `_bisect_threshold` for the seeding and candidate-verification
    scheme), snapping the feasible upper bound to the weight actually
    achieved by each feasible probe; the final upper bound is the exact
    optimum (with respect to the oracle's summation order in the float
    case).
    """
    trivial = _trivial_partition(T, q, node_w, edge_w)
    if trivial is not None:
        return trivial

    root, post_order, children, ew_par = _rooted_setup(T, edge_w)

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
    # feasible upper bound.  All weights are ints (integral mode) or floats
    # (otherwise), so every achieved sum is already on the bisection grid.
    _, best_cuts, hi = probe(math.inf)

    # Seed near total/q: with node-additive weights some part
    # must weigh at least total/q (pigeonhole); with edge
    # weights, cut edges leave the total, so it is only a heuristic starting
    # probe.  Correctness is unaffected either way — every probe verifies
    # which side of the optimum it lands on.
    guess = -(-hi // q) if integral else hi / q
    best_cuts = _bisect_threshold(
        probe, lo, hi, best_cuts, guess, minimize=True, integral=integral
    )
    return cuts_to_partition(best_cuts)


def _binary_search_maxmin(
    T: nx.Graph,
    q: int,
    node_w: dict[Hashable, int | float],
    edge_w: dict[tuple[Hashable, Hashable], int | float],
    integral: bool,
) -> list[tuple[frozenset[Hashable], int | float]]:
    """Partition into `q` parts via exact grid bisection + greedy oracle.

    Mirror image of `_binary_search_minmax`, sharing `_bisect_threshold`:
    the feasible lower bound snaps up to the lightest heavy part achieved
    by each feasible probe, and the optimum is the final lower bound.
    """
    trivial = _trivial_partition(T, q, node_w, edge_w)
    if trivial is not None:
        return trivial

    root, post_order, children, ew_par = _rooted_setup(T, edge_w)

    def probe(lam):
        heavy_count, cuts, min_heavy, root_w = _feasible_maxmin(
            lam, post_order, children, node_w, ew_par, root
        )
        return heavy_count >= q, cuts, min_heavy, root_w

    # lam = 0 cuts every edge (all weights are nonnegative), so it is
    # feasible whenever q <= n; the lightest part it actually
    # produced is the feasible lower bound.  All weights are ints (integral
    # mode) or floats (otherwise), so every achieved sum is already on the
    # grid.
    _, best_cuts, lo, _ = probe(0)

    # No q >= 2 disjoint parts can each weigh as much as the
    # whole tree (unless the total is zero, in which case lo == hi already),
    # so the whole-tree weight — the unbounded probe's residual — is an
    # infeasible upper bound.
    _, _, _, hi = probe(math.inf)

    # Pigeonhole seed: the lightest part can weigh at most
    # total/q (part weights sum to at most the whole-tree
    # weight in every weight mode).
    guess = hi // q if integral else hi / q
    best_cuts = _bisect_threshold(
        lambda lam: probe(lam)[:3],
        hi,
        lo,
        best_cuts,
        guess,
        minimize=False,
        integral=integral,
    )
    return _reduce_to_q_parts(T, {frozenset(e) for e in best_cuts}, q, node_w, edge_w)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sorted_components(
    partition: list[tuple[frozenset[Hashable], float]], descending: bool
) -> list[frozenset[Hashable]]:
    """Sort (nodes, weight) pairs by weight and return just the node sets."""
    ordered = sorted(partition, key=lambda x: x[1], reverse=descending)
    return [nodes for nodes, _ in ordered]


def _resolve_weight_function(
    weight_function: str, node_weight: str, edge_weight: str
) -> _WeightSpec:
    """Map a weight-function name to its additive weight specification."""
    # None is a common "use defaults" idiom elsewhere in networkx; here it
    # would become the attribute name, match no attribute, and silently
    # degrade e.g. vertex_weight_sum to vertex_count — so reject non-strings.
    for name, value in (("node_weight", node_weight), ("edge_weight", edge_weight)):
        if not isinstance(value, str):
            raise nx.NetworkXError(
                f"{name} must be a string naming an attribute, got {value!r}."
            )
    try:
        make_spec = _WEIGHT_FUNCTIONS[weight_function]
    except (KeyError, TypeError):
        raise nx.NetworkXError(
            f"weight_function must be one of {sorted(_WEIGHT_FUNCTIONS)}, "
            f"got {weight_function!r}."
        ) from None
    return make_spec(node_weight, edge_weight)


def _tree_partition(
    T: nx.Graph,
    q: int,
    node_weight: str,
    edge_weight: str,
    weight_function: str,
    solver,
    descending: bool,
) -> list[frozenset[Hashable]]:
    """Shared driver behind both public entry points: resolve and validate,
    build weight maps, pick the bisection grid, solve, and sort."""
    spec = _resolve_weight_function(weight_function, node_weight, edge_weight)
    _validate_partition_args(T, q, spec)
    node_w, edge_w = _weight_maps(T, spec)
    integral = _weights_integral(node_w, edge_w)
    if not integral:
        node_w, edge_w = _weights_as_floats(node_w, edge_w)
    partition = solver(T, q, node_w, edge_w, integral)
    return _sorted_components(partition, descending)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@nx.utils.not_implemented_for("directed")
@nx.utils.not_implemented_for("multigraph")
# node_attrs/edge_attrs are over-declared for weight functions that read
# fewer attributes (e.g. "vertex_count" reads neither): the dispatch DSL
# cannot condition declarations on the weight_function argument, so with a
# backend active, graph conversion may materialize an attribute the
# selected weight_function ignores — benign for numeric values, though a
# backend could choke on non-numeric values in an ignored attribute that
# the reference implementation would never read.  Dict form, not
# node_attrs="node_weight": string-form node_attrs declares a
# missing-attribute default of None to backends, while this implementation
# (and string-form edge_attrs) defaults missing attributes to 1.
@nx._dispatchable(
    graphs="T", node_attrs={"node_weight": 1}, edge_attrs={"edge_weight": 1}
)
def min_max_tree_partition(
    T: nx.Graph,
    q: int,
    node_weight: str = "weight",
    edge_weight: str = "weight",
    *,
    weight_function: str = "vertex_weight_sum",
) -> list[frozenset[Hashable]]:
    r"""Partition a weighted tree into ``q`` components minimizing
    the maximum component weight.

    Removes ``q - 1`` edges from a tree ``T`` to produce
    ``q`` connected subtrees (components) such that the weight
    of the heaviest component is as small as possible.

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
        Nodes missing the attribute are assigned weight 1.

    edge_weight : str, optional (default ``"weight"``)
        Edge attribute key read by ``"edge_weight_sum"`` and ``"mixed_sum"``.
        Edges missing the attribute are assigned weight 1.

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
        A list of ``q`` frozensets, each holding the node
        labels of one component.  The list is sorted in **descending** order
        of component weight (heaviest component first), because the heaviest
        component is the quantity being minimized.

    Raises
    ------
    NetworkXNotImplemented
        If ``T`` is directed or a multigraph.

    NetworkXError
        If ``q`` is not an integer, ``q`` is
        not in ``[1, len(T)]``, ``weight_function`` is unrecognised,
        ``node_weight`` or
        ``edge_weight`` is not a string, a node or edge weight is invalid
        for the selected weight function (non-numeric, non-finite, or out
        of range), or float weights are mixed with integer weights that
        cannot be represented exactly as floats.

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
    at any magnitude; otherwise every weight is converted to a float
    (raising if an integer weight would round), the grid is the IEEE-754
    doubles, and the result is exact with respect to floating-point
    summation of component weights.  With B feasibility probes (B = O(64)
    for float weights,
    B = O(log W) for integer weights of total magnitude W; a pigeonhole
    seed and candidate-verification probes usually finish far sooner),
    each costing O(n log n) for the sort, the overall complexity is
    :math:`O(n \log n \cdot B)`.

    ``weight_function`` is a fixed set of names rather than an arbitrary
    callable because the greedy feasibility oracle and the exact grid
    bisection both rely on component weight being additive over node and
    edge weights; non-additive objectives (e.g. component diameter) would
    silently break optimality.

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
    return _tree_partition(
        T,
        q,
        node_weight,
        edge_weight,
        weight_function,
        _binary_search_minmax,
        descending=True,
    )


@nx.utils.not_implemented_for("directed")
@nx.utils.not_implemented_for("multigraph")
# See min_max_tree_partition: node_attrs/edge_attrs are deliberately
# over-declared for the vertex_count weight function and use the dict form
# to declare the missing-attribute default of 1.
@nx._dispatchable(
    graphs="T", node_attrs={"node_weight": 1}, edge_attrs={"edge_weight": 1}
)
def max_min_tree_partition(
    T: nx.Graph,
    q: int,
    node_weight: str = "weight",
    edge_weight: str = "weight",
    *,
    weight_function: str = "vertex_weight_sum",
) -> list[frozenset[Hashable]]:
    r"""Partition a weighted tree into ``q`` components maximizing
    the minimum component weight.

    Removes ``q - 1`` edges from a tree ``T`` to produce
    ``q`` connected subtrees (components) such that the weight
    of the lightest component is as large as possible.

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
        Nodes missing the attribute are assigned weight 1.

    edge_weight : str, optional (default ``"weight"``)
        Edge attribute key read by ``"edge_weight_sum"`` and ``"mixed_sum"``.
        Edges missing the attribute are assigned weight 1.

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
        A list of ``q`` frozensets, each holding the node
        labels of one component.  The list is sorted in **ascending** order
        of component weight (lightest component first), because the lightest
        component is the quantity being maximized.

    Raises
    ------
    NetworkXNotImplemented
        If ``T`` is directed or a multigraph.

    NetworkXError
        If ``q`` is not an integer, ``q`` is
        not in ``[1, len(T)]``, ``weight_function`` is unrecognised,
        ``node_weight`` or
        ``edge_weight`` is not a string, a node or edge weight is invalid
        for the selected weight function (non-numeric, non-finite, or out
        of range), or float weights are mixed with integer weights that
        cannot be represented exactly as floats.

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
    at any magnitude; otherwise every weight is converted to a float
    (raising if an integer weight would round), the grid is the IEEE-754
    doubles, and the result is exact with respect to floating-point
    summation of component weights.  With B feasibility probes (B = O(64)
    for float weights,
    B = O(log W) for integer weights of total magnitude W; a pigeonhole
    seed and candidate-verification probes usually finish far sooner),
    each costing O(n), plus a final reduction that sorts the winning
    probe's cuts once, the overall complexity is
    :math:`O(n \cdot B + n \log n)`.

    ``weight_function`` is a fixed set of names rather than an arbitrary
    callable because the greedy feasibility oracle and the exact grid
    bisection both rely on component weight being additive over node and
    edge weights; non-additive objectives (e.g. component diameter) would
    silently break optimality.

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
    return _tree_partition(
        T,
        q,
        node_weight,
        edge_weight,
        weight_function,
        _binary_search_maxmin,
        descending=False,
    )


@nx.utils.not_implemented_for("directed")
@nx.utils.not_implemented_for("multigraph")
@nx._dispatchable(
    graphs="T", node_attrs={"node_weight": 1}, edge_attrs={"edge_weight": 1}
)
def tree_partition_weights(
    T: nx.Graph,
    partition: list[frozenset[Hashable]],
    node_weight: str = "weight",
    edge_weight: str = "weight",
    *,
    weight_function: str = "vertex_weight_sum",
) -> list[int | float]:
    r"""Return the weight of each component in a tree partition.

    Given a partition produced by :func:`min_max_tree_partition` or
    :func:`max_min_tree_partition`, return the component weight for each
    part under the same weight function.

    Parameters
    ----------
    T : NetworkX Graph
        An undirected tree (connected acyclic graph).

    partition : list of frozenset
        A list of frozensets of node labels, as returned by
        :func:`min_max_tree_partition` or :func:`max_min_tree_partition`.

    node_weight : str, optional (default ``"weight"``)
        Node attribute key read by ``"vertex_weight_sum"`` and
        ``"mixed_sum"``.  Nodes missing the attribute are assigned weight 1.

    edge_weight : str, optional (default ``"weight"``)
        Edge attribute key read by ``"edge_weight_sum"`` and ``"mixed_sum"``.
        Edges missing the attribute are assigned weight 1.

    weight_function : str, optional (default ``"vertex_weight_sum"``)
        How component weight is defined.  Must match the ``weight_function``
        used to produce *partition*.  One of ``"vertex_weight_sum"``,
        ``"edge_weight_sum"``, ``"mixed_sum"``, or ``"vertex_count"``.

    Returns
    -------
    weights : list of int or float
        Component weights in the same order as *partition*.

    Raises
    ------
    NetworkXNotImplemented
        If ``T`` is directed or a multigraph.

    NetworkXError
        If ``weight_function`` is unrecognised, ``node_weight`` or
        ``edge_weight`` is not a string, or a node or edge weight is
        invalid for the selected weight function.

    See Also
    --------
    min_max_tree_partition
    max_min_tree_partition

    Examples
    --------
    >>> G = nx.path_graph(6)
    >>> parts = nx.tree.min_max_tree_partition(G, 3)
    >>> nx.tree.tree_partition_weights(G, parts)
    [2, 2, 2]
    """
    spec = _resolve_weight_function(weight_function, node_weight, edge_weight)
    if spec.node_attr is not None:
        for v in T.nodes:
            _check_weight(T.nodes[v].get(spec.node_attr, 1), v, True)
    if spec.edge_attr is not None:
        for u, v, edata in T.edges(data=True):
            _check_weight(edata.get(spec.edge_attr, 1), (u, v), False)
    node_w, edge_w = _weight_maps(T, spec)
    return [_component_weight(T, list(comp), node_w, edge_w) for comp in partition]
