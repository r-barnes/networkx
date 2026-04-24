"""
Algorithms for partitioning weighted trees into connected components.

Supports MIN-MAX and MAX-MIN objectives over pluggable weight functions
satisfying:

  (D) Decomposability: bounded state sigma(T') for each subtree T', with
      W(T') determined by sigma(T'), and sigma(T_parent ∪ T_child) computable
      from sigma(T_parent), sigma(T_child), and the connecting edge data.

  (G) Greedy optimality: "merge-if-possible" (min-max) or "cut-once-heavy"
      (max-min) in post-order DFS yields the optimal number of parts.

(D) + (G) hold for additive functions (sum of vertex weights, sum of edge
lengths, vertex count, or any linear combination).

Complexity
----------
Let N = |V(T)|, W = weight of whole tree, eps = tolerance.

- ``min_max_tree_partition``: O(N log N · log(W / eps)) — the feasibility
  oracle sorts children by residual weight at each node.
- ``max_min_tree_partition``: O(N log(W / eps)) — feasibility is a single
  post-order pass with no per-node sort.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Hashable

import networkx as nx

__all__ = [
    "min_max_tree_partition",
    "max_min_tree_partition",
]


# ---------------------------------------------------------------------------
# Weight function interface and built-in implementations
# ---------------------------------------------------------------------------


class WeightFunction(ABC):
    """Abstract base for a tree-partition weight function.

    Subclasses define how W(T') is computed incrementally via:

    - ``singleton(v, node_data)`` — state for the subtree containing only v.
    - ``merge(parent_state, child_state, edge_data)`` — state for the union of
      the parent's subtree and a child's subtree, connected by an edge whose
      attribute dict is *edge_data*.
    - ``weight(state)`` — the scalar weight of a state.

    Parameters *node_data* and *edge_data* are the NetworkX attribute dicts
    ``T.nodes[v]`` and ``T.edges[u, v]``.
    """

    @abstractmethod
    def singleton(self, v: Hashable, node_data: dict[str, Any]) -> Any:
        """Return the initial state for a single-vertex subtree."""

    @abstractmethod
    def merge(self, parent_state: Any, child_state: Any, edge_data: dict[str, Any]) -> Any:
        """Return the merged state when joining a parent's subtree with a
        child's subtree across the edge described by *edge_data*."""

    @abstractmethod
    def weight(self, state: Any) -> float:
        """Return the scalar weight of *state*."""

    def validate_node(self, v: Hashable, node_data: dict[str, Any]) -> None:
        """Raise `nx.NetworkXError` if *node_data* is invalid for this weight
        function.  Called once per node during input validation.

        The default implementation does nothing.  Override to enforce
        constraints (e.g. positive finite weights).
        """

    def validate_edge(self, u: Hashable, v: Hashable, edge_data: dict[str, Any]) -> None:
        """Raise `nx.NetworkXError` if *edge_data* is invalid for this weight
        function.  Called once per edge during input validation.

        The default implementation does nothing.
        """


class VertexWeightSum(WeightFunction):
    """W(T') = sum of vertex weights.

    Parameters
    ----------
    weight_attr : str
        Node attribute key for vertex weight.
    default : float
        Weight assigned to nodes missing the attribute.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.set_node_attributes(G, {0: 3, 1: 1, 2: 1, 3: 3}, "weight")
    >>> parts = nx.tree.min_max_tree_partition(
    ...     G, 2, weight_function="vertex_weight_sum"
    ... )
    >>> [w for _, w in parts]
    [4.0, 4.0]
    """

    def __init__(self, weight_attr: str = "weight", default: float = 1.0):
        self._attr = weight_attr
        self._default = default

    def singleton(self, v: Hashable, node_data: dict[str, Any]) -> float:
        return float(node_data.get(self._attr, self._default))

    def merge(self, a: float, b: float, edge_data: dict[str, Any]) -> float:
        return a + b

    def weight(self, s: float) -> float:
        return s

    def validate_node(self, v: Hashable, node_data: dict[str, Any]) -> None:
        w = node_data.get(self._attr, self._default)
        if not math.isfinite(w):
            raise nx.NetworkXError(
                f"Node {v!r} has non-finite weight {w!r}; all weights must be finite."
            )
        if w <= 0:
            raise nx.NetworkXError(
                f"Node {v!r} has weight {w!r}; all weights must be > 0."
            )


class EdgeWeightSum(WeightFunction):
    """W(T') = sum of edge weights.

    Singleton weight is 0 (a single vertex has no edges).  Edge weights
    must be ``>= 0`` (unlike node-weight functions, which require ``> 0``).
    Zero is allowed because a tree with all-zero edge weights still has a
    well-defined partitioning: every component weighs 0.

    Parameters
    ----------
    weight_attr : str
        Edge attribute key for edge weight.
    default : float
        Weight assigned to edges missing the attribute.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.set_edge_attributes(G, {(0, 1): 2, (1, 2): 3, (2, 3): 5}, "len")
    >>> parts = nx.tree.min_max_tree_partition(
    ...     G, 2, edge_weight="len", weight_function="edge_weight_sum"
    ... )
    >>> max(w for _, w in parts)
    5.0
    """

    def __init__(self, weight_attr: str = "weight", default: float = 1.0):
        self._attr = weight_attr
        self._default = default

    def singleton(self, v: Hashable, node_data: dict[str, Any]) -> float:
        return 0.0

    def merge(self, a: float, b: float, edge_data: dict[str, Any]) -> float:
        return a + b + float(edge_data.get(self._attr, self._default))

    def weight(self, s: float) -> float:
        return s

    def validate_edge(self, u: Hashable, v: Hashable, edge_data: dict[str, Any]) -> None:
        w = edge_data.get(self._attr, self._default)
        if not math.isfinite(w):
            raise nx.NetworkXError(
                f"Edge ({u!r}, {v!r}) has non-finite weight {w!r}; "
                "all weights must be finite."
            )
        if w < 0:
            raise nx.NetworkXError(
                f"Edge ({u!r}, {v!r}) has weight {w!r}; "
                "all edge weights must be >= 0."
            )


class MixedSum(WeightFunction):
    """W(T') = sum of vertex weights + sum of edge weights.

    Parameters
    ----------
    node_weight_attr : str
        Node attribute key for vertex weight.
    edge_weight_attr : str
        Edge attribute key for edge weight.
    node_default : float
        Default vertex weight.
    edge_default : float
        Default edge weight.
    """

    def __init__(
        self,
        node_weight_attr: str = "weight",
        edge_weight_attr: str = "weight",
        node_default: float = 1.0,
        edge_default: float = 1.0,
    ):
        self._nattr = node_weight_attr
        self._eattr = edge_weight_attr
        self._ndef = node_default
        self._edef = edge_default

    def singleton(self, v: Hashable, node_data: dict[str, Any]) -> float:
        return float(node_data.get(self._nattr, self._ndef))

    def merge(self, a: float, b: float, edge_data: dict[str, Any]) -> float:
        return a + b + float(edge_data.get(self._eattr, self._edef))

    def weight(self, s: float) -> float:
        return s

    def validate_node(self, v: Hashable, node_data: dict[str, Any]) -> None:
        w = node_data.get(self._nattr, self._ndef)
        if not math.isfinite(w):
            raise nx.NetworkXError(
                f"Node {v!r} has non-finite weight {w!r}; "
                "all weights must be finite."
            )
        if w <= 0:
            raise nx.NetworkXError(
                f"Node {v!r} has weight {w!r}; all node weights must be > 0."
            )

    def validate_edge(self, u: Hashable, v: Hashable, edge_data: dict[str, Any]) -> None:
        w = edge_data.get(self._eattr, self._edef)
        if not math.isfinite(w):
            raise nx.NetworkXError(
                f"Edge ({u!r}, {v!r}) has non-finite weight {w!r}; "
                "all weights must be finite."
            )
        if w < 0:
            raise nx.NetworkXError(
                f"Edge ({u!r}, {v!r}) has weight {w!r}; "
                "all edge weights must be >= 0."
            )


class VertexCount(WeightFunction):
    """W(T') = number of vertices.

    Examples
    --------
    >>> G = nx.path_graph(6)
    >>> parts = nx.tree.min_max_tree_partition(G, 3, weight_function="vertex_count")
    >>> [w for _, w in parts]
    [2.0, 2.0, 2.0]
    """

    def singleton(self, v: Hashable, node_data: dict[str, Any]) -> int:
        return 1

    def merge(self, a: int, b: int, edge_data: dict[str, Any]) -> int:
        return a + b

    def weight(self, s: int) -> float:
        return float(s)


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
    dict[Hashable, dict[str, Any] | None],
    dict[Hashable, list[Hashable]],
]:
    """Root tree at *root*.

    Returns (post_order, parent, edge_to_parent, children) where:
    - post_order: list of nodes in post-order (leaves first)
    - parent: dict mapping node -> parent (root -> None)
    - edge_to_parent: dict mapping node -> edge attribute dict to parent
    - children: dict mapping node -> list of child nodes
    """
    parent = {root: None}
    edge_to_parent = {root: None}
    children = {v: [] for v in T}
    post_order = []
    stack = [(root, iter(T.adj[root].items()))]
    visited = {root}
    while stack:
        v, it = stack[-1]
        found = False
        for u, edata in it:
            if u not in visited:
                visited.add(u)
                parent[u] = v
                edge_to_parent[u] = edata
                children[v].append(u)
                stack.append((u, iter(T.adj[u].items())))
                found = True
                break
        if not found:
            post_order.append(v)
            stack.pop()
    return post_order, parent, edge_to_parent, children


def _validate_partition_args(
    T: nx.Graph, q: int, wf: WeightFunction
) -> None:
    """Raise appropriate errors for invalid inputs."""
    if not nx.is_tree(T):
        raise nx.NotATree("input graph is not a tree")
    n = len(T)
    if q < 1 or q > n:
        raise nx.NetworkXError(
            f"q must satisfy 1 <= q <= number of nodes ({n}), got q={q}."
        )
    for v in T.nodes:
        wf.validate_node(v, T.nodes[v])
    for u, v, edata in T.edges(data=True):
        wf.validate_edge(u, v, edata)


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


def _component_weight_via_wf(
    T: nx.Graph, comp_vertices: list[Hashable], wf: WeightFunction
) -> float:
    """Compute W(T') for a connected component by DFS within it."""
    if not comp_vertices:
        return 0.0
    if len(comp_vertices) == 1:
        v = comp_vertices[0]
        return wf.weight(wf.singleton(v, T.nodes[v]))
    comp_set = set(comp_vertices)
    root = comp_vertices[0]

    post_order = []
    children_of = {v: [] for v in comp_vertices}
    edge_to_par = {root: None}
    visited = {root}
    stack = [(root, iter(T.adj[root].items()))]
    while stack:
        v, it = stack[-1]
        found = False
        for u, edata in it:
            if u in comp_set and u not in visited:
                visited.add(u)
                edge_to_par[u] = edata
                children_of[v].append(u)
                stack.append((u, iter(T.adj[u].items())))
                found = True
                break
        if not found:
            post_order.append(v)
            stack.pop()

    state = {}
    for v in post_order:
        sv = wf.singleton(v, T.nodes[v])
        for c in children_of[v]:
            sv = wf.merge(sv, state[c], edge_to_par[c])
        state[v] = sv
    return wf.weight(state[root])


def _total_weight_via_wf(T: nx.Graph, wf: WeightFunction) -> float:
    """Total tree weight via WeightFunction."""
    return _component_weight_via_wf(T, list(T), wf)


# ---------------------------------------------------------------------------
# Feasibility tests
# ---------------------------------------------------------------------------


def _feasible_minmax(
    T: nx.Graph,
    wf: WeightFunction,
    lam: float,
    post_order: list[Hashable],
    children: dict[Hashable, list[Hashable]],
    edge_to_parent: dict[Hashable, dict[str, Any] | None],
    root: Hashable,
    eps: float = 1e-12,
) -> tuple[bool, set[frozenset[Hashable]] | None, float | None]:
    """Greedy feasibility test for min-max at threshold *lam*.

    For each vertex v in post-order, start from singleton(v) and greedily
    merge each child's residual whenever the merged weight stays <= lam + eps.
    Otherwise cut the edge (the child becomes a finished part with weight
    <= lam).

    Children are processed in ascending residual-weight order.  For the
    additive weight functions shipped here (vertex, edge, and mixed sums)
    this is the Kundu-Misra greedy: it yields the minimum number of cuts
    at threshold lam and a residual that is monotone non-increasing in
    lam, which is the property the outer binary search needs.

    Returns (ok, cut_edges, root_weight):
      ok          -- False iff some singleton weight > lam (infeasible).
      cut_edges   -- set of frozenset({child, parent}) for cut edges.
      root_weight -- W of root's final residual.
    """
    state = {}
    cuts = set()
    for v in post_order:
        sv = wf.singleton(v, T.nodes[v])
        w_sv = wf.weight(sv)
        if w_sv > lam + eps:
            return False, None, None
        # Note: for additive weight functions this does ~2x the strictly needed
        # merge calls (once in the sort key, once in the loop).  The redundancy
        # is intentional — predicting candidate weights from a cached delta
        # would silently break any non-additive subclass (e.g. diameter, where
        # merge is super-additive and a delta computed against the singleton
        # underestimates the real post-merge weight).
        kids = sorted(
            children[v],
            key=lambda c: wf.weight(wf.merge(sv, state[c], edge_to_parent[c])),
        )
        for c in kids:
            cand = wf.merge(sv, state[c], edge_to_parent[c])
            if wf.weight(cand) <= lam + eps:
                sv = cand
            else:
                cuts.add(frozenset((c, v)))
        state[v] = sv
    return True, cuts, wf.weight(state[root])


def _feasible_maxmin(
    T: nx.Graph,
    wf: WeightFunction,
    lam: float,
    post_order: list[Hashable],
    children: dict[Hashable, list[Hashable]],
    edge_to_parent: dict[Hashable, dict[str, Any] | None],
    root: Hashable,
    eps: float = 1e-12,
) -> tuple[int, set[frozenset[Hashable]], float, bool]:
    """Greedy feasibility test for max-min at threshold *lam*.

    For each vertex v in post-order, accumulate children's residuals; as soon
    as a child's residual weight is >= lam, cut the edge (that child becomes
    a finished "heavy" part with weight >= lam).  Otherwise merge it in.

    Returns (heavy_count, cut_edges, root_weight, root_heavy).
    """
    state = {}
    cuts = set()
    for v in post_order:
        sv = wf.singleton(v, T.nodes[v])
        for c in children[v]:
            if wf.weight(state[c]) >= lam - eps:
                cuts.add(frozenset((c, v)))
            else:
                sv = wf.merge(sv, state[c], edge_to_parent[c])
        state[v] = sv
    root_w = wf.weight(state[root])
    root_heavy = root_w >= lam - eps
    heavy_count = len(cuts) + (1 if root_heavy else 0)
    return heavy_count, cuts, root_w, root_heavy


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
        if need == 0:
            break
        key = frozenset((u, v))
        if key not in cut_edges:
            cut_edges.add(key)
            need -= 1
    return cut_edges


def _reduce_cuts_to_reach_q(
    T: nx.Graph,
    cut_edges: set[frozenset[Hashable]],
    q: int,
    wf: WeightFunction,
) -> set[frozenset[Hashable]]:
    """Reduce parts to exactly q for max-min by merging the lightest component
    with a neighbor.

    Uses union-find for O(n log n) performance instead of O(n^2) recomputation.
    """
    comps = _components_from_cut_edges(T, cut_edges)
    if len(comps) <= q:
        return cut_edges

    comp_of = {}
    comp_weights = []
    for i, comp in enumerate(comps):
        w = _component_weight_via_wf(T, comp, wf)
        comp_weights.append(w)
        for v in comp:
            comp_of[v] = i

    uf_parent = list(range(len(comps)))
    uf_weight = list(comp_weights)

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
        min_w = min(comp_weights[ca], comp_weights[cb])
        cut_info.append((min_w, edge_index[cut], cut, ca, cb))
    cut_info.sort()

    removed = set()
    n_comps = len(comps)
    for _, _, cut, ca, cb in cut_info:
        if n_comps <= q:
            break
        ra, rb = find(ca), find(cb)
        if ra == rb:
            continue
        removed.add(cut)
        if uf_weight[ra] > uf_weight[rb]:
            ra, rb = rb, ra
        uf_parent[ra] = rb
        uf_weight[rb] += uf_weight[ra]
        n_comps -= 1

    return cut_edges - removed


# ---------------------------------------------------------------------------
# Binary-search solvers
# ---------------------------------------------------------------------------


def _binary_search_minmax(
    T: nx.Graph,
    q: int,
    wf: WeightFunction,
    tol: float = 1e-9,
    max_iters: int = 200,
) -> list[tuple[frozenset, float]]:
    """Min-max q-partition via binary search + greedy feasibility oracle."""
    n = len(T)
    verts = list(T)

    if q == 1:
        value = _total_weight_via_wf(T, wf)
        return [(frozenset(verts), value)]

    if q == n:
        comps = [[v] for v in verts]
        return [
            (frozenset(c), _component_weight_via_wf(T, c, wf)) for c in comps
        ]

    root = _root_at_leaf(T)
    post_order, _, edge_to_parent, children = _root_tree(T, root)

    lo = max(wf.weight(wf.singleton(v, T.nodes[v])) for v in verts)
    hi = _total_weight_via_wf(T, wf)

    if hi <= lo:
        # Degenerate threshold (e.g. all-zero edge weights): any q-partition
        # is optimal since every component weight is <= hi.
        cuts_ud = _add_cuts_to_reach_q(T, set(), q)
        comps = _components_from_cut_edges(T, cuts_ud)
        return [
            (frozenset(c), _component_weight_via_wf(T, c, wf)) for c in comps
        ]

    # Check if lo itself is feasible.
    ok_lo, cuts_lo, _ = _feasible_minmax(
        T, wf, lo, post_order, children, edge_to_parent, root
    )
    if ok_lo and len(cuts_lo) + 1 <= q:
        cuts_ud = _add_cuts_to_reach_q(T, set(cuts_lo), q)
        comps = _components_from_cut_edges(T, cuts_ud)
        return [
            (frozenset(c), _component_weight_via_wf(T, c, wf)) for c in comps
        ]

    # Binary search for smallest feasible lambda in (lo, hi].
    lo_b, hi_b = lo, hi
    best_cuts = None
    abs_tol = max(tol, tol * abs(hi))
    for _ in range(max_iters):
        if hi_b - lo_b <= abs_tol:
            break
        mid = 0.5 * (lo_b + hi_b)
        ok, cuts, _ = _feasible_minmax(
            T, wf, mid, post_order, children, edge_to_parent, root
        )
        if ok and len(cuts) + 1 <= q:
            hi_b = mid
            best_cuts = cuts
        else:
            lo_b = mid

    ok, cuts_final, _ = _feasible_minmax(
        T, wf, hi_b, post_order, children, edge_to_parent, root
    )
    if ok and len(cuts_final) + 1 <= q:
        best_cuts = cuts_final

    if best_cuts is None:
        best_cuts = set()

    cuts_ud = _add_cuts_to_reach_q(T, set(best_cuts), q)
    comps = _components_from_cut_edges(T, cuts_ud)
    return [
        (frozenset(c), _component_weight_via_wf(T, c, wf)) for c in comps
    ]


def _binary_search_maxmin(
    T: nx.Graph,
    q: int,
    wf: WeightFunction,
    tol: float = 1e-9,
    max_iters: int = 200,
) -> list[tuple[frozenset, float]]:
    """Max-min q-partition via binary search + greedy feasibility oracle."""
    n = len(T)
    verts = list(T)

    if q == 1:
        value = _total_weight_via_wf(T, wf)
        return [(frozenset(verts), value)]

    if q == n:
        comps = [[v] for v in verts]
        return [
            (frozenset(c), _component_weight_via_wf(T, c, wf)) for c in comps
        ]

    root = _root_at_leaf(T)
    post_order, _, edge_to_parent, children = _root_tree(T, root)

    lo = 0.0
    hi = _total_weight_via_wf(T, wf)

    abs_tol = max(tol, tol * abs(hi))
    lo_b, hi_b = lo, hi
    best_cuts = None
    for _ in range(max_iters):
        if hi_b - lo_b <= abs_tol:
            break
        mid = 0.5 * (lo_b + hi_b)
        heavy_count, cuts, _, _ = _feasible_maxmin(
            T, wf, mid, post_order, children, edge_to_parent, root
        )
        if heavy_count >= q:
            lo_b = mid
            best_cuts = cuts
        else:
            hi_b = mid

    heavy_count, cuts_final, _, _ = _feasible_maxmin(
        T, wf, lo_b, post_order, children, edge_to_parent, root
    )
    if heavy_count >= q:
        best_cuts = cuts_final

    if best_cuts is None:
        # Degenerate case (e.g. all-zero weights): no threshold separated q
        # heavy parts; produce q arbitrary components.
        cuts_ud = _add_cuts_to_reach_q(T, set(), q)
        comps = _components_from_cut_edges(T, cuts_ud)
        return [
            (frozenset(c), _component_weight_via_wf(T, c, wf)) for c in comps
        ]

    cuts_ud = _reduce_cuts_to_reach_q(T, set(best_cuts), q, wf)
    comps = _components_from_cut_edges(T, cuts_ud)
    return [
        (frozenset(c), _component_weight_via_wf(T, c, wf)) for c in comps
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sort_partition(
    partition: list[tuple[frozenset, float]], descending: bool
) -> list[tuple[frozenset, float]]:
    return sorted(partition, key=lambda x: x[1], reverse=descending)


_WEIGHT_FUNCTION_NAMES = {
    "vertex_weight_sum",
    "edge_weight_sum",
    "mixed_sum",
    "vertex_count",
}


def _resolve_weight_function(
    weight_function: str, node_weight: str, edge_weight: str
) -> WeightFunction:
    """Map a weight-function name string to a WeightFunction instance."""
    if weight_function == "vertex_weight_sum":
        return VertexWeightSum(node_weight)
    if weight_function == "edge_weight_sum":
        return EdgeWeightSum(edge_weight)
    if weight_function == "mixed_sum":
        return MixedSum(node_weight, edge_weight)
    if weight_function == "vertex_count":
        return VertexCount()
    raise nx.NetworkXError(
        f"weight_function must be one of {sorted(_WEIGHT_FUNCTION_NAMES)}, "
        f"got {weight_function!r}."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@nx.utils.not_implemented_for("directed")
@nx.utils.not_implemented_for("multigraph")
@nx._dispatchable(node_attrs="node_weight", edge_attrs="edge_weight")
def min_max_tree_partition(
    T: nx.Graph,
    q: int,
    node_weight: str = "weight",
    edge_weight: str = "weight",
    *,
    weight_function: str = "vertex_weight_sum",
) -> list[tuple[frozenset, float]]:
    """Partition a weighted tree into ``q`` components minimizing the maximum
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
    partition : list of (frozenset, float)
        Each element is ``(nodes, w)`` where ``nodes`` is a ``frozenset`` of
        node labels in that component and ``w`` is the component weight.
        The list is sorted in **descending** order of weight (heaviest
        component first), because the heaviest component is the quantity
        being minimized.

    Raises
    ------
    NetworkXNotImplemented
        If ``T`` is directed or a multigraph.

    NetworkXError
        If ``q`` is not in ``[1, len(T)]``, or ``weight_function`` is
        unrecognised.

    NotATree
        If ``T`` is not a tree.

    Notes
    -----
    The algorithm uses binary search over threshold values combined with a
    greedy post-order DFS feasibility oracle.  At each candidate threshold
    λ, the oracle roots the tree and greedily merges children in ascending
    residual-weight order, cutting an edge whenever merging would exceed λ.
    The binary search converges in O(log(W/ε)) iterations, each costing
    O(n log n) for the sort, giving overall complexity
    :math:`O(n \\log n \\cdot \\log(W / \\varepsilon))`.

    References
    ----------
    .. [1] S. Kundu and J. Misra,
       "A linear tree partitioning algorithm",
       *SIAM Journal on Computing*, vol. 6, no. 1, pp. 151–154, 1977.

    See Also
    --------
    max_min_tree_partition

    Examples
    --------
    Partition a path graph of 6 nodes into 3 equal parts (all weights default
    to 1):

    >>> G = nx.path_graph(6)
    >>> parts = nx.tree.min_max_tree_partition(G, 3)
    >>> [w for _, w in parts]
    [2.0, 2.0, 2.0]

    With explicit vertex weights:

    >>> G = nx.path_graph(4)
    >>> nx.set_node_attributes(G, {0: 3, 1: 1, 2: 1, 3: 3}, "weight")
    >>> parts = nx.tree.min_max_tree_partition(G, 2, node_weight="weight")
    >>> [w for _, w in parts]
    [4.0, 4.0]
    """
    wf = _resolve_weight_function(weight_function, node_weight, edge_weight)
    _validate_partition_args(T, q, wf)
    return _sort_partition(_binary_search_minmax(T, q, wf), descending=True)


@nx.utils.not_implemented_for("directed")
@nx.utils.not_implemented_for("multigraph")
@nx._dispatchable(node_attrs="node_weight", edge_attrs="edge_weight")
def max_min_tree_partition(
    T: nx.Graph,
    q: int,
    node_weight: str = "weight",
    edge_weight: str = "weight",
    *,
    weight_function: str = "vertex_weight_sum",
) -> list[tuple[frozenset, float]]:
    """Partition a weighted tree into ``q`` components maximizing the minimum
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
    partition : list of (frozenset, float)
        Each element is ``(nodes, w)`` where ``nodes`` is a ``frozenset`` of
        node labels in that component and ``w`` is the component weight.
        The list is sorted in **ascending** order of weight (lightest
        component first), because the lightest component is the quantity
        being maximized.

    Raises
    ------
    NetworkXNotImplemented
        If ``T`` is directed or a multigraph.

    NetworkXError
        If ``q`` is not in ``[1, len(T)]``, or ``weight_function`` is
        unrecognised.

    NotATree
        If ``T`` is not a tree.

    Notes
    -----
    The algorithm uses binary search over threshold values combined with a
    greedy post-order DFS feasibility oracle.  At each candidate threshold
    λ, the oracle roots the tree and greedily cuts each child subtree whose
    accumulated weight reaches λ, creating a "heavy" component.  The binary
    search converges in O(log(W/ε)) iterations, each costing O(n), giving
    overall complexity :math:`O(n \\log(W / \\varepsilon))`.

    References
    ----------
    .. [1] Y. Perl and S. R. Schach,
       "Max-Min Tree Partitioning",
       *Journal of the ACM*, vol. 28, no. 1, pp. 5–15, 1981.
       https://doi.org/10.1145/322234.322236

    .. [2] S. Kundu and J. Misra,
       "A linear tree partitioning algorithm",
       *SIAM Journal on Computing*, vol. 6, no. 1, pp. 151–154, 1977.

    See Also
    --------
    min_max_tree_partition

    Examples
    --------
    Partition a path graph of 7 nodes into 2 parts (all weights default to 1):

    >>> G = nx.path_graph(7)
    >>> parts = nx.tree.max_min_tree_partition(G, 2)
    >>> [w for _, w in parts]
    [3.0, 4.0]

    With explicit vertex weights:

    >>> G = nx.path_graph(4)
    >>> nx.set_node_attributes(G, {0: 1, 1: 5, 2: 5, 3: 1}, "weight")
    >>> parts = nx.tree.max_min_tree_partition(G, 2, node_weight="weight")
    >>> [w for _, w in parts]
    [6.0, 6.0]
    """
    wf = _resolve_weight_function(weight_function, node_weight, edge_weight)
    _validate_partition_args(T, q, wf)
    return _sort_partition(_binary_search_maxmin(T, q, wf), descending=False)
