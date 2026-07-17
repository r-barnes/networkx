"""
Tests for min-max and max-min tree partitioning algorithms.

Covers:
- Analytical cases: path graphs, star graphs, balanced binary trees, caterpillar trees,
  spider graphs, chains of stars, broom graphs, comb graphs, binomial trees, and
  weighted paths with a single heavy node. These all have closed-form optimal values,
  so it's easy for us to check that the algorithm returns the correct value and a valid
  partition, even for large trees.
- Edge cases: q=1, q=n, single-node trees, two-node trees.
- Random sampling: random labeled trees compared against a brute-force reference
  to verify consistency of the public API for both min-max and max-min problems.
- Bound-sanity: lower bounds on OPT_minmax (average, max-node-weight,
  bottleneck-node) and upper bounds on OPT_maxmin (average, heavy-node) bracket
  the algorithm's output on large trees where brute force is not tractable.
- Weight functions: randomized brute-force validation for all four built-in
  weight functions (vertex_weight_sum, edge_weight_sum, mixed_sum,
  vertex_count).
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Hashable
from itertools import combinations

import pytest

import networkx as nx
from networkx.algorithms.tree.minmax_partitioning import (
    max_min_tree_partition,
    min_max_tree_partition,
    tree_partition_weights,
)


# ---------------------------------------------------------------------------
# Weight-annotated wrappers around the public API
# ---------------------------------------------------------------------------
#
# The public functions return bare partitions (lists of frozensets).  Most
# tests assert on component weights, so these wrappers call the public API
# and pair each component with its independently recomputed weight, giving
# the (nodes, weight) shape the assertion helpers below consume.


def _attach_weights(
    T: nx.Graph,
    parts: list[frozenset],
    node_weight: str,
    edge_weight: str,
    weight_function: str,
) -> list[tuple[frozenset, float]]:
    weights = tree_partition_weights(
        T, parts, node_weight, edge_weight, weight_function=weight_function
    )
    return list(zip(parts, weights))


def _minmax(
    T: nx.Graph,
    q: int,
    node_weight: str = "weight",
    edge_weight: str = "weight",
    *,
    weight_function: str = "vertex_weight_sum",
) -> list[tuple[frozenset, float]]:
    parts = min_max_tree_partition(
        T,
        q,
        node_weight=node_weight,
        edge_weight=edge_weight,
        weight_function=weight_function,
    )
    return _attach_weights(T, parts, node_weight, edge_weight, weight_function)


def _maxmin(
    T: nx.Graph,
    q: int,
    node_weight: str = "weight",
    edge_weight: str = "weight",
    *,
    weight_function: str = "vertex_weight_sum",
) -> list[tuple[frozenset, float]]:
    parts = max_min_tree_partition(
        T,
        q,
        node_weight=node_weight,
        edge_weight=edge_weight,
        weight_function=weight_function,
    )
    return _attach_weights(T, parts, node_weight, edge_weight, weight_function)


# ---------------------------------------------------------------------------
# Test-only brute-force oracle
# ---------------------------------------------------------------------------


def _brute_force_partition(
    T: nx.Graph,
    q: int,
    maximize_min: bool,
    *,
    weight_function: str = "vertex_weight_sum",
    node_weight: str = "weight",
    edge_weight: str = "weight",
) -> list[tuple[frozenset, float]]:
    """Brute-force optimal tree partition by exhaustive edge-cut enumeration.

    Tries all C(n-1, k) ways to remove k = q-1 edges and keeps the partition
    that minimizes the maximum component weight (``maximize_min=False``) or
    maximizes the minimum component weight (``maximize_min=True``).  Used
    here as an independent oracle for cross-checking the binary-search
    solvers on small inputs (n <= 12); O(C(n-1, q-1)) — do not use in
    production.
    """
    k = q - 1
    edges = list(T.edges())

    if k == 0:
        [w] = tree_partition_weights(
            T, [frozenset(T.nodes())], node_weight, edge_weight, weight_function=weight_function
        )
        return [(frozenset(T.nodes()), w)]

    best_val = float("inf") if not maximize_min else float("-inf")
    best_partition = []

    for cut_indices in combinations(range(len(edges)), k):
        cut = [edges[i] for i in cut_indices]
        comps = list(nx.connected_components(nx.restricted_view(T, [], cut)))
        if len(comps) != q:
            continue
        labeled = [
            (
                frozenset(c),
                tree_partition_weights(
                    T, [frozenset(c)], node_weight, edge_weight, weight_function=weight_function
                )[0],
            )
            for c in comps
        ]
        val = (
            max(w for _, w in labeled)
            if not maximize_min
            else min(w for _, w in labeled)
        )
        improved = (not maximize_min and val < best_val) or (
            maximize_min and val > best_val
        )
        if improved:
            best_val = val
            best_partition = sorted(
                labeled, key=lambda x: x[1], reverse=not maximize_min
            )

    return best_partition


# ---------------------------------------------------------------------------
# Internal test helpers
# ---------------------------------------------------------------------------


def _max_weight(partition: list[tuple[frozenset, float]]) -> float:
    """Weight of the heaviest component in a partition."""
    return max(w for _, w in partition)


def _min_weight(partition: list[tuple[frozenset, float]]) -> float:
    """Weight of the lightest component in a partition."""
    return min(w for _, w in partition)


def _is_valid_partition(
    T: nx.Graph,
    partition: list[tuple[frozenset, float]],
    q: int,
    *,
    weight_function: str = "vertex_weight_sum",
    node_weight: str = "weight",
    edge_weight: str = "weight",
) -> bool:
    """Return True iff partition is a valid q-partition of T.

    Checks that:
    - There are exactly q components.
    - Components are pairwise disjoint and cover V(T).
    - Each component induces a connected subgraph of T.
    - Reported weights match expected weights.
    """
    if len(partition) != q:
        return False
    all_nodes = set(T.nodes())
    seen: set = set()
    for nodes, reported_w in partition:
        if nodes & seen:
            return False
        seen |= nodes
        sub = T.subgraph(nodes)
        if not nx.is_connected(sub):
            return False
        [expected_w] = tree_partition_weights(
            T, [nodes], node_weight, edge_weight, weight_function=weight_function
        )
        if not math.isclose(reported_w, expected_w):
            return False
    return seen == all_nodes


def _is_sorted_descending(partition: list[tuple[frozenset, float]]) -> bool:
    weights = [w for _, w in partition]
    return all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))


def _is_sorted_ascending(partition: list[tuple[frozenset, float]]) -> bool:
    weights = [w for _, w in partition]
    return all(weights[i] <= weights[i + 1] for i in range(len(weights) - 1))


def _bfs_parents_order(T: nx.Graph) -> tuple[object, dict, list]:
    """Root T at an arbitrary node via BFS; return (root, parent, order).

    ``parent`` maps each node to its BFS parent (None for the root) and
    ``order`` lists nodes root-first in BFS discovery order, so iterating
    ``reversed(order)`` visits children before parents.
    """
    root = next(iter(T.nodes))
    pairs = list(nx.bfs_predecessors(T, root))
    parent = {root: None, **dict(pairs)}
    order = [root, *(v for v, _ in pairs)]
    return root, parent, order


def _greedy_partition_bounds(
    T: nx.Graph, q: int, weight: str = "weight"
) -> tuple[float, float]:
    """O(n) greedy partition bounds via post-order DFS cut + leaf-peel fallback.

    Returns (max_piece, min_piece) for an exact q-partition satisfying:
      max_piece >= OPT_minmax   (upper bound for min-max)
      min_piece <= OPT_maxmin   (lower bound for max-min)
    """
    W_node = {v: T.nodes[v].get(weight, 1) for v in T.nodes}
    total = sum(W_node.values())
    threshold = total / q

    root, parent, bfs_order = _bfs_parents_order(T)

    subtree_w = dict(W_node)
    piece_weights = []
    cut_nodes: set = set()
    cuts = 0
    for v in reversed(bfs_order):
        p = parent[v]
        if p is None:
            continue
        if subtree_w[v] >= threshold and cuts < q - 1:
            piece_weights.append(subtree_w[v])
            cut_nodes.add(v)
            cuts += 1
        else:
            subtree_w[p] += subtree_w[v]

    root_weight = subtree_w[root]
    remaining_cuts = q - 1 - cuts

    if remaining_cuts > 0:
        has_uncut_child: dict = {v: 0 for v in bfs_order if v not in cut_nodes}
        for v in bfs_order:
            p = parent[v]
            if v not in cut_nodes and p is not None and p not in cut_nodes:
                has_uncut_child[p] = has_uncut_child.get(p, 0) + 1

        made = 0
        for v in reversed(bfs_order):
            if made == remaining_cuts:
                break
            if v in cut_nodes or v == root:
                continue
            if has_uncut_child.get(v, 0) == 0:
                piece_weights.append(W_node[v])
                root_weight -= W_node[v]
                cut_nodes.add(v)
                p = parent[v]
                if p is not None and p not in cut_nodes:
                    has_uncut_child[p] -= 1
                made += 1

    piece_weights.append(root_weight)
    return max(piece_weights), min(piece_weights)


# ---------------------------------------------------------------------------
# Graph builders for analytical families
# ---------------------------------------------------------------------------


def _spider_graph(k: int, L: int) -> nx.Graph:
    """Hub node 0 with k arms, each a path of L nodes (unit weights)."""
    G = nx.Graph()
    node = 1
    for _ in range(k):
        G.add_edge(0, node)
        for _ in range(L - 1):
            G.add_edge(node, node + 1)
            node += 1
        node += 1
    return G


def _chain_of_stars(num_hubs: int, leaves_per_hub: int) -> nx.Graph:
    """Hubs 0..num_hubs-1 in a path; each hub has leaves_per_hub pendant leaves."""
    G = nx.Graph()
    for i in range(num_hubs - 1):
        G.add_edge(i, i + 1)
    leaf_id = num_hubs
    for i in range(num_hubs):
        for _ in range(leaves_per_hub):
            G.add_edge(i, leaf_id)
            leaf_id += 1
    return G


def _broom_graph(s: int, a: int) -> nx.Graph:
    """Center node 0, s pendant leaves (1..s), arm of length a (s+1..s+a)."""
    G = nx.Graph()
    for i in range(1, s + 1):
        G.add_edge(0, i)
    if a > 0:
        G.add_edge(0, s + 1)
    for i in range(s + 1, s + a):
        G.add_edge(i, i + 1)
    return G


def _comb_graph(s: int, d: int) -> nx.Graph:
    """Spine 0..s-1; each spine node i has a pendant path of d nodes."""
    G = nx.path_graph(s)
    node = s
    for i in range(s):
        G.add_edge(i, node)
        for _ in range(d - 1):
            G.add_edge(node, node + 1)
            node += 1
        node += 1
    return G


def _caterpillar(spine: int, leaves_per_node: int = 1) -> nx.Graph:
    """Path spine 0..spine-1; each spine node has leaves_per_node pendant
    leaves (labeled spine, spine+1, ...)."""
    G = nx.path_graph(spine)
    next_node = spine
    for v in range(spine):
        for _ in range(leaves_per_node):
            G.add_edge(v, next_node)
            next_node += 1
    return G


# ---------------------------------------------------------------------------
# Tighter analytical bounds
# ---------------------------------------------------------------------------


def _bottleneck_node_lb_minmax(T: nx.Graph, q: int, weight: str = "weight") -> float:
    """Lower bound on OPT_minmax from the bottleneck-node argument."""
    W_node = {v: T.nodes[v].get(weight, 1) for v in T.nodes}

    root, parent, bfs_order = _bfs_parents_order(T)

    sub_w: dict = dict(W_node)
    for v in reversed(bfs_order):
        p = parent[v]
        if p is not None:
            sub_w[p] += sub_w[v]

    W_total = sub_w[root]
    best_lb = W_total / q

    for v in T.nodes:
        d = T.degree(v)
        remaining = d - (q - 1)
        if remaining <= 0:
            continue
        adj_weights = []
        for u in T.neighbors(v):
            if parent.get(u) == v:
                adj_weights.append(sub_w[u])
            else:
                adj_weights.append(W_total - sub_w[v])
        adj_weights.sort()
        lb = W_node[v] + sum(adj_weights[:remaining])
        best_lb = max(best_lb, lb)

    return best_lb


# ---------------------------------------------------------------------------


class TestValidation:
    def test_directed_raises(self):
        G = nx.DiGraph([(0, 1), (1, 2)])
        with pytest.raises(nx.NetworkXNotImplemented):
            _minmax(G, 2)
        with pytest.raises(nx.NetworkXNotImplemented):
            _maxmin(G, 2)

    def test_multigraph_raises(self):
        G = nx.MultiGraph([(0, 1), (1, 2)])
        with pytest.raises(nx.NetworkXNotImplemented):
            _minmax(G, 2)
        with pytest.raises(nx.NetworkXNotImplemented):
            _maxmin(G, 2)

    def test_dispatchable_graph_argument_is_T(self):
        """The dispatcher must register the graph parameter under its real
        name (T).  With the default ("G"), keyword calls and backend graph
        conversion both fail whenever any backend is installed."""
        assert min_max_tree_partition.graphs == {"T": 0}
        assert max_min_tree_partition.graphs == {"T": 0}

    def test_dispatchable_declares_missing_attr_default_1(self):
        """The implementation defaults missing node/edge weight attributes
        to 1.  String-form node_attrs resolves to a default of None in the
        dispatch machinery (unlike edge_attrs, which resolves to 1), so the
        dict form with an explicit default is required for native backends
        to reproduce the reference behavior on graphs with missing
        attributes."""
        assert min_max_tree_partition.node_attrs == {"node_weight": 1}
        assert max_min_tree_partition.node_attrs == {"node_weight": 1}
        assert min_max_tree_partition.edge_attrs == {"edge_weight": 1}
        assert max_min_tree_partition.edge_attrs == {"edge_weight": 1}

    def test_graph_passable_by_keyword(self):
        G = nx.path_graph(4)
        assert len(min_max_tree_partition(T=G, q=2)) == 2
        assert len(max_min_tree_partition(T=G, q=2)) == 2

    def test_not_a_tree_cycle(self):
        G = nx.cycle_graph(4)
        with pytest.raises(nx.NotATree):
            _minmax(G, 2)
        with pytest.raises(nx.NotATree):
            _maxmin(G, 2)

    def test_not_a_tree_disconnected(self):
        G = nx.path_graph(3)
        G.add_node(99)
        with pytest.raises(nx.NotATree):
            _minmax(G, 2)

    def test_empty_graph_raises_pointless_concept(self):
        """Documented contract: an empty graph raises
        NetworkXPointlessConcept (from nx.is_tree), not NotATree."""
        G = nx.Graph()
        with pytest.raises(nx.NetworkXPointlessConcept):
            _minmax(G, 1)
        with pytest.raises(nx.NetworkXPointlessConcept):
            _maxmin(G, 1)

    @pytest.mark.parametrize("q", [0, -1, 5])
    def test_invalid_q_path4(self, q):
        G = nx.path_graph(4)
        with pytest.raises(nx.NetworkXError):
            _minmax(G, q)
        with pytest.raises(nx.NetworkXError):
            _maxmin(G, q)

    @pytest.mark.parametrize("bad_q", [2.5, 1.0, "2", None])
    def test_non_integer_q_raises(self, bad_q):
        """Pre-fix, q=2.5 silently returned n singletons from min-max and
        2 parts from max-min instead of raising."""
        G = nx.path_graph(5)
        with pytest.raises(nx.NetworkXError, match="integer"):
            _minmax(G, bad_q)
        with pytest.raises(nx.NetworkXError, match="integer"):
            _maxmin(G, bad_q)

    @pytest.mark.parametrize("bad_q", [True, False])
    def test_bool_q_raises(self, bad_q):
        """bool passes isinstance(q, Integral); pre-fix q=True silently
        returned the trivial 1-part partition."""
        G = nx.path_graph(5)
        with pytest.raises(nx.NetworkXError, match="integer"):
            _minmax(G, bad_q)
        with pytest.raises(nx.NetworkXError, match="integer"):
            _maxmin(G, bad_q)

    @pytest.mark.parametrize("fn", [min_max_tree_partition, max_min_tree_partition])
    @pytest.mark.parametrize("kwarg", ["node_weight", "edge_weight"])
    def test_non_string_weight_param_raises(self, fn, kwarg):
        """Pre-fix, node_weight=None silently degraded vertex_weight_sum to
        vertex_count (None became the attribute name, matching no attribute,
        and validation was skipped), returning a weight-blind partition."""
        G = nx.path_graph(4)
        G.nodes[0]["weight"] = 100
        with pytest.raises(nx.NetworkXError, match="must be a string"):
            fn(G, 2, **{kwarg: None})
        with pytest.raises(nx.NetworkXError, match="must be a string"):
            fn(G, 2, **{kwarg: 5})

    @pytest.mark.parametrize("bad_weight", [-1, -0.5, 0])
    def test_negative_weight_raises(self, bad_weight):
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {2: bad_weight}, "weight")
        with pytest.raises(nx.NetworkXError, match="> 0"):
            _minmax(G, 2, node_weight="weight")
        with pytest.raises(nx.NetworkXError, match="> 0"):
            _maxmin(G, 2, node_weight="weight")

    @pytest.mark.parametrize("bad_weight", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_weight_raises(self, bad_weight):
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {1: bad_weight}, "weight")
        with pytest.raises(nx.NetworkXError, match="non-finite"):
            _minmax(G, 2, node_weight="weight")
        with pytest.raises(nx.NetworkXError, match="non-finite"):
            _maxmin(G, 2, node_weight="weight")

    @pytest.mark.parametrize("weight_function", ["vertex_weight_sum", "mixed_sum"])
    def test_non_numeric_node_weight_raises(self, weight_function):
        """Pre-fix, a non-numeric weight escaped as a raw TypeError from
        math.isfinite instead of the documented NetworkXError."""
        G = nx.path_graph(3)
        G.nodes[1]["weight"] = "heavy"
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _minmax(G, 2, weight_function=weight_function)
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _maxmin(G, 2, weight_function=weight_function)

    @pytest.mark.parametrize("weight_function", ["edge_weight_sum", "mixed_sum"])
    def test_non_numeric_edge_weight_raises(self, weight_function):
        G = nx.path_graph(3)
        G.edges[0, 1]["weight"] = "long"
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _minmax(G, 2, weight_function=weight_function)
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _maxmin(G, 2, weight_function=weight_function)

    def test_none_node_weight_raises(self):
        G = nx.path_graph(3)
        G.nodes[1]["weight"] = None
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _minmax(G, 2)

    @pytest.mark.parametrize("bad_weight", ["3", "1e3", True])
    def test_numeric_looking_node_weight_raises(self, bad_weight):
        """Pre-fix, numeric strings passed validation via float(w) and were
        silently used as weights instead of raising the documented error;
        bools are near-certain bugs and are likewise rejected."""
        G = nx.path_graph(4)
        G.nodes[1]["weight"] = bad_weight
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _minmax(G, 2)
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _maxmin(G, 2)

    @pytest.mark.parametrize("bad_weight", ["2", True])
    def test_numeric_looking_edge_weight_raises(self, bad_weight):
        G = nx.path_graph(4)
        G.edges[1, 2]["weight"] = bad_weight
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _minmax(G, 2, weight_function="edge_weight_sum")
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _maxmin(G, 2, weight_function="edge_weight_sum")

    @pytest.mark.parametrize("fn", [min_max_tree_partition, max_min_tree_partition])
    def test_huge_non_integral_weight_raises_networkx_error(self, fn):
        """Pre-fix, a Fraction weight beyond float range leaked a bare
        OverflowError from float() instead of the documented NetworkXError."""
        from fractions import Fraction

        G = nx.path_graph(3)
        G.nodes[1]["weight"] = Fraction(10**400, 3)
        with pytest.raises(nx.NetworkXError, match="too large"):
            fn(G, 2)

    def test_mixed_huge_int_and_float_weights_raise(self):
        """Integers beyond float range are exact in all-integer trees, but
        cannot be summed with float weights; that mix raises clearly
        instead of leaking an OverflowError."""
        G = nx.path_graph(3)
        nx.set_node_attributes(G, {0: 10**400, 1: 1.5, 2: 1.5}, "weight")
        with pytest.raises(nx.NetworkXError, match="exactly representable"):
            _minmax(G, 2)
        with pytest.raises(nx.NetworkXError, match="exactly representable"):
            _maxmin(G, 2)

    def test_mixed_float_and_inexact_int_weights_raise(self):
        """Integers above 2**53 that are not exact doubles fall off the
        float bisection grid.  Pre-fix this mix silently returned a
        partition that was suboptimal under the documented
        floating-point-summation objective (probe thresholds could not
        separate candidates within one float gap)."""
        G = nx.Graph([(0, 1), (0, 2)])
        nx.set_node_attributes(G, {0: 4.0, 1: 2**60 + 4, 2: 2**60 + 1}, "weight")
        with pytest.raises(nx.NetworkXError, match="exactly representable"):
            _minmax(G, 2)
        with pytest.raises(nx.NetworkXError, match="exactly representable"):
            _maxmin(G, 2)

    def test_mixed_float_and_exact_representable_int_ok(self):
        """Large integers that ARE exact doubles (e.g. 2**60) mix fine with
        floats: everything lands on the float grid, so the documented
        float-summation exactness holds."""
        G = nx.Graph([(0, 1), (0, 2)])
        nx.set_node_attributes(G, {0: 4.0, 1: 2**60, 2: 2**60}, "weight")
        for fn in [min_max_tree_partition, max_min_tree_partition]:
            parts = fn(G, 2)
            assert len(parts) == 2
            assert set().union(*parts) == set(G.nodes)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_node_q1(self):
        G = nx.Graph()
        G.add_node(0, weight=5)
        p = _minmax(G, 1)
        assert p == [(frozenset({0}), 5)]
        p = _maxmin(G, 1)
        assert p == [(frozenset({0}), 5)]

    def test_two_nodes_q1(self):
        G = nx.path_graph(2)
        nx.set_node_attributes(G, {0: 3, 1: 7}, "weight")
        p = _minmax(G, 1, node_weight="weight")
        assert len(p) == 1
        assert p[0][1] == 10

    def test_two_nodes_q2(self):
        G = nx.path_graph(2)
        nx.set_node_attributes(G, {0: 3, 1: 7}, "weight")
        p_mm = _minmax(G, 2, node_weight="weight")
        assert _max_weight(p_mm) == 7
        p_mx = _maxmin(G, 2, node_weight="weight")
        assert _min_weight(p_mx) == 3

    def test_q_equals_n_all_singletons(self):
        """With q = n every node is its own component."""
        G = nx.path_graph(4)
        p = _minmax(G, 4)
        assert _is_valid_partition(G, p, 4)
        assert _max_weight(p) == 1

    def test_q1_whole_tree(self):
        G = nx.balanced_tree(2, 3)
        p = _minmax(G, 1)
        assert len(p) == 1
        total = len(G)
        assert p[0][1] == total


# ---------------------------------------------------------------------------
# Path graphs — analytically tractable
# ---------------------------------------------------------------------------


class TestPathGraphs:
    @pytest.mark.parametrize(
        "n,q,expected_minmax,expected_maxmin",
        [
            (6, 2, 3, 3),
            (7, 2, 4, 3),
            (6, 3, 2, 2),
            (7, 3, 3, 2),
            (9, 3, 3, 3),
            (10, 4, 3, 2),
            (12, 4, 3, 3),
        ],
    )
    def test_path_unit_weights(self, n, q, expected_minmax, expected_maxmin):
        G = nx.path_graph(n)
        p_mm = _minmax(G, q)
        assert _is_valid_partition(G, p_mm, q)
        assert _is_sorted_descending(p_mm)
        assert _max_weight(p_mm) == expected_minmax

        p_mx = _maxmin(G, q)
        assert _is_valid_partition(G, p_mx, q)
        assert _is_sorted_ascending(p_mx)
        assert _min_weight(p_mx) == expected_maxmin

    def test_path_weighted_min_max(self):
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {0: 3, 1: 1, 2: 1, 3: 3}, "weight")
        p = _minmax(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p, 2)
        assert _max_weight(p) == 4

    def test_path_weighted_max_min(self):
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {0: 1, 1: 5, 2: 5, 3: 1}, "weight")
        p = _maxmin(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p, 2)
        assert _min_weight(p) == 6

    def test_path_weighted_asymmetric(self):
        G = nx.path_graph(5)
        nx.set_node_attributes(G, {0: 1, 1: 1, 2: 10, 3: 1, 4: 1}, "weight")
        p = _minmax(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p, 2)
        assert _max_weight(p) == 12


# ---------------------------------------------------------------------------
# Star graphs — analytically tractable
# ---------------------------------------------------------------------------


class TestStarGraphs:
    @pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
    def test_star_q2_unit_weights(self, n):
        G = nx.star_graph(n)
        p_mm = _minmax(G, 2)
        assert _is_valid_partition(G, p_mm, 2)
        assert _max_weight(p_mm) == n

        p_mx = _maxmin(G, 2)
        assert _is_valid_partition(G, p_mx, 2)
        assert _min_weight(p_mx) == 1

    def test_star_q_equals_n_plus_1(self):
        n = 4
        G = nx.star_graph(n)
        q = n + 1
        p = _minmax(G, q)
        assert _is_valid_partition(G, p, q)
        assert _max_weight(p) == 1

    def test_star_weighted_q2(self):
        G = nx.star_graph(3)
        nx.set_node_attributes(G, {0: 10, 1: 1, 2: 1, 3: 1}, "weight")
        p_mm = _minmax(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p_mm, 2)
        assert _max_weight(p_mm) == 12

        p_mx = _maxmin(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p_mx, 2)
        assert _min_weight(p_mx) == 1


# ---------------------------------------------------------------------------
# Balanced binary trees
# ---------------------------------------------------------------------------


class TestBalancedBinaryTrees:
    def test_height2_q2_unit(self):
        G = nx.balanced_tree(2, 2)
        p_mm = _minmax(G, 2)
        assert _is_valid_partition(G, p_mm, 2)
        assert _max_weight(p_mm) == 4

        p_mx = _maxmin(G, 2)
        assert _is_valid_partition(G, p_mx, 2)
        assert _min_weight(p_mx) == 3

    def test_height2_q4_unit(self):
        G = nx.balanced_tree(2, 2)
        p_mm = _minmax(G, 4)
        assert _is_valid_partition(G, p_mm, 4)
        assert _max_weight(p_mm) >= 2

    def test_height3_q2(self):
        G = nx.balanced_tree(2, 3)
        p_mm = _minmax(G, 2)
        assert _is_valid_partition(G, p_mm, 2)
        assert _max_weight(p_mm) == 8

        p_mx = _maxmin(G, 2)
        assert _is_valid_partition(G, p_mx, 2)
        assert _min_weight(p_mx) == 7

    def test_ternary_tree_height2_q2(self):
        G = nx.balanced_tree(3, 2)
        p_mm = _minmax(G, 2)
        assert _is_valid_partition(G, p_mm, 2)
        assert _max_weight(p_mm) == 9

        p_mx = _maxmin(G, 2)
        assert _is_valid_partition(G, p_mx, 2)
        assert _min_weight(p_mx) == 4


# ---------------------------------------------------------------------------
# Caterpillar trees
# ---------------------------------------------------------------------------


class TestCaterpillarTrees:
    def test_caterpillar_spine3_leaves1_q2(self):
        G = _caterpillar(3, 1)
        assert len(G) == 6
        p_mm = _minmax(G, 2)
        assert _is_valid_partition(G, p_mm, 2)
        assert _max_weight(p_mm) == 4

        p_mx = _maxmin(G, 2)
        assert _is_valid_partition(G, p_mx, 2)
        assert _min_weight(p_mx) == 2

    def test_caterpillar_spine4_leaves1_q3(self):
        G = _caterpillar(4, 1)
        assert len(G) == 8
        p_mm = _minmax(G, 3)
        assert _is_valid_partition(G, p_mm, 3)
        assert _max_weight(p_mm) == 4

        p_mx = _maxmin(G, 3)
        assert _is_valid_partition(G, p_mx, 3)
        assert _min_weight(p_mx) == 2

    def test_caterpillar_spine3_leaves2_q3(self):
        G = _caterpillar(3, 2)
        assert len(G) == 9
        p_mm = _minmax(G, 3)
        assert _is_valid_partition(G, p_mm, 3)
        assert _max_weight(p_mm) == 3

        p_mx = _maxmin(G, 3)
        assert _is_valid_partition(G, p_mx, 3)
        assert _min_weight(p_mx) == 3

    def test_caterpillar_spine4_leaves2_q4(self):
        G = _caterpillar(4, 2)
        assert len(G) == 12
        p_mm = _minmax(G, 4)
        assert _is_valid_partition(G, p_mm, 4)
        assert _max_weight(p_mm) == 3

        p_mx = _maxmin(G, 4)
        assert _is_valid_partition(G, p_mx, 4)
        assert _min_weight(p_mx) == 3


# ---------------------------------------------------------------------------
# Explicit weighted examples
# ---------------------------------------------------------------------------


class TestExplicitWeightedTrees:
    def test_y_shaped_tree(self):
        G = nx.Graph([(0, 1), (0, 2), (0, 3)])
        nx.set_node_attributes(G, {0: 4, 1: 2, 2: 2, 3: 2}, "weight")
        p_mm = _minmax(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p_mm, 2)
        assert _max_weight(p_mm) == 8

        p_mx = _maxmin(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p_mx, 2)
        assert _min_weight(p_mx) == 2

    def test_unequal_arms_y_tree_q3(self):
        G = nx.Graph([(0, 1), (0, 2), (0, 3)])
        nx.set_node_attributes(G, {0: 1, 1: 2, 2: 3, 3: 6}, "weight")
        p_mm = _minmax(G, 3, node_weight="weight")
        assert _is_valid_partition(G, p_mm, 3)
        assert _max_weight(p_mm) == 6

        p_mx = _maxmin(G, 3, node_weight="weight")
        assert _is_valid_partition(G, p_mx, 3)
        assert _min_weight(p_mx) == 3

    def test_missing_weight_attribute_defaults_to_1(self):
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {0: 3, 2: 3}, "weight")
        p = _minmax(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p, 2)
        assert _max_weight(p) == 4

    def test_custom_weight_attribute_name(self):
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {0: 1, 1: 1, 2: 1, 3: 1}, "cost")
        p = _minmax(G, 2, node_weight="cost")
        assert _is_valid_partition(G, p, 2, node_weight="cost")
        assert _max_weight(p) == 2


# ---------------------------------------------------------------------------
# Return-value structure tests
# ---------------------------------------------------------------------------


class TestReturnFormat:
    """The raw public API returns a bare partition: a list of frozensets
    sorted by component weight (no weights in the return value)."""

    def test_returns_list_of_frozensets(self):
        G = nx.path_graph(6)
        for fn in [min_max_tree_partition, max_min_tree_partition]:
            p = fn(G, 3)
            assert isinstance(p, list)
            assert all(isinstance(nodes, frozenset) for nodes in p)

    def test_min_max_sorted_descending_by_weight(self):
        G = nx.path_graph(9)
        nx.set_node_attributes(G, {v: v + 1 for v in G}, "weight")
        p = min_max_tree_partition(G, 3, node_weight="weight")
        weights = [sum(v + 1 for v in nodes) for nodes in p]
        assert weights == sorted(weights, reverse=True)

    def test_max_min_sorted_ascending_by_weight(self):
        G = nx.path_graph(9)
        nx.set_node_attributes(G, {v: v + 1 for v in G}, "weight")
        p = max_min_tree_partition(G, 3, node_weight="weight")
        weights = [sum(v + 1 for v in nodes) for nodes in p]
        assert weights == sorted(weights)

    def test_q1_returns_single_frozenset(self):
        G = nx.path_graph(5)
        assert min_max_tree_partition(G, 1) == [frozenset(G.nodes())]
        assert max_min_tree_partition(G, 1) == [frozenset(G.nodes())]


# ---------------------------------------------------------------------------
# Random sampling tests — brute-force self-consistency
# ---------------------------------------------------------------------------


class TestRandomSampling:
    """Generate random labeled trees and verify the public functions match the
    internal brute-force reference."""

    SEEDS = [0, 1, 5, 7, 9, 13, 17, 31, 42, 50, 77, 99, 100, 123, 200, 256]

    @pytest.mark.parametrize("seed", SEEDS)
    def test_random_min_max_matches_brute_force(self, seed):
        rng = random.Random(seed)
        n = rng.randint(3, 11)
        T = nx.random_labeled_tree(n, seed=seed)
        for v in T.nodes:
            T.nodes[v]["weight"] = rng.randint(1, 10)
        q = rng.randint(1, min(n, 4))

        p_public = _minmax(T, q, node_weight="weight")
        p_ref = _brute_force_partition(T, q, False)

        assert _is_valid_partition(T, p_public, q)
        assert _is_sorted_descending(p_public)
        assert _max_weight(p_public) == _max_weight(p_ref)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_random_max_min_matches_brute_force(self, seed):
        rng = random.Random(seed)
        n = rng.randint(3, 11)
        T = nx.random_labeled_tree(n, seed=seed)
        for v in T.nodes:
            T.nodes[v]["weight"] = rng.randint(1, 10)
        q = rng.randint(1, min(n, 4))

        p_public = _maxmin(T, q, node_weight="weight")
        p_ref = _brute_force_partition(T, q, True)

        assert _is_valid_partition(T, p_public, q)
        assert _is_sorted_ascending(p_public)
        assert _min_weight(p_public) == _min_weight(p_ref)

    @pytest.mark.parametrize("n,q", [(5, 2), (6, 3), (8, 4), (10, 3), (12, 4)])
    def test_random_trees_various_sizes(self, n, q):
        for seed in range(5):
            T = nx.random_labeled_tree(n, seed=seed)
            p_mm = _minmax(T, q)
            p_mx = _maxmin(T, q)
            assert _is_valid_partition(T, p_mm, q)
            assert _is_valid_partition(T, p_mx, q)
            assert _min_weight(p_mx) <= n / q
            assert _max_weight(p_mm) >= n / q


# ---------------------------------------------------------------------------
# Float-weight vs brute-force agreement (small random trees)
# ---------------------------------------------------------------------------


class TestFloatWeightsVsBruteForce:
    """Verify the binary-search algorithm agrees with brute-force on small
    random trees with float weights (integer weights are covered by
    TestRandomSampling)."""

    @pytest.mark.parametrize("seed", [0, 1, 5, 9, 17])
    def test_min_max_float_weights(self, seed):
        rng = random.Random(seed)
        n = rng.randint(3, 11)
        T = nx.random_labeled_tree(n, seed=seed)
        for v in T.nodes:
            T.nodes[v]["weight"] = round(rng.uniform(0.5, 5.0), 2)
        q = rng.randint(1, min(n, 4))

        p_default = _minmax(T, q, node_weight="weight")
        p_brute = _brute_force_partition(T, q, False)

        assert _is_valid_partition(T, p_default, q)
        assert math.isclose(_max_weight(p_default), _max_weight(p_brute), rel_tol=1e-9)

    @pytest.mark.parametrize("seed", [0, 1, 5, 9, 17])
    def test_max_min_float_weights(self, seed):
        rng = random.Random(seed)
        n = rng.randint(3, 11)
        T = nx.random_labeled_tree(n, seed=seed)
        for v in T.nodes:
            T.nodes[v]["weight"] = round(rng.uniform(0.5, 5.0), 2)
        q = rng.randint(1, min(n, 4))

        p_default = _maxmin(T, q, node_weight="weight")
        p_brute = _brute_force_partition(T, q, True)

        assert _is_valid_partition(T, p_default, q)
        assert math.isclose(_min_weight(p_default), _min_weight(p_brute), rel_tol=1e-9)


class TestLargeAnalytical:
    """Trees with known closed-form optimal values at arbitrary scale."""

    @pytest.mark.parametrize(
        "n,q",
        [
            (100, 5),
            (500, 2),
            (1000, 4),
            (5000, 3),
            (20000, 4),
        ],
    )
    def test_path_unit_weights_large(self, n, q):
        G = nx.path_graph(n)
        expected_minmax = math.ceil(n / q)
        expected_maxmin = math.floor(n / q)

        p_mm = _minmax(G, q)
        assert _is_valid_partition(G, p_mm, q)
        assert _max_weight(p_mm) == expected_minmax

        p_mx = _maxmin(G, q)
        assert _is_valid_partition(G, p_mx, q)
        assert _min_weight(p_mx) == expected_maxmin

    @pytest.mark.parametrize("n", [500, 2000, 10000])
    def test_star_unit_weights_large(self, n):
        G = nx.star_graph(n)
        p_mm = _minmax(G, 2)
        assert _is_valid_partition(G, p_mm, 2)
        assert _max_weight(p_mm) == n

        p_mx = _maxmin(G, 2)
        assert _is_valid_partition(G, p_mx, 2)
        assert _min_weight(p_mx) == 1

    @pytest.mark.parametrize("n,q", [(500, 5), (2000, 10), (10000, 25)])
    def test_star_q_parts_large(self, n, q):
        G = nx.star_graph(n)
        p_mm = _minmax(G, q)
        assert _is_valid_partition(G, p_mm, q)
        assert _max_weight(p_mm) == n - q + 2

        p_mx = _maxmin(G, q)
        assert _is_valid_partition(G, p_mx, q)
        assert _min_weight(p_mx) == 1

    @pytest.mark.parametrize("m", [500, 2000, 10000])
    def test_double_star_q2_large(self, m):
        G = nx.Graph()
        G.add_edge(0, 1)
        for i in range(2, m + 2):
            G.add_edge(0, i)
        for i in range(m + 2, 2 * m + 2):
            G.add_edge(1, i)

        p_mm = _minmax(G, 2)
        assert _is_valid_partition(G, p_mm, 2)
        assert _max_weight(p_mm) == m + 1

        p_mx = _maxmin(G, 2)
        assert _is_valid_partition(G, p_mx, 2)
        assert _min_weight(p_mx) == m + 1

    @pytest.mark.parametrize("n,q", [(1000, 4), (5000, 5), (15000, 3)])
    def test_path_divisible_large(self, n, q):
        assert n % q == 0
        G = nx.path_graph(n)
        piece = n // q

        p_mm = _minmax(G, q)
        assert _is_valid_partition(G, p_mm, q)
        assert _max_weight(p_mm) == piece
        assert _min_weight(p_mm) == piece

        p_mx = _maxmin(G, q)
        assert _is_valid_partition(G, p_mx, q)
        assert _max_weight(p_mx) == piece
        assert _min_weight(p_mx) == piece

    @pytest.mark.parametrize(
        "k,h",
        [
            (2, 4),
            (2, 5),
            (3, 3),
            (3, 4),
        ],
    )
    def test_kary_balanced_tree_q2_large(self, k, h):
        G = nx.balanced_tree(k, h)
        expected_minmax = k**h
        expected_maxmin = (k**h - 1) // (k - 1)

        p_mm = _minmax(G, 2)
        assert _is_valid_partition(G, p_mm, 2)
        assert _max_weight(p_mm) == expected_minmax

        p_mx = _maxmin(G, 2)
        assert _is_valid_partition(G, p_mx, 2)
        assert _min_weight(p_mx) == expected_maxmin

    @pytest.mark.parametrize(
        "spine,leaves_per_node",
        [(5, 3), (10, 2), (8, 4)],
    )
    def test_uniform_caterpillar_perfect_partition_large(self, spine, leaves_per_node):
        G = _caterpillar(spine, leaves_per_node)
        q = spine
        expected = leaves_per_node + 1

        p_mm = _minmax(G, q)
        assert _is_valid_partition(G, p_mm, q)
        assert _max_weight(p_mm) == expected

        p_mx = _maxmin(G, q)
        assert _is_valid_partition(G, p_mx, q)
        assert _min_weight(p_mx) == expected

    @pytest.mark.parametrize(
        "m,a,b",
        [(5, 2, 3), (10, 1, 4), (20, 3, 5)],
    )
    def test_weighted_alternating_path_perfect_partition(self, m, a, b):
        G = nx.path_graph(2 * m)
        weights = {i: (a if i % 2 == 0 else b) for i in range(2 * m)}
        nx.set_node_attributes(G, weights, "weight")
        expected = a + b

        p_mm = _minmax(G, m, node_weight="weight")
        assert _is_valid_partition(G, p_mm, m)
        assert _max_weight(p_mm) == expected

        p_mx = _maxmin(G, m, node_weight="weight")
        assert _is_valid_partition(G, p_mx, m)
        assert _min_weight(p_mx) == expected

    def test_extreme_weight_ratio_minmax_exact(self):
        """Regression: a binary-search tolerance relative to total weight
        returned a suboptimal partition when weights spanned ~12 orders of
        magnitude; integer bisection is exact."""
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {0: 10**12, 1: 3, 2: 3, 3: 10**12}, "weight")
        p = _minmax(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p, 2)
        assert _max_weight(p) == 10**12 + 3

    def test_extreme_weight_ratio_maxmin_exact(self):
        G = nx.path_graph(3)
        nx.set_node_attributes(G, {0: 10**10, 1: 1, 2: 1}, "weight")
        p = _maxmin(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p, 2)
        assert _min_weight(p) == 2

    def test_extreme_weight_ratio_float_maxmin_exact(self):
        """Same regression with float weights: float-grid bisection is exact
        over IEEE-754 doubles."""
        G = nx.path_graph(3)
        nx.set_node_attributes(G, {0: 1e10, 1: 1.0, 2: 1.0}, "weight")
        p = _maxmin(G, 2, node_weight="weight")
        assert _min_weight(p) == 2.0

    @pytest.mark.parametrize("fn", ["minmax", "maxmin"])
    def test_arbitrary_precision_integer_weights(self, fn):
        """Integer weights beyond 2**53 (and beyond float range entirely)
        are partitioned exactly via integer bisection.  Uses the raw API
        and exact int sums: float conversion of these weights overflows."""
        big = 10**100
        w = {0: big, 1: 3, 2: 3, 3: big}
        G = nx.path_graph(4)
        nx.set_node_attributes(G, w, "weight")
        raw = min_max_tree_partition if fn == "minmax" else max_min_tree_partition
        parts = raw(G, 2, node_weight="weight")
        # The unique optimum for both objectives cuts the middle edge.
        assert sorted(sorted(p) for p in parts) == [[0, 1], [2, 3]]
        part_weights = [sum(w[v] for v in p) for p in parts]
        assert (max(part_weights) if fn == "minmax" else min(part_weights)) == big + 3

    def test_tiny_float_weights_exact(self):
        """Regression: an absolute eps=1e-12 inside the oracles degraded
        results for weights near or below 1e-12."""
        G = nx.path_graph(6)
        nx.set_node_attributes(G, dict.fromkeys(G, 1e-15), "weight")
        p_mm = _minmax(G, 3, node_weight="weight")
        assert _max_weight(p_mm) == 2e-15
        p_mx = _maxmin(G, 3, node_weight="weight")
        assert _min_weight(p_mx) == 2e-15


# ---------------------------------------------------------------------------
# Bound-sanity tests for large trees
# ---------------------------------------------------------------------------


class TestBoundSanity:
    def _check_bounds(self, T: nx.Graph, q: int, weight: str = "weight") -> None:
        W_node = {v: T.nodes[v].get(weight, 1) for v in T.nodes}
        W = sum(W_node.values())
        max_w = max(W_node.values())

        lb_minmax = max(
            W / q,
            max_w,
            _bottleneck_node_lb_minmax(T, q, weight=weight),
        )

        ub_maxmin = W / q
        if q >= 2:
            ub_maxmin = min(ub_maxmin, (W - max_w) / (q - 1))

        greedy_max, greedy_min = _greedy_partition_bounds(T, q, weight=weight)

        p_mm = _minmax(T, q, node_weight=weight)
        opt_minmax = _max_weight(p_mm)
        assert opt_minmax >= lb_minmax - 1e-9, (
            f"min-max OPT {opt_minmax} is below lower bound {lb_minmax}"
        )
        assert opt_minmax <= greedy_max + 1e-9, (
            f"min-max OPT {opt_minmax} exceeds greedy upper bound {greedy_max}"
        )

        p_mx = _maxmin(T, q, node_weight=weight)
        opt_maxmin = _min_weight(p_mx)
        assert opt_maxmin <= ub_maxmin + 1e-9, (
            f"max-min OPT {opt_maxmin} exceeds upper bound {ub_maxmin}"
        )
        assert opt_maxmin >= greedy_min - 1e-9, (
            f"max-min OPT {opt_maxmin} is below greedy lower bound {greedy_min}"
        )

    @pytest.mark.parametrize(
        "n,q",
        [
            (500, 3),
            (1000, 5),
            (5000, 4),
            (10000, 7),
        ],
    )
    def test_path_unit_weights(self, n, q):
        self._check_bounds(nx.path_graph(n), q)

    @pytest.mark.parametrize(
        "n,q",
        [
            (500, 3),
            (2000, 5),
            (8000, 10),
        ],
    )
    def test_star_unit_weights(self, n, q):
        self._check_bounds(nx.star_graph(n), q)

    @pytest.mark.parametrize(
        "k,h,q",
        [
            (2, 7, 3),
            (2, 9, 5),
            (3, 5, 4),
            (3, 6, 7),
        ],
    )
    def test_balanced_tree_unit_weights(self, k, h, q):
        self._check_bounds(nx.balanced_tree(k, h), q)

    @pytest.mark.parametrize(
        "seed,n,q",
        [
            (0, 500, 3),
            (1, 1000, 4),
            (2, 2000, 6),
            (3, 5000, 5),
            (4, 10000, 8),
            (5, 500, 2),
            (6, 3000, 10),
        ],
    )
    def test_random_tree_unit_weights(self, seed, n, q):
        T = nx.random_labeled_tree(n, seed=seed)
        self._check_bounds(T, q)

    @pytest.mark.parametrize(
        "seed,n,q",
        [
            (10, 300, 3),
            (11, 800, 5),
            (12, 2000, 4),
            (13, 5000, 6),
        ],
    )
    def test_random_tree_random_weights(self, seed, n, q):
        rng = random.Random(seed)
        T = nx.random_labeled_tree(n, seed=seed)
        weights = {v: rng.randint(1, 20) for v in T.nodes}
        nx.set_node_attributes(T, weights, "weight")
        self._check_bounds(T, q, weight="weight")

    @pytest.mark.parametrize(
        "seed,n,q",
        [
            (20, 400, 4),
            (21, 1000, 5),
            (22, 3000, 8),
        ],
    )
    def test_random_tree_skewed_weights(self, seed, n, q):
        rng = random.Random(seed)
        T = nx.random_labeled_tree(n, seed=seed)
        heavy = rng.choice(list(T.nodes))
        weights = {v: (n if v == heavy else 1) for v in T.nodes}
        nx.set_node_attributes(T, weights, "weight")
        self._check_bounds(T, q, weight="weight")

    @pytest.mark.parametrize(
        "k,L,case",
        [
            (3, 100, "q2"),
            (5, 50, "q2"),
            (4, 200, "q2"),
            (4, 100, "qk"),
            (6, 50, "qk"),
            (3, 100, "qk1"),
            (4, 80, "qk1"),
        ],
    )
    def test_spider_graph(self, k, L, case):
        G = _spider_graph(k, L)
        assert len(G) == 1 + k * L

        if case == "q2":
            q, expected_minmax, expected_maxmin = 2, 1 + (k - 1) * L, L
        elif case == "qk":
            q, expected_minmax, expected_maxmin = k, L + 1, L
        else:
            n = 1 + k * L
            q = k + 1
            expected_minmax = math.ceil(n / q)
            expected_maxmin = n // q

        p_mm = _minmax(G, q)
        assert _is_valid_partition(G, p_mm, q)
        assert _max_weight(p_mm) == expected_minmax

        p_mx = _maxmin(G, q)
        assert _is_valid_partition(G, p_mx, q)
        assert _min_weight(p_mx) == expected_maxmin

    @pytest.mark.parametrize(
        "num_hubs,m",
        [(5, 10), (8, 20), (10, 5)],
    )
    def test_chain_of_stars_perfect(self, num_hubs, m):
        G = _chain_of_stars(num_hubs, m)
        assert len(G) == num_hubs * (m + 1)
        expected = m + 1

        p_mm = _minmax(G, num_hubs)
        assert _is_valid_partition(G, p_mm, num_hubs)
        assert _max_weight(p_mm) == expected

        p_mx = _maxmin(G, num_hubs)
        assert _is_valid_partition(G, p_mx, num_hubs)
        assert _min_weight(p_mx) == expected

    @pytest.mark.parametrize(
        "num_hubs,m",
        [(6, 15), (4, 8), (10, 3)],
    )
    def test_chain_of_stars_one_fewer_cut(self, num_hubs, m):
        G = _chain_of_stars(num_hubs, m)
        q = num_hubs - 1

        p_mm = _minmax(G, q)
        assert _is_valid_partition(G, p_mm, q)
        assert _max_weight(p_mm) == 2 * (m + 1)

        p_mx = _maxmin(G, q)
        assert _is_valid_partition(G, p_mx, q)
        assert _min_weight(p_mx) == m + 1

    @pytest.mark.parametrize("s", [5, 10, 50, 100])
    def test_broom_perfect_q2(self, s):
        G = _broom_graph(s, s + 1)
        assert len(G) == 2 * (s + 1)
        expected = s + 1

        p_mm = _minmax(G, 2)
        assert _is_valid_partition(G, p_mm, 2)
        assert _max_weight(p_mm) == expected

        p_mx = _maxmin(G, 2)
        assert _is_valid_partition(G, p_mx, 2)
        assert _min_weight(p_mx) == expected

    @pytest.mark.parametrize(
        "n,H",
        [(100, 10_000), (200, 50_000), (1001, 100_000)],
    )
    def test_heavy_node_path_q2(self, n, H):
        p = n // 2
        G = nx.path_graph(n)
        nx.set_node_attributes(G, {v: (H if v == p else 1) for v in G.nodes}, "weight")

        pm = _minmax(G, 2, node_weight="weight")
        assert _is_valid_partition(G, pm, 2)
        assert _max_weight(pm) == H + (n - 1 - p)

        px = _maxmin(G, 2, node_weight="weight")
        assert _is_valid_partition(G, px, 2)
        assert _min_weight(px) == p

    @pytest.mark.parametrize(
        "s,d",
        [(6, 4), (10, 3), (8, 7)],
    )
    def test_comb_perfect_partition(self, s, d):
        G = _comb_graph(s, d)
        assert len(G) == s * (d + 1)
        expected = d + 1

        p_mm = _minmax(G, s)
        assert _is_valid_partition(G, p_mm, s)
        assert _max_weight(p_mm) == expected

        p_mx = _maxmin(G, s)
        assert _is_valid_partition(G, p_mx, s)
        assert _min_weight(p_mx) == expected

    @pytest.mark.parametrize(
        "n,j",
        [
            (5, 1),
            (6, 1),
            (6, 2),
            (7, 1),
            (7, 2),
            (7, 3),
        ],
    )
    def test_binomial_tree_perfect_partition(self, n, j):
        G = nx.binomial_tree(n)
        assert len(G) == 2**n
        q = 2**j
        expected = 2 ** (n - j)

        p_mm = _minmax(G, q)
        assert _is_valid_partition(G, p_mm, q)
        assert _max_weight(p_mm) == expected

        p_mx = _maxmin(G, q)
        assert _is_valid_partition(G, p_mx, q)
        assert _min_weight(p_mx) == expected

    @pytest.mark.parametrize(
        "k,L,q",
        [(4, 500, 2), (5, 200, 3), (3, 1000, 4)],
    )
    def test_spider_bounds_sanity(self, k, L, q):
        self._check_bounds(_spider_graph(k, L), q)

    @pytest.mark.parametrize(
        "num_hubs,m,q",
        [(10, 50, 5), (8, 30, 4), (6, 100, 3)],
    )
    def test_chain_of_stars_bounds_sanity(self, num_hubs, m, q):
        self._check_bounds(_chain_of_stars(num_hubs, m), q)

    @pytest.mark.parametrize(
        "s,a,q",
        [(50, 100, 2), (30, 200, 2), (100, 50, 2)],
    )
    def test_broom_bounds_sanity(self, s, a, q):
        self._check_bounds(_broom_graph(s, a), q)

    @pytest.mark.parametrize(
        "n,H,q",
        [(500, 100_000, 3), (1000, 50_000, 2), (300, 200_000, 4)],
    )
    def test_heavy_node_path_bounds_sanity(self, n, H, q):
        G = nx.path_graph(n)
        nx.set_node_attributes(
            G, {v: (H if v == n // 2 else 1) for v in G.nodes}, "weight"
        )
        self._check_bounds(G, q, weight="weight")

    @pytest.mark.parametrize(
        "seed,n,q",
        [(0, 1000, 4), (0, 500, 3), (6, 750, 5)],
    )
    def test_powerlaw_tree_bounds_sanity(self, seed, n, q):
        T = nx.random_powerlaw_tree(n, seed=seed, tries=1000)
        self._check_bounds(T, q)


# ---------------------------------------------------------------------------
# Prior bug case regression test
# ---------------------------------------------------------------------------


class TestPriorBugCase:
    def test_specific_tree_minmax_q5(self):
        """Specific 8-node tree that triggered a bug in a prior implementation."""
        G = nx.Graph()
        weights = {
            0: 1.0000031805205596,
            1: 0.5664521167048723,
            2: 1.1855263904715256,
            3: 6.311715812860592,
            4: 7.941585707193345,
            5: 4.279383671316872,
            6: 0.7289242909043756,
            7: 3.878030936414714,
        }
        for v, w in weights.items():
            G.add_node(v, weight=w)
        for u, v in [(0, 1), (1, 2), (1, 3), (1, 4), (1, 5), (4, 6), (3, 7)]:
            G.add_edge(u, v)

        expected = 7.941585707193345
        p = _minmax(G, 5, node_weight="weight")
        assert _is_valid_partition(G, p, 5)
        assert math.isclose(_max_weight(p), expected, rel_tol=1e-6)

        # Verify against brute force
        p_bf = _brute_force_partition(G, 5, False)
        assert math.isclose(_max_weight(p), _max_weight(p_bf), rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Weight function tests
# ---------------------------------------------------------------------------


def _random_tree_with_weights(n, rng):
    """Build a random tree on n nodes with random vertex and edge weights."""
    T = nx.random_labeled_tree(n, seed=rng.randint(0, 10**6))
    for v in T.nodes:
        T.nodes[v]["weight"] = round(rng.uniform(0.1, 10.0), 4)
    for u, v in T.edges:
        T.edges[u, v]["weight"] = round(rng.uniform(0.1, 5.0), 4)
    return T


class TestWeightFunctionRandomized:
    """For each additive weight function, compare binary-search result against
    generalized brute force on small random trees."""

    ADDITIVE_WFS = [
        ("vertex_weight_sum", "vsum"),
        ("edge_weight_sum", "esum"),
        ("mixed_sum", "mixed"),
        ("vertex_count", "count"),
    ]

    @pytest.mark.parametrize("seed", [0, 1, 7, 13, 42, 99, 123, 200])
    def test_minmax_additive(self, seed):
        rng = random.Random(seed)
        for trial in range(5):
            n = rng.randint(3, 9)
            q = rng.randint(2, min(n, 5))
            T = _random_tree_with_weights(n, rng)

            for wf_name, label in self.ADDITIVE_WFS:
                p_algo = _minmax(T, q, weight_function=wf_name)
                p_bf = _brute_force_partition(T, q, False, weight_function=wf_name)

                assert _is_valid_partition(T, p_algo, q, weight_function=wf_name), (
                    f"Invalid partition: seed={seed} trial={trial} n={n} q={q} wf={label}"
                )
                assert math.isclose(
                    _max_weight(p_algo), _max_weight(p_bf), rel_tol=1e-6
                ), (
                    f"Mismatch: seed={seed} trial={trial} n={n} q={q} wf={label} "
                    f"got={_max_weight(p_algo)} expected={_max_weight(p_bf)}"
                )

    @pytest.mark.parametrize("seed", [0, 1, 7, 13, 42, 99, 123, 200])
    def test_maxmin_additive(self, seed):
        rng = random.Random(seed)
        for trial in range(5):
            n = rng.randint(3, 9)
            q = rng.randint(2, min(n, 5))
            T = _random_tree_with_weights(n, rng)

            for wf_name, label in self.ADDITIVE_WFS:
                p_algo = _maxmin(T, q, weight_function=wf_name)
                p_bf = _brute_force_partition(T, q, True, weight_function=wf_name)

                assert _is_valid_partition(T, p_algo, q, weight_function=wf_name), (
                    f"Invalid partition: seed={seed} trial={trial} n={n} q={q} wf={label}"
                )
                assert math.isclose(
                    _min_weight(p_algo), _min_weight(p_bf), rel_tol=1e-6
                ), (
                    f"Mismatch: seed={seed} trial={trial} n={n} q={q} wf={label} "
                    f"got={_min_weight(p_algo)} expected={_min_weight(p_bf)}"
                )


# ---------------------------------------------------------------------------
# Component consistency for multiple weight functions
# ---------------------------------------------------------------------------


class TestComponentConsistencyMultiWF:
    """For each weight function on medium trees, verify structural properties:
    exactly q components, disjoint node coverage, connectivity, and reported
    weight matches recomputed weight."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 99, 123])
    def test_consistency(self, seed):
        rng = random.Random(seed)
        for trial in range(4):
            n = rng.randint(5, 50)
            q = rng.randint(2, min(n, 10))
            T = _random_tree_with_weights(n, rng)

            for wf_name in ["vertex_weight_sum", "edge_weight_sum", "mixed_sum", "vertex_count"]:
                for fn in [_minmax, _maxmin]:
                    p = fn(T, q, weight_function=wf_name)
                    assert len(p) == q, f"expected {q} comps, got {len(p)}"
                    all_verts = set()
                    for nodes, w in p:
                        assert not (nodes & all_verts), "vertex in two components"
                        all_verts |= nodes
                    assert all_verts == set(T.nodes), "missing or extra vertices"

                    for nodes, reported_w in p:
                        [computed_w] = tree_partition_weights(
                            T, [frozenset(nodes)], weight_function=wf_name
                        )
                        assert math.isclose(reported_w, computed_w, rel_tol=1e-9), (
                            f"reported {reported_w} != computed {computed_w}"
                        )


# ---------------------------------------------------------------------------
# Analytic tests for weight_function="edge_weight_sum"
# ---------------------------------------------------------------------------


class TestEdgeWeightSumAnalytic:
    """Analytical tests for weight_function='edge_weight_sum'.

    Component weight = sum of edge weights within the component.
    Singletons have weight 0 (no edges).
    """

    @staticmethod
    def _path_with_uniform_edge_weight(n, w):
        G = nx.path_graph(n)
        for u, v in G.edges:
            G.edges[u, v]["weight"] = w
        return G

    @pytest.mark.parametrize(
        "n,q,w",
        [
            (6, 3, 1.0),  # 6/3=2 nodes each, 1 edge each → w
            (9, 3, 2.0),  # 9/3=3 nodes each, 2 edges each → 2w
            (12, 4, 1.0),  # 12/4=3 nodes each, 2 edges each → 2w
            (10, 5, 3.0),  # 10/5=2 nodes each, 1 edge each → w
        ],
    )
    def test_path_uniform_edge_weights(self, n, q, w):
        """P_n with uniform edge weights w, n divisible by q.

        Each component has n/q nodes and n/q - 1 edges → weight (n/q - 1)*w.
        Perfect partition: min-max = max-min = (n/q - 1)*w.
        """
        assert n % q == 0
        G = self._path_with_uniform_edge_weight(n, w)
        expected = (n // q - 1) * w

        p_mm = _minmax(G, q, weight_function="edge_weight_sum")
        assert _is_valid_partition(G, p_mm, q, weight_function="edge_weight_sum")
        assert math.isclose(_max_weight(p_mm), expected)

        p_mx = _maxmin(G, q, weight_function="edge_weight_sum")
        assert _is_valid_partition(G, p_mx, q, weight_function="edge_weight_sum")
        assert math.isclose(_min_weight(p_mx), expected)

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_star_uniform_edge_weights_q2(self, n):
        """S_n with uniform edge weights 1.0, q=2.

        One cut isolates a leaf (weight 0) from the rest (n-1 edges, weight n-1).
        min-max = n-1, max-min = 0.
        """
        G = nx.star_graph(n)
        for u, v in G.edges:
            G.edges[u, v]["weight"] = 1.0

        p_mm = _minmax(G, 2, weight_function="edge_weight_sum")
        assert math.isclose(_max_weight(p_mm), n - 1)

        p_mx = _maxmin(G, 2, weight_function="edge_weight_sum")
        assert math.isclose(_min_weight(p_mx), 0.0)

    @pytest.mark.parametrize("n,q", [(5, 3), (8, 5), (10, 6)])
    def test_star_multiple_cuts(self, n, q):
        """S_n with uniform edge weights, q parts.

        q-1 cuts isolate q-1 leaves (weight 0 each).  Center keeps n-q+1
        leaves → n-q+1 edges → weight n-q+1.
        min-max = n-q+1, max-min = 0.
        """
        G = nx.star_graph(n)
        for u, v in G.edges:
            G.edges[u, v]["weight"] = 1.0

        p_mm = _minmax(G, q, weight_function="edge_weight_sum")
        assert math.isclose(_max_weight(p_mm), n - q + 1)

        p_mx = _maxmin(G, q, weight_function="edge_weight_sum")
        assert math.isclose(_min_weight(p_mx), 0.0)

    @pytest.mark.parametrize(
        "m,a,b",
        [(3, 1.0, 2.0), (5, 2.0, 3.0), (4, 1.5, 4.5)],
    )
    def test_path_paired_edge_weights(self, m, a, b):
        """P_{2m+1} with alternating edge weights [a, b, a, b, ...], q=m:
        non-uniform edge weights checked against the brute-force oracle."""
        n = 2 * m + 1
        G = nx.path_graph(n)
        for i, (u, v) in enumerate(sorted(G.edges)):
            G.edges[u, v]["weight"] = a if i % 2 == 0 else b

        p_mm = _minmax(G, m, weight_function="edge_weight_sum")
        p_bf = _brute_force_partition(G, m, False, weight_function="edge_weight_sum")
        assert math.isclose(_max_weight(p_mm), _max_weight(p_bf), rel_tol=1e-6)

        p_mx = _maxmin(G, m, weight_function="edge_weight_sum")
        p_bf2 = _brute_force_partition(G, m, True, weight_function="edge_weight_sum")
        assert math.isclose(_min_weight(p_mx), _min_weight(p_bf2), rel_tol=1e-6)

    @pytest.mark.parametrize(
        "num_hubs,m,w",
        [(3, 5, 1.0), (5, 3, 2.0), (4, 4, 1.5)],
    )
    def test_chain_of_stars_edge_weights(self, num_hubs, m, w):
        """Chain of num_hubs stars with m leaves each, all edge weights w.

        q=num_hubs: cut all hub-hub edges.
        Each component = hub + m leaves → m edges → weight m*w.
        Perfect partition: min-max = max-min = m*w.
        """
        G = _chain_of_stars(num_hubs, m)
        for u, v in G.edges:
            G.edges[u, v]["weight"] = w

        p_mm = _minmax(G, num_hubs, weight_function="edge_weight_sum")
        assert math.isclose(_max_weight(p_mm), m * w)

        p_mx = _maxmin(G, num_hubs, weight_function="edge_weight_sum")
        assert math.isclose(_min_weight(p_mx), m * w)


# ---------------------------------------------------------------------------
# Analytic tests for weight_function="vertex_count"
# ---------------------------------------------------------------------------


class TestVertexCountAnalytic:
    """Analytical tests for weight_function='vertex_count'.

    Component weight = number of vertices.  Equivalent to vertex_weight_sum
    with all weights = 1, but tested explicitly via the string API.
    """

    @pytest.mark.parametrize(
        "n,q",
        [(6, 2), (6, 3), (7, 2), (9, 3), (10, 4), (12, 4)],
    )
    def test_path_vertex_count(self, n, q):
        """P_n, q parts: min-max = ceil(n/q), max-min = floor(n/q)."""
        G = nx.path_graph(n)
        expected_minmax = math.ceil(n / q)
        expected_maxmin = math.floor(n / q)

        p_mm = _minmax(G, q, weight_function="vertex_count")
        assert _max_weight(p_mm) == expected_minmax

        p_mx = _maxmin(G, q, weight_function="vertex_count")
        assert _min_weight(p_mx) == expected_maxmin

    @pytest.mark.parametrize("n", [4, 6, 10])
    def test_star_vertex_count_q2(self, n):
        """S_n, q=2: min-max = n (center + n-1 leaves), max-min = 1."""
        G = nx.star_graph(n)

        p_mm = _minmax(G, 2, weight_function="vertex_count")
        assert _max_weight(p_mm) == n

        p_mx = _maxmin(G, 2, weight_function="vertex_count")
        assert _min_weight(p_mx) == 1

    @pytest.mark.parametrize(
        "spine,leaves_per_node",
        [(3, 2), (4, 2), (5, 3)],
    )
    def test_caterpillar_vertex_count_perfect(self, spine, leaves_per_node):
        """Caterpillar q=spine: each component = spine_node + leaves = lp+1."""
        G = _caterpillar(spine, leaves_per_node)
        expected = leaves_per_node + 1

        p_mm = _minmax(G, spine, weight_function="vertex_count")
        assert _max_weight(p_mm) == expected

        p_mx = _maxmin(G, spine, weight_function="vertex_count")
        assert _min_weight(p_mx) == expected


# ---------------------------------------------------------------------------
# Analytic tests for weight_function="mixed_sum"
# ---------------------------------------------------------------------------


class TestMixedSumAnalytic:
    """Analytical tests for weight_function='mixed_sum'.

    Component weight = sum of vertex weights + sum of edge weights.
    """

    @pytest.mark.parametrize(
        "n,q",
        [(6, 3), (9, 3), (10, 5), (12, 4)],
    )
    def test_path_unit_weights(self, n, q):
        """P_n with unit vertex and edge weights, q parts (n divisible by q).

        Component with k nodes: weight = k + (k-1) = 2k-1.
        With n/q nodes per part: weight = 2*(n/q) - 1.
        """
        assert n % q == 0
        G = nx.path_graph(n)
        for u, v in G.edges:
            G.edges[u, v]["weight"] = 1.0
        piece = n // q
        expected = 2 * piece - 1

        p_mm = _minmax(G, q, weight_function="mixed_sum")
        assert _is_valid_partition(G, p_mm, q, weight_function="mixed_sum")
        assert math.isclose(_max_weight(p_mm), expected)

        p_mx = _maxmin(G, q, weight_function="mixed_sum")
        assert _is_valid_partition(G, p_mx, q, weight_function="mixed_sum")
        assert math.isclose(_min_weight(p_mx), expected)

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_star_unit_weights_q2(self, n):
        """S_n with unit vertex and edge weights, q=2.

        One cut: singleton leaf (weight 1, no edges) vs rest (n nodes, n-1 edges
        → weight n + (n-1) = 2n-1).
        min-max = 2n-1, max-min = 1.
        """
        G = nx.star_graph(n)
        for u, v in G.edges:
            G.edges[u, v]["weight"] = 1.0

        p_mm = _minmax(G, 2, weight_function="mixed_sum")
        assert math.isclose(_max_weight(p_mm), 2 * n - 1)

        p_mx = _maxmin(G, 2, weight_function="mixed_sum")
        assert math.isclose(_min_weight(p_mx), 1.0)

    @pytest.mark.parametrize(
        "num_hubs,m",
        [(3, 4), (5, 3), (4, 5)],
    )
    def test_chain_of_stars_mixed_perfect(self, num_hubs, m):
        """Chain of stars, unit vertex + edge weights, q=num_hubs.

        Each component = hub + m leaves → m+1 vertices + m edges → weight 2m+1.
        Perfect partition: min-max = max-min = 2m+1.
        """
        G = _chain_of_stars(num_hubs, m)
        for u, v in G.edges:
            G.edges[u, v]["weight"] = 1.0
        expected = 2 * m + 1

        p_mm = _minmax(G, num_hubs, weight_function="mixed_sum")
        assert math.isclose(_max_weight(p_mm), expected)

        p_mx = _maxmin(G, num_hubs, weight_function="mixed_sum")
        assert math.isclose(_min_weight(p_mx), expected)


# ---------------------------------------------------------------------------
# String API validation
# ---------------------------------------------------------------------------


class TestWeightFunctionStringAPI:
    """Verify that the string-based weight_function parameter works correctly."""

    def test_invalid_weight_function_raises(self):
        G = nx.path_graph(4)
        with pytest.raises(nx.NetworkXError, match="weight_function"):
            _minmax(G, 2, weight_function="nonexistent")
        with pytest.raises(nx.NetworkXError, match="weight_function"):
            _maxmin(G, 2, weight_function="bad")

    def test_default_matches_vertex_weight_sum(self):
        """Default weight_function gives the same result as explicit 'vertex_weight_sum'."""
        G = nx.path_graph(8)
        for v in G.nodes:
            G.nodes[v]["weight"] = v + 1

        p_default = _minmax(G, 3, node_weight="weight")
        p_explicit = _minmax(
            G, 3, node_weight="weight", weight_function="vertex_weight_sum"
        )
        assert _max_weight(p_default) == _max_weight(p_explicit)


# ---------------------------------------------------------------------------
# Regression tests for review-cycle bug fixes
# ---------------------------------------------------------------------------


class TestReviewRegressions:
    """Pin down the contract of each fix applied during code review."""

    # All-zero edge_weight_sum: contract is len(partition) == q, not 1.
    @pytest.mark.parametrize("n,q", [(4, 2), (4, 3), (6, 2), (6, 4), (10, 5)])
    def test_edge_weight_sum_all_zero_returns_q_parts(self, n, q):
        G = nx.path_graph(n)
        nx.set_edge_attributes(G, 0.0, "weight")

        p_mm = _minmax(G, q, weight_function="edge_weight_sum")
        assert len(p_mm) == q
        assert all(w == 0.0 for _, w in p_mm)
        assert _is_valid_partition(G, p_mm, q, weight_function="edge_weight_sum")

        p_mx = _maxmin(G, q, weight_function="edge_weight_sum")
        assert len(p_mx) == q
        assert all(w == 0.0 for _, w in p_mx)
        assert _is_valid_partition(G, p_mx, q, weight_function="edge_weight_sum")

    # Deterministic tiebreaker in _reduce_cuts_to_reach_q: same input must
    # always yield the same output.  Pre-fix, id(cut) varied across runs.
    @pytest.mark.parametrize("seed", [0, 1, 7, 13, 42])
    def test_tiebreaker_determinism_maxmin(self, seed):
        """Repeated calls on the same tree give identical partitions."""
        rng = random.Random(seed)
        n = rng.randint(8, 20)
        T = nx.random_labeled_tree(n, seed=seed)
        for v in T.nodes:
            T.nodes[v]["weight"] = rng.randint(1, 3)  # small range → many ties
        q = rng.randint(2, min(n, 5))

        first = _maxmin(T, q, node_weight="weight")
        for _ in range(5):
            again = _maxmin(T, q, node_weight="weight")
            assert again == first

    def test_tiebreaker_determinism_symmetric_star(self):
        """Symmetric star forces ties in _reduce_cuts_to_reach_q."""
        G = nx.star_graph(8)  # center 0 + 8 leaves, all unit weight
        first = _maxmin(G, 4)
        for _ in range(10):
            assert _maxmin(G, 4) == first

    # Custom edge_weight attribute name: edge_weight_sum / mixed_sum must
    # honor a non-default attribute name.
    def test_edge_weight_custom_attr_name(self):
        G = nx.path_graph(6)
        nx.set_edge_attributes(G, {e: 2.0 for e in G.edges}, "cost")
        # edge_weight_sum with custom name
        p = _minmax(G, 3, edge_weight="cost", weight_function="edge_weight_sum")
        # 6 nodes in 3 parts of 2 nodes each → 1 edge per part → weight = 2.0
        assert math.isclose(_max_weight(p), 2.0)
        assert _is_valid_partition(
            G, p, 3, weight_function="edge_weight_sum", edge_weight="cost"
        )

    def test_mixed_sum_distinct_node_and_edge_attrs(self):
        """mixed_sum with separately-named node and edge attributes — newly
        reachable via the string API after the node_weight/edge_weight split.
        """
        G = nx.path_graph(6)
        nx.set_node_attributes(G, 3, "size")
        nx.set_edge_attributes(G, 1, "len")
        p = _minmax(
            G,
            3,
            node_weight="size",
            edge_weight="len",
            weight_function="mixed_sum",
        )
        # 2 nodes (2 * 3 = 6) + 1 edge (1) = 7 per component
        assert math.isclose(_max_weight(p), 7.0)
        assert _is_valid_partition(
            G, p, 3, weight_function="mixed_sum", node_weight="size", edge_weight="len"
        )


# ---------------------------------------------------------------------------
# numpy weights
# ---------------------------------------------------------------------------


class TestNumpyWeights:
    """numpy scalar weights must behave like their Python counterparts.

    numpy fixed-width integers pass isinstance(w, numbers.Integral); pre-fix
    they skipped coercion and silently wrapped at 64 bits during probe and
    grid arithmetic, returning a wrong partition with only a RuntimeWarning.
    """

    def test_numpy_int_weights_are_exact_minmax(self):
        np = pytest.importorskip("numpy")
        G = nx.path_graph(4)
        for v, w in enumerate([2**61, 2**61, 2**62, 2**61]):
            G.nodes[v]["weight"] = np.int64(w)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            parts = min_max_tree_partition(G, 2)
        # Optimal cut is (1, 2): max part weight 3 * 2**61 beats 2**63.
        assert set(parts) == {frozenset({0, 1}), frozenset({2, 3})}

    def test_numpy_int_weights_are_exact_maxmin(self):
        np = pytest.importorskip("numpy")
        G = nx.path_graph(4)
        for v, w in enumerate([2**61, 2**61, 2**62, 2**61]):
            G.nodes[v]["weight"] = np.int64(w)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            parts = max_min_tree_partition(G, 2)
        # Optimal cut is (1, 2): min part weight 2**62 beats 2**61.
        assert set(parts) == {frozenset({0, 1}), frozenset({2, 3})}

    def test_numpy_float_weights(self):
        np = pytest.importorskip("numpy")
        G = nx.path_graph(4)
        for v, w in enumerate([3.0, 1.0, 1.0, 3.0]):
            G.nodes[v]["weight"] = np.float64(w)
        parts = min_max_tree_partition(G, 2)
        assert set(parts) == {frozenset({0, 1}), frozenset({2, 3})}


# ---------------------------------------------------------------------------
# Missing edge attributes default to 1
# ---------------------------------------------------------------------------


class TestMissingEdgeAttrDefault:
    """Edges missing the weight attribute default to 1 (documented).

    These trees are built so the optimal SHAPE depends on the default: with
    a wrong default (e.g. 0) the returned partition itself changes, so the
    tests cannot be fooled by weight bookkeeping that is consistently wrong
    on both sides of a comparison.  On the 6-path with explicit weight 1 on
    edges (0,1), (1,2), (4,5) and edges (2,3), (3,4) missing, cutting (2,3)
    is the unique optimum for both objectives under default 1 (cut values
    4/3/2/3/4 for min-max, 0/1/2/1/0 for max-min), while default 0 would
    move the min-max optimum to cutting (1,2).
    """

    @pytest.mark.parametrize("weight_function", ["edge_weight_sum", "mixed_sum"])
    def test_path_missing_edge_attrs_minmax(self, weight_function):
        G = nx.path_graph(6)
        G.edges[0, 1]["weight"] = 1
        G.edges[1, 2]["weight"] = 1
        G.edges[4, 5]["weight"] = 1
        parts = min_max_tree_partition(G, 2, weight_function=weight_function)
        assert set(parts) == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}

    @pytest.mark.parametrize("weight_function", ["edge_weight_sum", "mixed_sum"])
    def test_path_missing_edge_attrs_maxmin(self, weight_function):
        G = nx.path_graph(6)
        G.edges[0, 1]["weight"] = 1
        G.edges[1, 2]["weight"] = 1
        G.edges[4, 5]["weight"] = 1
        parts = max_min_tree_partition(G, 2, weight_function=weight_function)
        assert set(parts) == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}

    @pytest.mark.parametrize("weight_function", ["edge_weight_sum", "mixed_sum"])
    @pytest.mark.parametrize("seed", range(4))
    def test_random_trees_missing_edge_attrs(self, seed, weight_function):
        """Randomized sweep against the independent brute-force oracle with
        roughly half the edge (and node) attributes missing."""
        rng = random.Random(seed)
        T = nx.random_labeled_tree(8, seed=seed)
        for v in T.nodes:
            if rng.random() < 0.5:
                T.nodes[v]["weight"] = rng.randint(1, 9)
        for u, v in T.edges:
            if rng.random() < 0.5:
                T.edges[u, v]["weight"] = rng.randint(1, 9)
        q = rng.randint(2, 5)

        p_mm = _minmax(T, q, weight_function=weight_function)
        bf_mm = _brute_force_partition(T, q, False, weight_function=weight_function)
        assert _is_valid_partition(T, p_mm, q, weight_function=weight_function)
        assert _max_weight(p_mm) == _max_weight(bf_mm)

        p_mx = _maxmin(T, q, weight_function=weight_function)
        bf_mx = _brute_force_partition(T, q, True, weight_function=weight_function)
        assert _is_valid_partition(T, p_mx, q, weight_function=weight_function)
        assert _min_weight(p_mx) == _min_weight(bf_mx)
