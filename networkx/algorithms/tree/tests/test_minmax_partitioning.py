"""
Tests for min-max and max-min tree partitioning algorithms.

Covers:
- Analytical cases: path graphs, star graphs, balanced binary trees, caterpillar trees,
  spider graphs, chains of stars, broom graphs, comb graphs, binomial trees, and
  weighted paths with a single heavy node. These all have closed-form optimal values,
  so it's easy for us to check that the algorithm returns the correct value and a valid
  partition, even for large trees.
- Edge cases: number_of_pieces=1, number_of_pieces=n, single-node trees, two-node trees.
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
from typing import Any

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
    parts: list[frozenset[Hashable]],
    node_weight: str,
    edge_weight: str,
    weight_function: str,
) -> list[tuple[frozenset[Hashable], float]]:
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
) -> list[tuple[frozenset[Hashable], float]]:
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
) -> list[tuple[frozenset[Hashable], float]]:
    parts = max_min_tree_partition(
        T,
        q,
        node_weight=node_weight,
        edge_weight=edge_weight,
        weight_function=weight_function,
    )
    return _attach_weights(T, parts, node_weight, edge_weight, weight_function)


# ---------------------------------------------------------------------------
# Internal test helpers
# ---------------------------------------------------------------------------


def _max_weight(partition: list[tuple[frozenset[Hashable], float]]) -> float:
    """Weight of the heaviest component in a partition."""
    return max(w for _, w in partition)


def _min_weight(partition: list[tuple[frozenset[Hashable], float]]) -> float:
    """Weight of the lightest component in a partition."""
    return min(w for _, w in partition)


def _is_valid_partition(
    T: nx.Graph,
    partition: list[tuple[frozenset[Hashable], float]],
    q: int,
    *,
    weight_function: str = "vertex_weight_sum",
    node_weight: str = "weight",
    edge_weight: str = "weight",
) -> bool:
    """Return True iff partition is a valid partition of T into q parts.

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
        # If these nodes overlap with any we've seen previously then components
        # are not disjoint
        if nodes & seen:
            return False
        seen |= nodes

        # Ensure component is connected
        sub = T.subgraph(nodes)
        if not nx.is_connected(sub):
            return False

        [expected_w] = tree_partition_weights(
            T, [nodes], node_weight, edge_weight, weight_function=weight_function
        )
        if not math.isclose(reported_w, expected_w):
            return False

    return seen == all_nodes


def _is_sorted_descending(partition: list[tuple[frozenset[Hashable], float]]) -> bool:
    weights = [w for _, w in partition]
    return all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))


def _is_sorted_ascending(partition: list[tuple[frozenset[Hashable], float]]) -> bool:
    weights = [w for _, w in partition]
    return all(weights[i] <= weights[i + 1] for i in range(len(weights) - 1))


# ---------------------------------------------------------------------------
# Brute force and analytic bound functions
# ---------------------------------------------------------------------------


def _brute_force_partition(
    T: nx.Graph,
    q: int,
    maximize_min: bool,
    *,
    weight_function: str = "vertex_weight_sum",
    node_weight: str = "weight",
    edge_weight: str = "weight",
) -> list[tuple[frozenset[Hashable], float]]:
    """Brute-force optimal tree partition by exhaustive edge-cut enumeration.

    Tries all C(n-1, k) ways to remove k = q-1 edges and keeps
    the partition that minimizes the maximum component weight
    (``maximize_min=False``) or maximizes the minimum component weight
    (``maximize_min=True``).  Used here as an independent oracle for
    cross-checking the binary-search solvers on small inputs (n <= 12);
    O(C(n-1, q-1)) — do not use in production.

    Returns a list of ``q`` ``(nodes, weight)`` pairs.  The
    objective-defining component is always first: heaviest first for min-max,
    lightest first for max-min.  This matches the sort order of the public API
    so that ``p[0][1]`` directly gives the optimal objective value.
    """
    k = q - 1
    edges = list(T.edges())

    if k == 0:
        [w] = tree_partition_weights(
            T,
            [frozenset(T.nodes())],
            node_weight,
            edge_weight,
            weight_function=weight_function,
        )
        return [(frozenset(T.nodes()), w)]

    best_val = float("inf") if not maximize_min else float("-inf")
    best_partition = []

    for cut_indices in combinations(range(len(edges)), k):
        cut = [edges[i] for i in cut_indices]
        # Removing k distinct edges from a tree always yields exactly k+1
        # connected components; anything else means T is not a simple tree.
        comps = list(nx.connected_components(nx.restricted_view(T, [], cut)))
        assert len(comps) == q, f"expected {q} components, got {len(comps)}"
        comps_and_weights = [
            (
                frozenset(c),
                tree_partition_weights(
                    T,
                    [frozenset(c)],
                    node_weight,
                    edge_weight,
                    weight_function=weight_function,
                )[0],
            )
            for c in comps
        ]
        val = (
            max(w for _, w in comps_and_weights)
            if not maximize_min
            else min(w for _, w in comps_and_weights)
        )
        improved = (not maximize_min and val < best_val) or (
            maximize_min and val > best_val
        )
        if improved:
            best_val = val
            best_partition = comps_and_weights

    return sorted(best_partition, key=lambda x: x[1], reverse=not maximize_min)


def _bfs_parents_order(
    T: nx.Graph,
) -> tuple[Hashable, dict[Hashable, Hashable | None], list[Hashable]]:
    """Root T at an arbitrary node via BFS; return (root, parent, order).

    ``parent`` maps each node to its BFS parent (None for the root) and
    ``order`` lists nodes root-first in BFS discovery order, so iterating
    ``reversed(order)`` visits children before parents.
    """
    root = next(iter(T.nodes))
    # bfs_predecessors yields (node, parent) pairs in BFS discovery order.
    pairs = list(nx.bfs_predecessors(T, root))
    parent = {root: None, **dict(pairs)}  # root has no parent
    order = [root, *(v for v, _ in pairs)]  # root-first BFS order
    return root, parent, order


def _greedy_partition_bounds(
    T: nx.Graph, q: int, weight: str = "weight"
) -> tuple[float, float]:
    """Return a quick upper bound on OPT_minmax and lower bound on OPT_maxmin.

    The idea is simple: make any valid partition of the tree into
    `q` parts, then read off the heaviest and lightest pieces.
    Because no split can do better than the best possible split, the heaviest
    piece we see is at least as heavy as OPT_minmax, and the lightest piece we
    see is at most as light as OPT_maxmin.

    To split the tree we use a two-pass greedy strategy:

    Pass 1 — cut heavy subtrees first.
        Target weight per piece = total / q.  Walk from leaves
        toward the root, keeping a running subtree weight for each node.
        Whenever a subtree's weight reaches the target, snip it off as a
        finished piece and stop growing it.  Repeat until q-1
        pieces have been snipped.

    Pass 2 — peel leaves if Pass 1 came up short.
        If Pass 1 didn't snip off q-1 subtrees (e.g. many nodes
        have tiny weights so no subtree ever hit the target), peel individual
        leaf nodes off the part of the tree still attached to the root, one at
        a time, until q-1 total cuts are reached.  Each peeled
        leaf becomes its own single-node piece.

    After both passes, whatever is still attached to the root is the last
    piece.  That gives exactly q pieces total.

    Returns ``(max_piece_weight, min_piece_weight)`` for this split.
    """
    assert 1 <= q <= len(T), f"q={q} out of range for tree with {len(T)} nodes"
    W_node = {v: T.nodes[v].get(weight, 1) for v in T.nodes}
    total = sum(W_node.values())
    threshold = total / q  # target weight per piece

    root, parent, bfs_order = _bfs_parents_order(T)

    # --- Pass 1: cut heavy subtrees ---
    # Walk leaves-first.  Each node starts with just its own weight; as we
    # finish each child, add its accumulated weight to its parent.  When a
    # node's accumulated weight hits the target, snip it off.
    subtree_w = dict(W_node)  # accumulates as children roll up into parents
    piece_weights = []  # weight of each finished piece
    cut_nodes: set = set()  # root of each snipped-off subtree
    cuts = 0
    for v in reversed(bfs_order):
        p = parent[v]
        if p is None:  # root has no parent edge to cut
            continue
        if subtree_w[v] >= threshold and cuts < q - 1:
            # This subtree is heavy enough — snip it off as a finished piece.
            piece_weights.append(subtree_w[v])
            cut_nodes.add(v)
            cuts += 1
        else:
            # Not heavy enough yet — roll this subtree's weight into the parent.
            subtree_w[p] += subtree_w[v]

    # After Pass 1, everything not yet snipped is still attached to the root.
    # subtree_w[root] now holds the combined weight of that remaining part.
    root_weight = subtree_w[root]
    remaining_cuts = q - 1 - cuts

    # --- Pass 2: peel leaves if Pass 1 didn't make enough cuts ---
    # This only runs if Pass 1 ran out of heavy-enough subtrees before
    # reaching q-1 cuts.  We peel one leaf at a time (smallest pieces first)
    # until we have made q-1 cuts in total.
    if remaining_cuts > 0:
        # Count how many uncut children each uncut node has.  A node is a
        # current leaf of the remaining tree when that count reaches zero.
        # Only count v→parent edges where both v and its parent are uncut,
        # so that already-snipped subtrees don't confuse the leaf detection.
        has_uncut_child = {v: 0 for v in bfs_order if v not in cut_nodes}
        for v in bfs_order:
            p = parent[v]
            if v not in cut_nodes and p is not None and p not in cut_nodes:
                has_uncut_child[p] = has_uncut_child.get(p, 0) + 1

        made = 0
        for v in reversed(bfs_order):  # children before parents → peel leaves first
            if made == remaining_cuts:
                break
            if v in cut_nodes or v == root:
                continue
            if has_uncut_child.get(v, 0) == 0:  # v is currently a leaf
                # Peel v off as a single-node piece.
                piece_weights.append(W_node[v])
                root_weight -= W_node[v]
                cut_nodes.add(v)
                p = parent[v]
                if p is not None and p not in cut_nodes:
                    # Parent lost a child and may now itself be a leaf.
                    has_uncut_child[p] -= 1
                made += 1

    # Everything still attached to the root is the final piece.
    piece_weights.append(root_weight)
    return max(piece_weights), min(piece_weights)


def _bottleneck_node_lb_minmax(T: nx.Graph, q: int, weight: str = "weight") -> float:
    """Return a lower bound on OPT_minmax using the bottleneck-node argument.

    A partition into `q` parts cuts exactly q-1
    edges.  For a node v with degree d, at most q-1 of those
    cuts can be on edges touching v.  If d > q-1, then at least
    d-(q-1) of v's branches cannot be cut and must stay in the
    same piece as v.  The best we can do is leave the lightest branches
    attached, so the piece containing v weighs at least:

        W_node[v] + (sum of the d-(q-1) lightest branch weights)

    The heaviest piece in any valid partition is at least this large.
    Checking every node and taking the max gives a lower bound on OPT_minmax.
    """
    W_node = {v: T.nodes[v].get(weight, 1) for v in T.nodes}

    # Root the tree and compute each node's subtree weight.
    # For a neighbor u of v: if u is v's child, the branch toward u weighs
    # sub_w[u]; if u is v's parent, the branch toward u weighs W_total - sub_w[v].
    root, parent, bfs_order = _bfs_parents_order(T)
    sub_w = dict(W_node)
    for v in reversed(bfs_order):
        p = parent[v]
        if p is not None:
            sub_w[p] += sub_w[v]

    W_total = sub_w[root]

    # Trivial lower bound: pieces average W_total/q, so the max
    # is at least that.
    best_lb = W_total / q

    for v in T.nodes:
        d = T.degree(v)
        # Number of branches that must stay in v's piece because we run out of cuts.
        remaining = d - (q - 1)
        if remaining <= 0:
            continue  # all branches can be cut; no forced contribution from v

        # Collect the weight of each branch hanging off v.
        adj_weights = []
        for u in T.neighbors(v):
            if parent.get(u) == v:
                # u is a child: its branch weighs sub_w[u].
                adj_weights.append(sub_w[u])
            else:
                # u is v's parent: the branch on that side weighs W_total - sub_w[v].
                adj_weights.append(W_total - sub_w[v])

        # Keep the lightest branches attached (best case for minimizing max piece).
        adj_weights.sort()
        lb = W_node[v] + sum(adj_weights[:remaining])
        best_lb = max(best_lb, lb)

    return best_lb


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
    # Bristles
    for i in range(1, s + 1):
        G.add_edge(0, i)
    # Start of handle
    if a > 0:
        G.add_edge(0, s + 1)
    # Rest of handle
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
# Tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_directed_raises(self) -> None:
        """Both functions must reject directed graphs.

        Tree partitioning is defined only for undirected trees.  Passing a
        DiGraph should raise NetworkXNotImplemented immediately, before any
        computation starts.
        """
        G = nx.DiGraph([(0, 1), (1, 2)])
        with pytest.raises(nx.NetworkXNotImplemented):
            _minmax(G, 2)
        with pytest.raises(nx.NetworkXNotImplemented):
            _maxmin(G, 2)

    def test_multigraph_raises(self) -> None:
        """Both functions must reject multigraphs.

        Multiple edges between the same pair of nodes are not meaningful for
        tree partitioning.  Passing a MultiGraph should raise
        NetworkXNotImplemented immediately, before any computation starts.
        """
        G = nx.MultiGraph([(0, 1), (1, 2)])
        with pytest.raises(nx.NetworkXNotImplemented):
            _minmax(G, 2)
        with pytest.raises(nx.NetworkXNotImplemented):
            _maxmin(G, 2)

    def test_dispatchable_graph_argument_is_T(self) -> None:
        """The dispatcher must register the graph parameter under its real
        name (T).  With the default ("G"), keyword calls and backend graph
        conversion both fail whenever any backend is installed."""
        assert min_max_tree_partition.graphs == {"T": 0}
        assert max_min_tree_partition.graphs == {"T": 0}

    def test_dispatchable_declares_missing_attr_default_1(self) -> None:
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

    def test_graph_passable_by_keyword(self) -> None:
        G = nx.path_graph(4)
        assert len(min_max_tree_partition(T=G, q=2)) == 2
        assert len(max_min_tree_partition(T=G, q=2)) == 2

    def test_not_a_tree_cycle(self) -> None:
        G = nx.cycle_graph(4)
        with pytest.raises(nx.NotATree):
            _minmax(G, 2)
        with pytest.raises(nx.NotATree):
            _maxmin(G, 2)

    def test_not_a_tree_disconnected(self) -> None:
        G = nx.path_graph(3)
        G.add_node(99)
        with pytest.raises(nx.NotATree):
            _minmax(G, 2)

    def test_empty_graph_raises_pointless_concept(self) -> None:
        """Documented contract: an empty graph raises
        NetworkXPointlessConcept (from nx.is_tree), not NotATree."""
        G = nx.Graph()
        with pytest.raises(nx.NetworkXPointlessConcept):
            _minmax(G, 1)
        with pytest.raises(nx.NetworkXPointlessConcept):
            _maxmin(G, 1)

    @pytest.mark.parametrize("number_of_pieces", [0, -1, 5])
    def test_invalid_number_of_pieces_path4(self, number_of_pieces) -> None:
        """number_of_pieces must be between 1 and the number of nodes (inclusive).

        number_of_pieces=0 and number_of_pieces=-1 are below the minimum;
        number_of_pieces=5 exceeds the 4-node tree's node count.
        All three should raise NetworkXError.
        """
        G = nx.path_graph(4)
        with pytest.raises(nx.NetworkXError):
            _minmax(G, number_of_pieces)
        with pytest.raises(nx.NetworkXError):
            _maxmin(G, number_of_pieces)

    @pytest.mark.parametrize("bad_number_of_pieces", [2.5, 1.0, "2", None, True, False])
    def test_non_integer_number_of_pieces_raises(
        self, bad_number_of_pieces: Any
    ) -> None:
        """Number of pieces must be an integer"""
        G = nx.path_graph(5)
        with pytest.raises(nx.NetworkXError, match="integer"):
            _minmax(G, bad_number_of_pieces)
        with pytest.raises(nx.NetworkXError, match="integer"):
            _maxmin(G, bad_number_of_pieces)

    @pytest.mark.parametrize("fn", [min_max_tree_partition, max_min_tree_partition])
    @pytest.mark.parametrize("kwarg", ["node_weight", "edge_weight"])
    def test_non_string_weight_param_raises(self, fn, kwarg: str) -> None:
        """node_weight and edge_weight must be strings (attribute names).

        Passing None or a non-string must raise NetworkXError before any
        computation starts, not silently fall back to a weight-blind result.
        """
        G = nx.path_graph(4)
        G.nodes[0]["weight"] = 100
        with pytest.raises(nx.NetworkXError, match="must be a string"):
            fn(G, 2, **{kwarg: None})
        with pytest.raises(nx.NetworkXError, match="must be a string"):
            fn(G, 2, **{kwarg: 5})

    @pytest.mark.parametrize("bad_weight", [-1, -0.5, 0])
    def test_negative_weight_raises(self, bad_weight: float | int) -> None:
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {2: bad_weight}, "weight")
        with pytest.raises(nx.NetworkXError, match="> 0"):
            _minmax(G, 2, node_weight="weight")
        with pytest.raises(nx.NetworkXError, match="> 0"):
            _maxmin(G, 2, node_weight="weight")

    @pytest.mark.parametrize("bad_weight", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_weight_raises(self, bad_weight: float) -> None:
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {1: bad_weight}, "weight")
        with pytest.raises(nx.NetworkXError, match="non-finite"):
            _minmax(G, 2, node_weight="weight")
        with pytest.raises(nx.NetworkXError, match="non-finite"):
            _maxmin(G, 2, node_weight="weight")

    @pytest.mark.parametrize("weight_function", ["vertex_weight_sum", "mixed_sum"])
    def test_non_numeric_node_weight_raises(self, weight_function: str) -> None:
        """Node weight attribute values must be numbers; a string raises NetworkXError."""
        G = nx.path_graph(3)
        G.nodes[1]["weight"] = "heavy"
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _minmax(G, 2, weight_function=weight_function)
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _maxmin(G, 2, weight_function=weight_function)

    @pytest.mark.parametrize("weight_function", ["edge_weight_sum", "mixed_sum"])
    def test_non_numeric_edge_weight_raises(self, weight_function: str) -> None:
        G = nx.path_graph(3)
        G.edges[0, 1]["weight"] = "long"
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _minmax(G, 2, weight_function=weight_function)
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _maxmin(G, 2, weight_function=weight_function)

    def test_none_node_weight_raises(self) -> None:
        G = nx.path_graph(3)
        G.nodes[1]["weight"] = None
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _minmax(G, 2)
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _maxmin(G, 2)

    @pytest.mark.parametrize("bad_weight", ["3", "1e3", True])
    def test_numeric_looking_node_weight_raises(self, bad_weight: Any) -> None:
        """Numeric-looking strings and bools are rejected as node weights.

        Only actual int/float values are accepted; strings like "3" or "1e3"
        must raise NetworkXError, and so must True/False.
        """
        G = nx.path_graph(4)
        G.nodes[1]["weight"] = bad_weight
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _minmax(G, 2)
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _maxmin(G, 2)

    @pytest.mark.parametrize("bad_weight", ["2", True])
    def test_numeric_looking_edge_weight_raises(self, bad_weight: Any) -> None:
        G = nx.path_graph(4)
        G.edges[1, 2]["weight"] = bad_weight
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _minmax(G, 2, weight_function="edge_weight_sum")
        with pytest.raises(nx.NetworkXError, match="non-numeric"):
            _maxmin(G, 2, weight_function="edge_weight_sum")

    @pytest.mark.parametrize("fn", [min_max_tree_partition, max_min_tree_partition])
    def test_huge_non_integral_weight_raises_networkx_error(self, fn) -> None:
        """A Fraction weight too large to represent as a float raises NetworkXError."""
        from fractions import Fraction

        G = nx.path_graph(3)
        G.nodes[1]["weight"] = Fraction(10**400, 3)
        with pytest.raises(nx.NetworkXError, match="too large"):
            fn(G, 2)

    def test_mixed_huge_int_and_float_weights_raise(self) -> None:
        """Integers beyond float range are exact in all-integer trees, but
        cannot be summed with float weights; that mix raises clearly
        instead of leaking an OverflowError."""
        G = nx.path_graph(3)
        nx.set_node_attributes(G, {0: 10**400, 1: 1.5, 2: 1.5}, "weight")
        with pytest.raises(nx.NetworkXError, match="exactly representable"):
            _minmax(G, 2)
        with pytest.raises(nx.NetworkXError, match="exactly representable"):
            _maxmin(G, 2)

    def test_mixed_float_and_inexact_int_weights_raise(self) -> None:
        """Integers above 2**53 mixed with floats raise NetworkXError.

        Such integers cannot be exactly represented as floats, so the
        float bisection grid cannot reliably separate candidates.
        """
        G = nx.Graph([(0, 1), (0, 2)])
        nx.set_node_attributes(G, {0: 4.0, 1: 2**60 + 4, 2: 2**60 + 1}, "weight")
        with pytest.raises(nx.NetworkXError, match="exactly representable"):
            _minmax(G, 2)
        with pytest.raises(nx.NetworkXError, match="exactly representable"):
            _maxmin(G, 2)

    def test_mixed_float_and_exact_representable_int_ok(self) -> None:
        """Large integers that ARE exact doubles (e.g. 2**60) mix fine with
        floats: everything lands on the float grid, so the documented
        float-summation exactness holds."""
        G = nx.Graph([(0, 1), (0, 2)])
        nx.set_node_attributes(G, {0: 4.0, 1: 2**60, 2: 2**60}, "weight")
        for fn in [min_max_tree_partition, max_min_tree_partition]:
            parts = fn(G, 2)
            assert len(parts) == 2
            assert set().union(*parts) == set(G.nodes)


class TestEdgeCases:
    def test_single_node_number_of_pieces_1(self) -> None:
        G = nx.Graph()
        G.add_node(0, weight=5)
        p = _minmax(G, 1)
        assert p == [(frozenset({0}), 5)]
        p = _maxmin(G, 1)
        assert p == [(frozenset({0}), 5)]

    def test_two_nodes_number_of_pieces_1(self) -> None:
        G = nx.path_graph(2)
        nx.set_node_attributes(G, {0: 3, 1: 7}, "weight")
        p = _minmax(G, 1, node_weight="weight")
        assert len(p) == 1
        assert p[0][1] == 10

    def test_two_nodes_number_of_pieces_2(self) -> None:
        G = nx.path_graph(2)
        nx.set_node_attributes(G, {0: 3, 1: 7}, "weight")
        p_mm = _minmax(G, 2, node_weight="weight")
        assert _max_weight(p_mm) == 7
        p_mx = _maxmin(G, 2, node_weight="weight")
        assert _min_weight(p_mx) == 3

    def test_number_of_pieces_equals_n_all_singletons(self) -> None:
        """With number_of_pieces = n every node is its own component."""
        G = nx.path_graph(4)
        p = _minmax(G, 4)
        assert _is_valid_partition(G, p, 4)
        assert _max_weight(p) == 1

    def test_number_of_pieces_1_whole_tree(self) -> None:
        G = nx.balanced_tree(2, 3)
        p = _minmax(G, 1)
        assert len(p) == 1
        total = len(G)
        assert p[0][1] == total


class TestPathGraphs:
    """Tests for path graphs P_n under various weight functions and partition counts."""

    @pytest.mark.parametrize(
        "n, number_of_pieces",
        [
            # small — divisible (max = min = n / q)
            (6, 2),
            (6, 3),
            (9, 3),
            (12, 4),
            # small — non-divisible (max = ceil, min = floor)
            (7, 2),
            (7, 3),
            (10, 4),
            (11, 3),
            (13, 5),
            # medium
            (50, 7),
            (99, 4),
            (100, 3),
            # large — divisible
            (1000, 4),
            (5000, 5),
            (15000, 3),
            # large — non-divisible
            (100, 7),
            (500, 2),
            (1000, 9),
            (5000, 3),
            (20000, 4),
        ],
    )
    def test_path_unit_weights(self, n: int, number_of_pieces: int) -> None:
        """Every piece is a contiguous subpath; the most even division puts
        r = n mod q pieces at ceil(n/q) nodes and the rest at floor(n/q):

            OPT_minmax = ceil(n / q)
            OPT_maxmin = floor(n / q)
        """
        G = nx.path_graph(n)
        expected_minmax = math.ceil(n / number_of_pieces)
        expected_maxmin = math.floor(n / number_of_pieces)

        p_mm = _minmax(G, number_of_pieces)
        assert _is_valid_partition(G, p_mm, number_of_pieces)
        assert _is_sorted_descending(p_mm)
        assert _max_weight(p_mm) == expected_minmax

        p_mx = _maxmin(G, number_of_pieces)
        assert _is_valid_partition(G, p_mx, number_of_pieces)
        assert _is_sorted_ascending(p_mx)
        assert _min_weight(p_mx) == expected_maxmin

    @pytest.mark.parametrize(
        "node_weights, number_of_pieces, expected_minmax, expected_maxmin",
        [
            ([3, 1, 1, 3], 2, 4, None),  # symmetric heavy endpoints
            ([1, 5, 5, 1], 2, None, 6),  # heavy interior stays together
            ([1, 1, 10, 1, 1], 2, 12, None),  # single dominant node
        ],
    )
    def test_path_weighted(
        self,
        node_weights: list[int],
        number_of_pieces: int,
        expected_minmax: int | None,
        expected_maxmin: int | None,
    ) -> None:
        G = nx.path_graph(len(node_weights))
        nx.set_node_attributes(G, dict(enumerate(node_weights)), "weight")
        if expected_minmax is not None:
            p = _minmax(G, number_of_pieces, node_weight="weight")
            assert _is_valid_partition(G, p, number_of_pieces)
            assert _max_weight(p) == expected_minmax
        if expected_maxmin is not None:
            p = _maxmin(G, number_of_pieces, node_weight="weight")
            assert _is_valid_partition(G, p, number_of_pieces)
            assert _min_weight(p) == expected_maxmin

    @pytest.mark.parametrize(
        "m, a, b",
        [
            # small
            (5, 2, 3),
            (10, 1, 4),
            # large
            (20, 3, 5),
            (100, 1, 2),
            (500, 3, 7),
        ],
    )
    def test_path_alternating_weights(self, m: int, a: int, b: int) -> None:
        """P_{2m} with alternating weights a, b, a, b, ...  and q = m pieces.

        The only cuts that don't split an (a, b) pair are between positions
        (2i-1, 2i) for i = 1, ..., m-1.  Taking all m-1 such cuts gives
        m pieces each weighing a+b, so OPT_minmax = OPT_maxmin = a+b.
        """
        G = nx.path_graph(2 * m)
        nx.set_node_attributes(
            G, {i: (a if i % 2 == 0 else b) for i in range(2 * m)}, "weight"
        )

        p_mm = _minmax(G, m, node_weight="weight")
        assert _is_valid_partition(G, p_mm, m)
        assert _max_weight(p_mm) == a + b

        p_mx = _maxmin(G, m, node_weight="weight")
        assert _is_valid_partition(G, p_mx, m)
        assert _min_weight(p_mx) == a + b


class TestStarGraphs:
    """Tests for star graphs S_n under various weight configurations and partition counts."""

    @pytest.mark.parametrize(
        "n, number_of_pieces",
        [
            # small — q = 2
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 2),
            # small — larger q
            (5, 3),
            (5, 4),
            (6, 5),
            (6, 6),
            # medium
            (20, 5),
            (50, 10),
            (100, 3),
            # large
            (500, 2),
            (2000, 5),
            (10000, 10),
            (10000, 25),
        ],
    )
    def test_star_unit_weights(self, n: int, number_of_pieces: int) -> None:
        """Any q-partition cuts exactly q-1 edges.  All edges are incident to
        the center, so each cut separates exactly one leaf.  After q-1 cuts:
            - q-1 singleton leaf pieces (weight 1 each)
            - one center piece with n-(q-1) leaves plus the center = n-q+2 nodes

        The center piece is always the heaviest (n-q+2 >= 1 for q <= n), so:

            OPT_minmax = n - q + 2   (2 <= q <= n)
            OPT_maxmin = 1            (q >= 2; every leaf is a singleton)
        """
        G = nx.star_graph(n)

        p_mm = _minmax(G, number_of_pieces)
        assert _is_valid_partition(G, p_mm, number_of_pieces)
        assert _max_weight(p_mm) == n - number_of_pieces + 2

        p_mx = _maxmin(G, number_of_pieces)
        assert _is_valid_partition(G, p_mx, number_of_pieces)
        assert _min_weight(p_mx) == 1

    def test_star_all_singletons(self) -> None:
        """q = n+1: every node is its own piece; OPT_minmax = 1."""
        n = 4
        G = nx.star_graph(n)
        number_of_pieces = n + 1
        p = _minmax(G, number_of_pieces)
        assert _is_valid_partition(G, p, number_of_pieces)
        assert _max_weight(p) == 1

    def test_star_weighted(self) -> None:
        """Heavy center forces all leaves into the center piece for min-max."""
        G = nx.star_graph(3)
        nx.set_node_attributes(G, {0: 10, 1: 1, 2: 1, 3: 1}, "weight")

        p_mm = _minmax(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p_mm, 2)
        assert _max_weight(p_mm) == 12

        p_mx = _maxmin(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p_mx, 2)
        assert _min_weight(p_mx) == 1


class TestBalancedTrees:
    """Tests for balanced k-ary trees under various branching factors and partition counts."""

    @pytest.mark.parametrize(
        "k, h",
        [
            # small
            (2, 2),
            (2, 3),
            (3, 2),
            # larger
            (2, 4),
            (2, 5),
            (3, 3),
            (3, 4),
        ],
    )
    def test_kary_tree_two_pieces(self, k: int, h: int) -> None:
        """Balanced k-ary tree of height h has total nodes (k^(h+1)-1)/(k-1).
        No clean closed form exists for general q.  For q = 2, the optimal cut
        severs one of the root's child edges, splitting the tree into:
            - a child subtree of  (k^h - 1) / (k-1)  nodes
            - the rest:           k^h                  nodes

        (Proof: total - child_size = (k^(h+1)-1)/(k-1) - (k^h-1)/(k-1) = k^h.)
        Every child subtree has the same size, so no other single cut produces
        a more balanced 2-way split.  Therefore:

            OPT_minmax(q=2) = k^h
            OPT_maxmin(q=2) = (k^h - 1) // (k - 1)
        """
        G = nx.balanced_tree(k, h)
        expected_minmax = k**h
        expected_maxmin = (k**h - 1) // (k - 1)

        p_mm = _minmax(G, 2)
        assert _is_valid_partition(G, p_mm, 2)
        assert _max_weight(p_mm) == expected_minmax

        p_mx = _maxmin(G, 2)
        assert _is_valid_partition(G, p_mx, 2)
        assert _min_weight(p_mx) == expected_maxmin

    def test_binary_tree_height2_four_pieces(self) -> None:
        """q=4 on a 7-node binary tree: no closed form, check validity only."""
        G = nx.balanced_tree(2, 2)
        p_mm = _minmax(G, 4)
        assert _is_valid_partition(G, p_mm, 4)
        assert _max_weight(p_mm) >= 2  # pigeonhole: at least one piece >= ceil(7/4)


class TestCaterpillarTrees:
    """Uniform caterpillar: a path (spine) of s nodes, each with l pendant
    leaves.  Total nodes = s * (l+1).

    Setting q = s, the s-1 spine-to-spine edges are the only inter-component
    edges available, so the unique q-1 = s-1 cuts place exactly one spine
    node and its l leaves into each piece.  Every piece weighs l+1, giving:

        OPT_minmax = OPT_maxmin = l + 1   (with q = s)

    This is also the pigeonhole lower bound (total/q = l+1), so the optimum
    is exact.
    """

    @pytest.mark.parametrize(
        "spine, leaves_per_node",
        [
            # small
            (3, 1),
            (4, 1),
            (3, 2),
            (4, 2),
            # medium
            (5, 3),
            (10, 2),
            (8, 4),
            # large
            (50, 10),
            (100, 5),
            (200, 3),
        ],
    )
    def test_caterpillar_uniform_partition(
        self, spine: int, leaves_per_node: int
    ) -> None:
        G = _caterpillar(spine, leaves_per_node)
        number_of_pieces = spine
        expected = leaves_per_node + 1

        p_mm = _minmax(G, number_of_pieces)
        assert _is_valid_partition(G, p_mm, number_of_pieces)
        assert _max_weight(p_mm) == expected

        p_mx = _maxmin(G, number_of_pieces)
        assert _is_valid_partition(G, p_mx, number_of_pieces)
        assert _min_weight(p_mx) == expected


class TestExplicitWeightedTrees:
    def test_y_shaped_tree(self) -> None:
        G = nx.Graph([(0, 1), (0, 2), (0, 3)])
        nx.set_node_attributes(G, {0: 4, 1: 2, 2: 2, 3: 2}, "weight")
        p_mm = _minmax(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p_mm, 2)
        assert _max_weight(p_mm) == 8

        p_mx = _maxmin(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p_mx, 2)
        assert _min_weight(p_mx) == 2

    def test_unequal_arms_y_tree_number_of_pieces_3(self) -> None:
        G = nx.Graph([(0, 1), (0, 2), (0, 3)])
        nx.set_node_attributes(G, {0: 1, 1: 2, 2: 3, 3: 6}, "weight")
        p_mm = _minmax(G, 3, node_weight="weight")
        assert _is_valid_partition(G, p_mm, 3)
        assert _max_weight(p_mm) == 6

        p_mx = _maxmin(G, 3, node_weight="weight")
        assert _is_valid_partition(G, p_mx, 3)
        assert _min_weight(p_mx) == 3

    def test_missing_weight_attribute_defaults_to_1(self) -> None:
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {0: 3, 2: 3}, "weight")
        p = _minmax(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p, 2)
        assert _max_weight(p) == 4

    def test_custom_weight_attribute_name(self) -> None:
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {0: 1, 1: 1, 2: 1, 3: 1}, "cost")
        p = _minmax(G, 2, node_weight="cost")
        assert _is_valid_partition(G, p, 2, node_weight="cost")
        assert _max_weight(p) == 2


class TestReturnFormat:
    """The raw public API returns a bare partition: a list of frozensets
    sorted by component weight (no weights in the return value)."""

    def test_returns_list_of_frozensets(self) -> None:
        G = nx.path_graph(6)
        for fn in [min_max_tree_partition, max_min_tree_partition]:
            p = fn(G, 3)
            assert isinstance(p, list)
            assert all(isinstance(nodes, frozenset) for nodes in p)

    def test_min_max_sorted_descending_by_weight(self) -> None:
        G = nx.path_graph(9)
        nx.set_node_attributes(G, {v: v + 1 for v in G}, "weight")
        p = min_max_tree_partition(G, 3, node_weight="weight")
        weights = [sum(v + 1 for v in nodes) for nodes in p]
        assert weights == sorted(weights, reverse=True)

    def test_max_min_sorted_ascending_by_weight(self) -> None:
        G = nx.path_graph(9)
        nx.set_node_attributes(G, {v: v + 1 for v in G}, "weight")
        p = max_min_tree_partition(G, 3, node_weight="weight")
        weights = [sum(v + 1 for v in nodes) for nodes in p]
        assert weights == sorted(weights)

    def test_number_of_pieces_1_returns_single_frozenset(self) -> None:
        G = nx.path_graph(5)
        assert min_max_tree_partition(G, 1) == [frozenset(G.nodes())]
        assert max_min_tree_partition(G, 1) == [frozenset(G.nodes())]


class TestRandomSampling:
    """Generate random labeled trees and verify the public functions match the
    internal brute-force reference."""

    SEEDS = [0, 1, 5, 7, 9, 13, 17, 31, 42, 50, 77, 99, 100, 123, 200, 256]

    @pytest.mark.parametrize("seed", SEEDS)
    def test_random_min_max_matches_brute_force(self, seed: int) -> None:
        rng = random.Random(seed)
        n = rng.randint(3, 11)
        T = nx.random_labeled_tree(n, seed=seed)
        for v in T.nodes:
            T.nodes[v]["weight"] = rng.randint(1, 10)
        number_of_pieces = rng.randint(1, min(n, 4))

        p_public = _minmax(T, number_of_pieces, node_weight="weight")
        p_ref = _brute_force_partition(T, number_of_pieces, False)

        assert _is_valid_partition(T, p_public, number_of_pieces)
        assert _is_sorted_descending(p_public)
        assert _max_weight(p_public) == _max_weight(p_ref)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_random_max_min_matches_brute_force(self, seed: int) -> None:
        rng = random.Random(seed)
        n = rng.randint(3, 11)
        T = nx.random_labeled_tree(n, seed=seed)
        for v in T.nodes:
            T.nodes[v]["weight"] = rng.randint(1, 10)
        number_of_pieces = rng.randint(1, min(n, 4))

        p_public = _maxmin(T, number_of_pieces, node_weight="weight")
        p_ref = _brute_force_partition(T, number_of_pieces, True)

        assert _is_valid_partition(T, p_public, number_of_pieces)
        assert _is_sorted_ascending(p_public)
        assert _min_weight(p_public) == _min_weight(p_ref)

    @pytest.mark.parametrize(
        "n,number_of_pieces", [(5, 2), (6, 3), (8, 4), (10, 3), (12, 4)]
    )
    def test_random_trees_various_sizes(self, n: int, number_of_pieces: int) -> None:
        for seed in range(5):
            T = nx.random_labeled_tree(n, seed=seed)
            p_mm = _minmax(T, number_of_pieces)
            p_mx = _maxmin(T, number_of_pieces)
            assert _is_valid_partition(T, p_mm, number_of_pieces)
            assert _is_valid_partition(T, p_mx, number_of_pieces)
            assert _min_weight(p_mx) <= n / number_of_pieces
            assert _max_weight(p_mm) >= n / number_of_pieces


class TestFloatWeightsVsBruteForce:
    """Verify the binary-search algorithm agrees with brute-force on small
    random trees with float weights (integer weights are covered by
    TestRandomSampling)."""

    @pytest.mark.parametrize("seed", [0, 1, 5, 9, 17])
    def test_min_max_float_weights(self, seed: int) -> None:
        rng = random.Random(seed)
        n = rng.randint(3, 11)
        T = nx.random_labeled_tree(n, seed=seed)
        for v in T.nodes:
            T.nodes[v]["weight"] = round(rng.uniform(0.5, 5.0), 2)
        number_of_pieces = rng.randint(1, min(n, 4))

        p_default = _minmax(T, number_of_pieces, node_weight="weight")
        p_brute = _brute_force_partition(T, number_of_pieces, False)

        assert _is_valid_partition(T, p_default, number_of_pieces)
        assert math.isclose(_max_weight(p_default), _max_weight(p_brute), rel_tol=1e-9)

    @pytest.mark.parametrize("seed", [0, 1, 5, 9, 17])
    def test_max_min_float_weights(self, seed: int) -> None:
        rng = random.Random(seed)
        n = rng.randint(3, 11)
        T = nx.random_labeled_tree(n, seed=seed)
        for v in T.nodes:
            T.nodes[v]["weight"] = round(rng.uniform(0.5, 5.0), 2)
        number_of_pieces = rng.randint(1, min(n, 4))

        p_default = _maxmin(T, number_of_pieces, node_weight="weight")
        p_brute = _brute_force_partition(T, number_of_pieces, True)

        assert _is_valid_partition(T, p_default, number_of_pieces)
        assert math.isclose(_min_weight(p_default), _min_weight(p_brute), rel_tol=1e-9)


class TestLargeAnalytical:
    """Graph families with closed-form optima not covered by the main analytic classes."""

    @pytest.mark.parametrize("m", [500, 2000, 10000])
    def test_double_star_number_of_pieces_2_large(self, m) -> None:
        """Double star: two hubs each with m leaves, connected by a bridge edge.

        The unique optimal 2-cut is the bridge.  Each half weighs m+1 (one hub
        plus m leaves), so OPT_minmax = OPT_maxmin = m+1.
        """
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

    def test_extreme_weight_ratio_minmax_exact(self) -> None:
        """When node weights span ~12 orders of magnitude, the algorithm still
        finds the exact optimal partition (integer bisection, no float tolerance)."""
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {0: 10**12, 1: 3, 2: 3, 3: 10**12}, "weight")
        p = _minmax(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p, 2)
        assert _max_weight(p) == 10**12 + 3

    def test_extreme_weight_ratio_maxmin_exact(self) -> None:
        G = nx.path_graph(3)
        nx.set_node_attributes(G, {0: 10**10, 1: 1, 2: 1}, "weight")
        p = _maxmin(G, 2, node_weight="weight")
        assert _is_valid_partition(G, p, 2)
        assert _min_weight(p) == 2

    def test_extreme_weight_ratio_float_maxmin_exact(self) -> None:
        """With float weights spanning ~10 orders of magnitude, the algorithm
        still finds the exact optimal partition (float-grid bisection over
        IEEE-754 doubles)."""
        G = nx.path_graph(3)
        nx.set_node_attributes(G, {0: 1e10, 1: 1.0, 2: 1.0}, "weight")
        p = _maxmin(G, 2, node_weight="weight")
        assert _min_weight(p) == 2.0

    @pytest.mark.parametrize("fn", ["minmax", "maxmin"])
    def test_arbitrary_precision_integer_weights(self, fn) -> None:
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

    def test_tiny_float_weights_exact(self) -> None:
        """With node weights at 1e-15 (well below any absolute epsilon), the
        algorithm still returns the exact optimal partition."""
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
        """Assert that the optimal values returned by both algorithms lie within
        proven analytical bounds and within the greedy algorithm's bounds.

        For min-max, the optimum must satisfy:
            lb_minmax <= OPT_minmax <= greedy_max
        where lb_minmax = max(W/q, max_node_weight, bottleneck_node_lb).

        For max-min, the optimum must satisfy:
            greedy_min <= OPT_maxmin <= ub_maxmin
        where ub_maxmin = min(W/q, (W - max_node_weight) / (q-1)).
        """
        W_node = {v: T.nodes[v].get(weight, 1) for v in T.nodes}
        W = sum(W_node.values())
        max_w = max(W_node.values())

        # Lower bound for min-max: the heaviest piece is at least W/q (pigeonhole),
        # at least max_node_weight (a node can't be split), and at least the
        # bottleneck-node bound (high-degree nodes force branches to share a piece).
        lb_minmax = max(
            W / q,
            max_w,
            _bottleneck_node_lb_minmax(T, q, weight=weight),
        )

        # Upper bound for max-min: the lightest piece is at most W/q (pigeonhole),
        # and at most (W - max_node_weight) / (q-1) because the heaviest node
        # occupies one piece, leaving at most that average for the rest.
        ub_maxmin = W / q
        if q >= 2:
            ub_maxmin = min(ub_maxmin, (W - max_w) / (q - 1))

        # Greedy algorithm bounds: used as upper bound for min-max and lower
        # bound for max-min (the greedy solution is always feasible but not
        # necessarily optimal).
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
        "n,number_of_pieces",
        [
            (500, 3),
            (1000, 5),
            (5000, 4),
            (10000, 7),
        ],
    )
    def test_path_unit_weights(self, n: int, number_of_pieces: int) -> None:
        self._check_bounds(nx.path_graph(n), number_of_pieces)

    @pytest.mark.parametrize(
        "n,number_of_pieces",
        [
            (500, 3),
            (2000, 5),
            (8000, 10),
        ],
    )
    def test_star_unit_weights(self, n: int, number_of_pieces: int) -> None:
        self._check_bounds(nx.star_graph(n), number_of_pieces)

    @pytest.mark.parametrize(
        "k,h,number_of_pieces",
        [
            (2, 7, 3),
            (2, 9, 5),
            (3, 5, 4),
            (3, 6, 7),
        ],
    )
    def test_balanced_tree_unit_weights(
        self, k: int, h: int, number_of_pieces: int
    ) -> None:
        self._check_bounds(nx.balanced_tree(k, h), number_of_pieces)

    @pytest.mark.parametrize(
        "seed,n,number_of_pieces",
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
    def test_random_tree_unit_weights(
        self, seed: int, n: int, number_of_pieces: int
    ) -> None:
        T = nx.random_labeled_tree(n, seed=seed)
        self._check_bounds(T, number_of_pieces)

    @pytest.mark.parametrize(
        "seed,n,number_of_pieces",
        [
            (10, 300, 3),
            (11, 800, 5),
            (12, 2000, 4),
            (13, 5000, 6),
        ],
    )
    def test_random_tree_random_weights(
        self, seed: int, n: int, number_of_pieces: int
    ) -> None:
        rng = random.Random(seed)
        T = nx.random_labeled_tree(n, seed=seed)
        weights = {v: rng.randint(1, 20) for v in T.nodes}
        nx.set_node_attributes(T, weights, "weight")
        self._check_bounds(T, number_of_pieces, weight="weight")

    @pytest.mark.parametrize(
        "seed,n,number_of_pieces",
        [
            (20, 400, 4),
            (21, 1000, 5),
            (22, 3000, 8),
        ],
    )
    def test_random_tree_skewed_weights(
        self, seed: int, n: int, number_of_pieces: int
    ) -> None:
        rng = random.Random(seed)
        T = nx.random_labeled_tree(n, seed=seed)
        heavy = rng.choice(list(T.nodes))
        weights = {v: (n if v == heavy else 1) for v in T.nodes}
        nx.set_node_attributes(T, weights, "weight")
        self._check_bounds(T, number_of_pieces, weight="weight")

    @pytest.mark.parametrize(
        "k,L,case",
        [
            (3, 100, "2_pieces"),
            (5, 50, "2_pieces"),
            (4, 200, "2_pieces"),
            (4, 100, "k_pieces"),
            (6, 50, "k_pieces"),
            (3, 100, "k_plus_1_pieces"),
            (4, 80, "k_plus_1_pieces"),
        ],
    )
    def test_spider_graph(self, k: int, L: int, case: str) -> None:
        G = _spider_graph(k, L)
        assert len(G) == 1 + k * L

        if case == "2_pieces":
            number_of_pieces = 2
            expected_minmax = 1 + (k - 1) * L
            expected_maxmin = L
        elif case == "k_pieces":
            number_of_pieces = k
            expected_minmax = L + 1
            expected_maxmin = L
        else:
            n = 1 + k * L
            number_of_pieces = k + 1
            expected_minmax = math.ceil(n / number_of_pieces)
            expected_maxmin = n // number_of_pieces

        p_mm = _minmax(G, number_of_pieces)
        assert _is_valid_partition(G, p_mm, number_of_pieces)
        assert _max_weight(p_mm) == expected_minmax

        p_mx = _maxmin(G, number_of_pieces)
        assert _is_valid_partition(G, p_mx, number_of_pieces)
        assert _min_weight(p_mx) == expected_maxmin

    @pytest.mark.parametrize(
        "num_hubs,m",
        [(5, 10), (8, 20), (10, 5)],
    )
    def test_chain_of_stars_perfect(self, num_hubs: int, m: int) -> None:
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
    def test_chain_of_stars_one_fewer_cut(self, num_hubs: int, m: int) -> None:
        G = _chain_of_stars(num_hubs, m)
        number_of_pieces = num_hubs - 1

        p_mm = _minmax(G, number_of_pieces)
        assert _is_valid_partition(G, p_mm, number_of_pieces)
        assert _max_weight(p_mm) == 2 * (m + 1)

        p_mx = _maxmin(G, number_of_pieces)
        assert _is_valid_partition(G, p_mx, number_of_pieces)
        assert _min_weight(p_mx) == m + 1

    @pytest.mark.parametrize("s", [5, 10, 50, 100])
    def test_broom_perfect_number_of_pieces_2(self, s: int) -> None:
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
    def test_heavy_node_path_number_of_pieces_2(self, n: int, H: int) -> None:
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
    def test_comb_perfect_partition(self, s: int, d: int) -> None:
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
    def test_binomial_tree_perfect_partition(self, n: int, j: int) -> None:
        G = nx.binomial_tree(n)
        assert len(G) == 2**n
        number_of_pieces = 2**j
        expected = 2 ** (n - j)

        p_mm = _minmax(G, number_of_pieces)
        assert _is_valid_partition(G, p_mm, number_of_pieces)
        assert _max_weight(p_mm) == expected

        p_mx = _maxmin(G, number_of_pieces)
        assert _is_valid_partition(G, p_mx, number_of_pieces)
        assert _min_weight(p_mx) == expected

    @pytest.mark.parametrize(
        "k,L,number_of_pieces",
        [(4, 500, 2), (5, 200, 3), (3, 1000, 4)],
    )
    def test_spider_bounds_sanity(self, k: int, L: int, number_of_pieces: int) -> None:
        self._check_bounds(_spider_graph(k, L), number_of_pieces)

    @pytest.mark.parametrize(
        "num_hubs,m,number_of_pieces",
        [(10, 50, 5), (8, 30, 4), (6, 100, 3)],
    )
    def test_chain_of_stars_bounds_sanity(
        self, num_hubs: int, m: int, number_of_pieces: int
    ) -> None:
        self._check_bounds(_chain_of_stars(num_hubs, m), number_of_pieces)

    @pytest.mark.parametrize(
        "s,a,number_of_pieces",
        [(50, 100, 2), (30, 200, 2), (100, 50, 2)],
    )
    def test_broom_bounds_sanity(self, s: int, a: int, number_of_pieces: int) -> None:
        self._check_bounds(_broom_graph(s, a), number_of_pieces)

    @pytest.mark.parametrize(
        "n,H,number_of_pieces",
        [(500, 100_000, 3), (1000, 50_000, 2), (300, 200_000, 4)],
    )
    def test_heavy_node_path_bounds_sanity(
        self, n: int, H: int, number_of_pieces: int
    ) -> None:
        G = nx.path_graph(n)
        nx.set_node_attributes(
            G, {v: (H if v == n // 2 else 1) for v in G.nodes}, "weight"
        )
        self._check_bounds(G, number_of_pieces, weight="weight")

    @pytest.mark.parametrize(
        "seed,n,number_of_pieces",
        [(0, 1000, 4), (0, 500, 3), (6, 750, 5)],
    )
    def test_powerlaw_tree_bounds_sanity(
        self, seed: int, n: int, number_of_pieces: int
    ) -> None:
        T = nx.random_powerlaw_tree(n, seed=seed, tries=1000)
        self._check_bounds(T, number_of_pieces)


class TestRegressions:
    """Pin down contracts for bugs fixed during development and code review."""

    def test_specific_tree_minmax_number_of_pieces_5(self) -> None:
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


def _random_tree_with_weights(n: int, rng):
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
    def test_minmax_additive(self, seed) -> None:
        rng = random.Random(seed)
        for trial in range(5):
            n = rng.randint(3, 9)
            number_of_pieces = rng.randint(2, min(n, 5))
            T = _random_tree_with_weights(n, rng)

            for wf_name, label in self.ADDITIVE_WFS:
                p_algo = _minmax(T, number_of_pieces, weight_function=wf_name)
                p_bf = _brute_force_partition(
                    T, number_of_pieces, False, weight_function=wf_name
                )

                assert _is_valid_partition(
                    T, p_algo, number_of_pieces, weight_function=wf_name
                ), (
                    f"Invalid partition: seed={seed} trial={trial} n={n} number_of_pieces={number_of_pieces} wf={label}"
                )
                assert math.isclose(
                    _max_weight(p_algo), _max_weight(p_bf), rel_tol=1e-6
                ), (
                    f"Mismatch: seed={seed} trial={trial} n={n} number_of_pieces={number_of_pieces} wf={label} "
                    f"got={_max_weight(p_algo)} expected={_max_weight(p_bf)}"
                )

    @pytest.mark.parametrize("seed", [0, 1, 7, 13, 42, 99, 123, 200])
    def test_maxmin_additive(self, seed) -> None:
        rng = random.Random(seed)
        for trial in range(5):
            n = rng.randint(3, 9)
            number_of_pieces = rng.randint(2, min(n, 5))
            T = _random_tree_with_weights(n, rng)

            for wf_name, label in self.ADDITIVE_WFS:
                p_algo = _maxmin(T, number_of_pieces, weight_function=wf_name)
                p_bf = _brute_force_partition(
                    T, number_of_pieces, True, weight_function=wf_name
                )

                assert _is_valid_partition(
                    T, p_algo, number_of_pieces, weight_function=wf_name
                ), (
                    f"Invalid partition: seed={seed} trial={trial} n={n} number_of_pieces={number_of_pieces} wf={label}"
                )
                assert math.isclose(
                    _min_weight(p_algo), _min_weight(p_bf), rel_tol=1e-6
                ), (
                    f"Mismatch: seed={seed} trial={trial} n={n} number_of_pieces={number_of_pieces} wf={label} "
                    f"got={_min_weight(p_algo)} expected={_min_weight(p_bf)}"
                )


class TestComponentConsistencyMultiWF:
    """For each weight function on medium trees, verify structural validity:
    exactly number_of_pieces connected components, disjoint node coverage, and
    reported weight matching recomputed weight."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 99, 123])
    def test_consistency(self, seed: int) -> None:
        rng = random.Random(seed)
        for _ in range(4):
            n = rng.randint(5, 50)
            number_of_pieces = rng.randint(2, min(n, 10))
            T = _random_tree_with_weights(n, rng)

            for wf_name in [
                "vertex_weight_sum",
                "edge_weight_sum",
                "mixed_sum",
                "vertex_count",
            ]:
                for fn in [_minmax, _maxmin]:
                    p = fn(T, number_of_pieces, weight_function=wf_name)
                    assert _is_valid_partition(
                        T, p, number_of_pieces, weight_function=wf_name
                    )


class TestEdgeWeightSumAnalytic:
    """Analytical tests for weight_function='edge_weight_sum'.

    Component weight = sum of edge weights within the component.
    Singletons have weight 0 (no edges).
    """

    @staticmethod
    def _path_with_uniform_edge_weight(n: int, w: float) -> nx.Graph:
        G = nx.path_graph(n)
        for u, v in G.edges:
            G.edges[u, v]["weight"] = w
        return G

    @pytest.mark.parametrize(
        "n,number_of_pieces,w",
        [
            (6, 3, 1.0),  # 6/3=2 nodes each, 1 edge each → w
            (9, 3, 2.0),  # 9/3=3 nodes each, 2 edges each → 2w
            (12, 4, 1.0),  # 12/4=3 nodes each, 2 edges each → 2w
            (10, 5, 3.0),  # 10/5=2 nodes each, 1 edge each → w
        ],
    )
    def test_path_uniform_edge_weights(
        self, n: int, number_of_pieces: int, w: float
    ) -> None:
        """P_n with uniform edge weights w, n divisible by number_of_pieces.

        Each component has n/number_of_pieces nodes and n/number_of_pieces - 1
        edges → weight (n/number_of_pieces - 1)*w.
        Perfect partition: min-max = max-min = (n/number_of_pieces - 1)*w.
        """
        assert n % number_of_pieces == 0
        G = self._path_with_uniform_edge_weight(n, w)
        expected = (n // number_of_pieces - 1) * w

        p_mm = _minmax(G, number_of_pieces, weight_function="edge_weight_sum")
        assert _is_valid_partition(
            G, p_mm, number_of_pieces, weight_function="edge_weight_sum"
        )
        assert math.isclose(_max_weight(p_mm), expected)

        p_mx = _maxmin(G, number_of_pieces, weight_function="edge_weight_sum")
        assert _is_valid_partition(
            G, p_mx, number_of_pieces, weight_function="edge_weight_sum"
        )
        assert math.isclose(_min_weight(p_mx), expected)

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_star_uniform_edge_weights_number_of_pieces_2(self, n: int) -> None:
        """S_n with uniform edge weights 1.0, number_of_pieces=2.

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

    @pytest.mark.parametrize("n,number_of_pieces", [(5, 3), (8, 5), (10, 6)])
    def test_star_multiple_cuts(self, n: int, number_of_pieces: int) -> None:
        """S_n with uniform edge weights, number_of_pieces parts.

        number_of_pieces-1 cuts isolate number_of_pieces-1 leaves (weight 0 each).
        Center keeps n-number_of_pieces+1 leaves
        → n-number_of_pieces+1 edges → weight n-number_of_pieces+1.
        min-max = n-number_of_pieces+1, max-min = 0.
        """
        G = nx.star_graph(n)
        for u, v in G.edges:
            G.edges[u, v]["weight"] = 1.0

        p_mm = _minmax(G, number_of_pieces, weight_function="edge_weight_sum")
        assert math.isclose(_max_weight(p_mm), n - number_of_pieces + 1)

        p_mx = _maxmin(G, number_of_pieces, weight_function="edge_weight_sum")
        assert math.isclose(_min_weight(p_mx), 0.0)

    @pytest.mark.parametrize(
        "m,a,b",
        [(3, 1.0, 2.0), (5, 2.0, 3.0), (4, 1.5, 4.5)],
    )
    def test_path_paired_edge_weights(self, m: int, a: float, b: float) -> None:
        """P_{2m+1} with alternating edge weights [a, b, a, b, ...], number_of_pieces=m:
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
    def test_chain_of_stars_edge_weights(self, num_hubs: int, m: int, w: float) -> None:
        """Chain of num_hubs stars with m leaves each, all edge weights w.

        number_of_pieces=num_hubs: cut all hub-hub edges.
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
        "n,number_of_pieces",
        [(6, 2), (6, 3), (7, 2), (9, 3), (10, 4), (12, 4)],
    )
    def test_path_vertex_count(self, n: int, number_of_pieces: int) -> None:
        """P_n, number_of_pieces parts:
        min-max = ceil(n/number_of_pieces), max-min = floor(n/number_of_pieces).
        """
        G = nx.path_graph(n)
        expected_minmax = math.ceil(n / number_of_pieces)
        expected_maxmin = math.floor(n / number_of_pieces)

        p_mm = _minmax(G, number_of_pieces, weight_function="vertex_count")
        assert _max_weight(p_mm) == expected_minmax

        p_mx = _maxmin(G, number_of_pieces, weight_function="vertex_count")
        assert _min_weight(p_mx) == expected_maxmin

    @pytest.mark.parametrize("n", [4, 6, 10])
    def test_star_vertex_count_number_of_pieces_2(self, n: int) -> None:
        """S_n, number_of_pieces=2: min-max = n (center + n-1 leaves), max-min = 1."""
        G = nx.star_graph(n)

        p_mm = _minmax(G, 2, weight_function="vertex_count")
        assert _max_weight(p_mm) == n

        p_mx = _maxmin(G, 2, weight_function="vertex_count")
        assert _min_weight(p_mx) == 1

    @pytest.mark.parametrize(
        "spine,leaves_per_node",
        [(3, 2), (4, 2), (5, 3)],
    )
    def test_caterpillar_vertex_count_perfect(
        self, spine: int, leaves_per_node: int
    ) -> None:
        """Caterpillar number_of_pieces=spine: each component = spine_node + leaves = lp+1."""
        G = _caterpillar(spine, leaves_per_node)
        expected = leaves_per_node + 1

        p_mm = _minmax(G, spine, weight_function="vertex_count")
        assert _max_weight(p_mm) == expected

        p_mx = _maxmin(G, spine, weight_function="vertex_count")
        assert _min_weight(p_mx) == expected


class TestMixedSumAnalytic:
    """Analytical tests for weight_function='mixed_sum'.

    Component weight = sum of vertex weights + sum of edge weights.
    """

    @pytest.mark.parametrize(
        "n,number_of_pieces",
        [(6, 3), (9, 3), (10, 5), (12, 4)],
    )
    def test_path_unit_weights(self, n: int, number_of_pieces: int) -> None:
        """P_n with unit vertex and edge weights, number_of_pieces parts (n divisible by number_of_pieces).

        Component with k nodes: weight = k + (k-1) = 2k-1.
        With n/number_of_pieces nodes per part: weight = 2*(n/number_of_pieces) - 1.
        """
        assert n % number_of_pieces == 0
        G = nx.path_graph(n)
        for u, v in G.edges:
            G.edges[u, v]["weight"] = 1.0
        piece = n // number_of_pieces
        expected = 2 * piece - 1

        p_mm = _minmax(G, number_of_pieces, weight_function="mixed_sum")
        assert _is_valid_partition(
            G, p_mm, number_of_pieces, weight_function="mixed_sum"
        )
        assert math.isclose(_max_weight(p_mm), expected)

        p_mx = _maxmin(G, number_of_pieces, weight_function="mixed_sum")
        assert _is_valid_partition(
            G, p_mx, number_of_pieces, weight_function="mixed_sum"
        )
        assert math.isclose(_min_weight(p_mx), expected)

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_star_unit_weights_number_of_pieces_2(self, n: int) -> None:
        """S_n with unit vertex and edge weights, number_of_pieces=2.

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
    def test_chain_of_stars_mixed_perfect(self, num_hubs: int, m: int) -> None:
        """Chain of stars, unit vertex + edge weights, number_of_pieces=num_hubs.

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


class TestWeightFunctionStringAPI:
    """Verify that the string-based weight_function parameter works correctly."""

    def test_invalid_weight_function_raises(self) -> None:
        G = nx.path_graph(4)
        with pytest.raises(nx.NetworkXError, match="weight_function"):
            _minmax(G, 2, weight_function="nonexistent")
        with pytest.raises(nx.NetworkXError, match="weight_function"):
            _maxmin(G, 2, weight_function="nonexistent")

    def test_default_matches_vertex_weight_sum(self) -> None:
        """Default weight_function gives the same result as explicit 'vertex_weight_sum'."""
        G = nx.path_graph(8)
        for v in G.nodes:
            G.nodes[v]["weight"] = v + 1

        p_default = _minmax(G, 3, node_weight="weight")
        p_explicit = _minmax(
            G, 3, node_weight="weight", weight_function="vertex_weight_sum"
        )
        assert _max_weight(p_default) == _max_weight(p_explicit)

    # All-zero edge_weight_sum: contract is len(partition) == number_of_pieces, not 1.
    @pytest.mark.parametrize(
        "n,number_of_pieces", [(4, 2), (4, 3), (6, 2), (6, 4), (10, 5)]
    )
    def test_edge_weight_sum_all_zero_returns_number_of_pieces_parts(
        self, n: int, number_of_pieces: int
    ) -> None:
        G = nx.path_graph(n)
        nx.set_edge_attributes(G, 0.0, "weight")

        p_mm = _minmax(G, number_of_pieces, weight_function="edge_weight_sum")
        assert len(p_mm) == number_of_pieces
        assert all(w == 0.0 for _, w in p_mm)
        assert _is_valid_partition(
            G, p_mm, number_of_pieces, weight_function="edge_weight_sum"
        )

        p_mx = _maxmin(G, number_of_pieces, weight_function="edge_weight_sum")
        assert len(p_mx) == number_of_pieces
        assert all(w == 0.0 for _, w in p_mx)
        assert _is_valid_partition(
            G, p_mx, number_of_pieces, weight_function="edge_weight_sum"
        )

    # _add_cuts_to_reach_q uses sorted() to break ties deterministically;
    # the same input must always produce the same output.
    @pytest.mark.parametrize("seed", [0, 1, 7, 13, 42])
    def test_tiebreaker_determinism_maxmin(self, seed: int) -> None:
        """Repeated calls on the same tree give identical partitions."""
        rng = random.Random(seed)
        n = rng.randint(8, 20)
        T = nx.random_labeled_tree(n, seed=seed)
        for v in T.nodes:
            T.nodes[v]["weight"] = rng.randint(1, 3)  # small range → many ties
        number_of_pieces = rng.randint(2, min(n, 5))

        first = _maxmin(T, number_of_pieces, node_weight="weight")
        for _ in range(5):
            again = _maxmin(T, number_of_pieces, node_weight="weight")
            assert again == first

    def test_tiebreaker_determinism_symmetric_star(self) -> None:
        """Symmetric star forces ties in _reduce_cuts_to_reach_number_of_pieces."""
        G = nx.star_graph(8)  # center 0 + 8 leaves, all unit weight
        first = _maxmin(G, 4)
        for _ in range(10):
            assert _maxmin(G, 4) == first

    # Custom edge_weight attribute name: edge_weight_sum / mixed_sum must
    # honor a non-default attribute name.
    def test_edge_weight_custom_attr_name(self) -> None:
        G = nx.path_graph(6)
        nx.set_edge_attributes(G, {e: 2.0 for e in G.edges}, "cost")
        # edge_weight_sum with custom name
        p = _minmax(G, 3, edge_weight="cost", weight_function="edge_weight_sum")
        # 6 nodes in 3 parts of 2 nodes each → 1 edge per part → weight = 2.0
        assert math.isclose(_max_weight(p), 2.0)
        assert _is_valid_partition(
            G, p, 3, weight_function="edge_weight_sum", edge_weight="cost"
        )

    def test_mixed_sum_distinct_node_and_edge_attrs(self) -> None:
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


class TestNumpyWeights:
    """numpy scalar weights (int64, float64) produce correct partitions without
    overflow, silent wrap, or RuntimeWarning."""

    def test_numpy_int_weights_are_exact_minmax(self) -> None:
        np = pytest.importorskip("numpy")
        G = nx.path_graph(4)
        for v, w in enumerate([2**61, 2**61, 2**62, 2**61]):
            G.nodes[v]["weight"] = np.int64(w)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            parts = min_max_tree_partition(G, 2)
        # Optimal cut is (1, 2): max part weight 3 * 2**61 beats 2**63.
        assert set(parts) == {frozenset({0, 1}), frozenset({2, 3})}

    def test_numpy_int_weights_are_exact_maxmin(self) -> None:
        np = pytest.importorskip("numpy")
        G = nx.path_graph(4)
        for v, w in enumerate([2**61, 2**61, 2**62, 2**61]):
            G.nodes[v]["weight"] = np.int64(w)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            parts = max_min_tree_partition(G, 2)
        # Optimal cut is (1, 2): min part weight 2**62 beats 2**61.
        assert set(parts) == {frozenset({0, 1}), frozenset({2, 3})}

    def test_numpy_float_weights(self) -> None:
        np = pytest.importorskip("numpy")
        G = nx.path_graph(4)
        for v, w in enumerate([3.0, 1.0, 1.0, 3.0]):
            G.nodes[v]["weight"] = np.float64(w)
        parts = min_max_tree_partition(G, 2)
        assert set(parts) == {frozenset({0, 1}), frozenset({2, 3})}


class TestMissingEdgeAttrDefault:
    """Edges missing the weight attribute are treated as weight 1."""

    @pytest.mark.parametrize("weight_function", ["edge_weight_sum", "mixed_sum"])
    def test_path_missing_edge_attrs_minmax(self, weight_function) -> None:
        G = nx.path_graph(6)
        G.edges[0, 1]["weight"] = 1
        G.edges[1, 2]["weight"] = 1
        G.edges[4, 5]["weight"] = 1
        parts = min_max_tree_partition(G, 2, weight_function=weight_function)
        assert set(parts) == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}

    @pytest.mark.parametrize("weight_function", ["edge_weight_sum", "mixed_sum"])
    def test_path_missing_edge_attrs_maxmin(self, weight_function) -> None:
        G = nx.path_graph(6)
        G.edges[0, 1]["weight"] = 1
        G.edges[1, 2]["weight"] = 1
        G.edges[4, 5]["weight"] = 1
        parts = max_min_tree_partition(G, 2, weight_function=weight_function)
        assert set(parts) == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}

    @pytest.mark.parametrize("weight_function", ["edge_weight_sum", "mixed_sum"])
    @pytest.mark.parametrize("seed", range(4))
    def test_random_trees_missing_edge_attrs(self, seed, weight_function) -> None:
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
        number_of_pieces = rng.randint(2, 5)

        p_mm = _minmax(T, number_of_pieces, weight_function=weight_function)
        bf_mm = _brute_force_partition(
            T, number_of_pieces, False, weight_function=weight_function
        )
        assert _is_valid_partition(
            T, p_mm, number_of_pieces, weight_function=weight_function
        )
        assert _max_weight(p_mm) == _max_weight(bf_mm)

        p_mx = _maxmin(T, number_of_pieces, weight_function=weight_function)
        bf_mx = _brute_force_partition(
            T, number_of_pieces, True, weight_function=weight_function
        )
        assert _is_valid_partition(
            T, p_mx, number_of_pieces, weight_function=weight_function
        )
        assert _min_weight(p_mx) == _min_weight(bf_mx)
