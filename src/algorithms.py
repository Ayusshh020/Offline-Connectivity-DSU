"""Algorithm implementations for Offline Connectivity After Edge Removals.

Supports Multigraphs (graphs with parallel edges and edge multiplicity).

Approaches:
1. Approach A: Reverse Processing with Disjoint Set Union (DSU)
   Time Complexity: O((M + K) * alpha(N))
   Auxiliary Space: O(N + M)

2. Approach B: Forward Edge Removals with BFS Recomputation (Multiplicity-aware)
   Time Complexity: O(K * (N + M))
   Auxiliary Space: O(N + M)

3. Approach C: Forward Edge Removals with DFS Recomputation (Multiplicity-aware)
   Time Complexity: O(K * (N + M))
   Auxiliary Space: O(N + M)

4. Approach D (Oracle): Fresh DSU Rebuilt from scratch after each removal
   Time Complexity: O(K * M * alpha(N))
   Used as an independent correctness verification oracle.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, Dict, List, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.dsu import DisjointSetUnion  # type: ignore[import-not-found,import-untyped]
    from src.graph_utils import normalize_edge, validate_graph_data  # type: ignore[import-not-found,import-untyped]
except ImportError:
    from dsu import DisjointSetUnion  # type: ignore[import-not-found,import-untyped]
    from graph_utils import normalize_edge, validate_graph_data  # type: ignore[import-not-found,import-untyped]


def reverse_dsu_connectivity(
    n: int,
    edges: List[List[int]],
    removals: List[List[int]],
    queries: List[List[int]],
    validate: bool = True,
) -> List[bool]:
    """Determine connectivity after each edge removal using Reverse DSU.

    Supports multigraphs with parallel edges. Each edge removal deletes exactly
    one instance of that edge.
    
    Reverse Processing Steps:
    1. Count removals by edge multiplicity using a Counter of normalized edges.
    2. Iterate over initial edges; if an edge appears in removal_count (> 0),
       decrement its removal count and omit it from residual edges. Otherwise,
       retain it in the final residual graph.
    3. Initialize DSU with the residual graph's remaining edges.
    4. Traverse the removal sequence in reverse order (K-1 down to 0):
       a. Evaluate query[i] on the current component structure.
       b. Restore the removed edge removals[i] using DSU union.
    5. Return query answers in original chronological order.

    Args:
        n: Number of vertices.
        edges: Initial list of edges [[u, v], ...].
        removals: Chronological sequence of edges to remove.
        queries: Connectivity queries [[u, v], ...], one per removal.
        validate: Whether to validate input graph constraints first.

    Returns:
        List of booleans where index i is True if query[i] vertices were
        connected after removal[i], False otherwise.
    """
    if validate:
        validate_graph_data({
            "n": n,
            "edges": edges,
            "removals": removals,
            "queries": queries,
        })

    k = len(removals)
    if k == 0:
        return []

    # 1. Build Counter of normalized removals
    removal_count: Counter[Tuple[int, int]] = Counter(
        normalize_edge(u, v) for u, v in removals
    )

    # 2. Filter remaining edges accounting for multiplicity
    remaining_edges: List[List[int]] = []
    for edge in edges:
        norm = normalize_edge(edge[0], edge[1])
        if removal_count[norm] > 0:
            removal_count[norm] -= 1
        else:
            remaining_edges.append(edge)

    # 3. Build DSU and unite remaining edges
    dsu = DisjointSetUnion(n)
    for u, v in remaining_edges:
        dsu.union(u, v)

    # 4. Process removals in reverse order
    answers: List[bool] = [False] * k

    for i in range(k - 1, -1, -1):
        # DSU reflects graph state immediately after removal[i]
        q_u, q_v = queries[i]
        answers[i] = dsu.connected(q_u, q_v)

        # Restore the edge removed at step i
        rem_u, rem_v = removals[i]
        dsu.union(rem_u, rem_v)

    return answers


def bfs_recomputation_connectivity(
    n: int,
    edges: List[List[int]],
    removals: List[List[int]],
    queries: List[List[int]],
    validate: bool = True,
) -> List[bool]:
    """Determine connectivity using forward BFS recomputation after each removal.

    Maintains adjacency using multiplicity counters (Counter per vertex)
    to accurately support parallel edges. A removal decrements the count and deletes
    the neighbor key only when the count reaches zero.

    Args:
        n: Number of vertices.
        edges: Initial list of edges [[u, v], ...].
        removals: Chronological sequence of edges to remove.
        queries: Connectivity queries [[u, v], ...], one per removal.
        validate: Whether to validate input graph constraints first.

    Returns:
        List of booleans where index i is True if query[i] vertices were
        connected after removal[i], False otherwise.
    """
    if validate:
        validate_graph_data({
            "n": n,
            "edges": edges,
            "removals": removals,
            "queries": queries,
        })

    k = len(removals)
    if k == 0:
        return []

    # Adjacency list storing multiplicity via Counter
    adj: List[Counter[int]] = [Counter() for _ in range(n)]
    for u, v in edges:
        adj[u][v] += 1
        adj[v][u] += 1

    answers: List[bool] = []

    for i in range(k):
        rem_u, rem_v = removals[i]
        # Decrement multiplicity; delete neighbor entry only when count drops to 0
        adj[rem_u][rem_v] -= 1
        if adj[rem_u][rem_v] == 0:
            del adj[rem_u][rem_v]

        adj[rem_v][rem_u] -= 1
        if adj[rem_v][rem_u] == 0:
            del adj[rem_v][rem_u]

        # Check connectivity for query[i] using BFS
        q_u, q_v = queries[i]
        if q_u == q_v:
            answers.append(True)
            continue

        visited = [False] * n
        visited[q_u] = True
        queue: deque[int] = deque([q_u])
        found = False

        while queue:
            curr = queue.popleft()
            if curr == q_v:
                found = True
                break
            for neighbor in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        answers.append(found)

    return answers


def dfs_recomputation_connectivity(
    n: int,
    edges: List[List[int]],
    removals: List[List[int]],
    queries: List[List[int]],
    validate: bool = True,
) -> List[bool]:
    """Determine connectivity using forward DFS recomputation after each removal.

    Uses an explicit stack and multiplicity tracking to support multigraphs.

    Args:
        n: Number of vertices.
        edges: Initial list of edges.
        removals: Chronological sequence of edges to remove.
        queries: Connectivity queries, one per removal.
        validate: Whether to validate input graph constraints first.

    Returns:
        List of booleans representing query answers after each removal.
    """
    if validate:
        validate_graph_data({
            "n": n,
            "edges": edges,
            "removals": removals,
            "queries": queries,
        })

    k = len(removals)
    if k == 0:
        return []

    adj: List[Counter[int]] = [Counter() for _ in range(n)]
    for u, v in edges:
        adj[u][v] += 1
        adj[v][u] += 1

    answers: List[bool] = []

    for i in range(k):
        rem_u, rem_v = removals[i]
        adj[rem_u][rem_v] -= 1
        if adj[rem_u][rem_v] == 0:
            del adj[rem_u][rem_v]

        adj[rem_v][rem_u] -= 1
        if adj[rem_v][rem_u] == 0:
            del adj[rem_v][rem_u]

        q_u, q_v = queries[i]
        if q_u == q_v:
            answers.append(True)
            continue

        visited = [False] * n
        visited[q_u] = True
        stack: List[int] = [q_u]
        found = False

        while stack:
            curr = stack.pop()
            if curr == q_v:
                found = True
                break
            for neighbor in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        answers.append(found)

    return answers


def brute_force_oracle_connectivity(
    n: int,
    edges: List[List[int]],
    removals: List[List[int]],
    queries: List[List[int]],
) -> List[bool]:
    """Independent brute-force oracle that rebuilds a fresh DSU after every removal.

    Used strictly for differential testing to guarantee ground-truth correctness.
    """
    curr_edges = list(edges)
    answers: List[bool] = []

    for (rem_u, rem_v), (q_u, q_v) in zip(removals, queries):
        norm_rem = normalize_edge(rem_u, rem_v)
        # Find and remove exactly one matching copy
        removed = False
        for idx, e in enumerate(curr_edges):
            if normalize_edge(e[0], e[1]) == norm_rem:
                curr_edges.pop(idx)
                removed = True
                break
        if not removed:
            raise ValueError(f"Oracle error: edge {rem_u, rem_v} not found in current edges")

        # Rebuild fresh union-find from scratch
        dsu = DisjointSetUnion(n)
        for u, v in curr_edges:
            dsu.union(u, v)

        answers.append(dsu.connected(q_u, q_v))

    return answers


def solve_from_dict(data: Dict[str, Any], method: str = "dsu") -> List[bool]:
    """Solve connectivity queries given a validated graph data dictionary.

    Args:
        data: Dictionary with 'n', 'edges', 'removals', 'queries'.
        method: One of 'dsu', 'bfs', 'dfs', or 'oracle'.

    Returns:
        List of boolean query answers.
    """
    n = data["n"]
    edges = data["edges"]
    removals = data["removals"]
    queries = data["queries"]

    method_lower = method.lower().strip()
    if method_lower == "dsu":
        return reverse_dsu_connectivity(n, edges, removals, queries, validate=True)
    elif method_lower == "bfs":
        return bfs_recomputation_connectivity(n, edges, removals, queries, validate=True)
    elif method_lower == "dfs":
        return dfs_recomputation_connectivity(n, edges, removals, queries, validate=True)
    elif method_lower == "oracle":
        return brute_force_oracle_connectivity(n, edges, removals, queries)
    else:
        raise ValueError(f"Unknown algorithm method '{method}'. Choose 'dsu', 'bfs', 'dfs', or 'oracle'.")

