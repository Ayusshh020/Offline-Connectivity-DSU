"""Graph utilities and validation module for Multigraphs.

Handles data serialization, normalization of undirected edges, and strict
validation of multigraph inputs according to the problem specification.
Parallel edges are permitted and tracked by multiplicity; self-loops remain disallowed.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def normalize_edge(u: int, v: int) -> Tuple[int, int]:
    """Normalize an undirected edge so that the smaller vertex index comes first.

    Args:
        u: First vertex index.
        v: Second vertex index.

    Returns:
        Canonical tuple representation (min(u, v), max(u, v)).
    """
    return (u, v) if u <= v else (v, u)


def validate_graph_data(data: Dict[str, Any]) -> None:
    """Validate a graph dataset against all problem constraints.

    Ensures:
    1. Required keys ('n', 'edges', 'removals', 'queries') are present.
    2. 'n' is a non-negative integer.
    3. Vertices are within bounds [0, n - 1].
    4. Graph is a multigraph: parallel (duplicate) edges are allowed, but self-loops are disallowed.
    5. Each removal refers to an active edge instance in the graph (multiplicity tracking).
       If an edge has k parallel copies, it may be removed at most k times.
    6. Number of queries equals number of removals (one query after each removal).

    Raises:
        ValueError: If any validation rule is violated.
        TypeError: If types are incorrect.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Input data must be a JSON object (dict), got {type(data).__name__}")

    required_keys = ("n", "edges", "removals", "queries")
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key in graph data: '{key}'")

    n = data["n"]
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(f"'n' must be a non-negative integer, got {n}")

    edges = data["edges"]
    removals = data["removals"]
    queries = data["queries"]

    if not isinstance(edges, list):
        raise TypeError(f"'edges' must be a list, got {type(edges).__name__}")
    if not isinstance(removals, list):
        raise TypeError(f"'removals' must be a list, got {type(removals).__name__}")
    if not isinstance(queries, list):
        raise TypeError(f"'queries' must be a list, got {type(queries).__name__}")

    # Validate Edges & count multiplicities
    edge_counts: Counter[Tuple[int, int]] = Counter()
    for idx, edge in enumerate(edges):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError(f"Edge at index {idx} must be a pair [u, v], got {edge}")
        u, v = edge
        if not isinstance(u, int) or isinstance(u, bool) or not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"Edge vertices must be integers, got {edge}")
        if u < 0 or u >= n or v < 0 or v >= n:
            raise ValueError(f"Edge vertex out of bounds [0, {n - 1}]: {edge}")
        if u == v:
            raise ValueError(f"Self-loop detected at vertex {u}: self-loops are disallowed")

        norm_e = normalize_edge(u, v)
        edge_counts[norm_e] += 1

    # Validate Removals with multiplicity tracking
    remaining_counts: Counter[Tuple[int, int]] = Counter(edge_counts)
    for idx, rem in enumerate(removals):
        if not isinstance(rem, (list, tuple)) or len(rem) != 2:
            raise ValueError(f"Removal at index {idx} must be a pair [u, v], got {rem}")
        u, v = rem
        if not isinstance(u, int) or isinstance(u, bool) or not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"Removal vertices must be integers, got {rem}")
        if u < 0 or u >= n or v < 0 or v >= n:
            raise ValueError(f"Removal vertex out of bounds [0, {n - 1}]: {rem}")
        if u == v:
            raise ValueError(f"Invalid removal self-loop: {rem}")

        norm_rem = normalize_edge(u, v)
        if remaining_counts[norm_rem] <= 0:
            if edge_counts[norm_rem] == 0:
                raise ValueError(
                    f"Invalid removal at step {idx}: edge {rem} does not exist in the graph"
                )
            else:
                raise ValueError(
                    f"Invalid removal at step {idx}: edge {rem} had {edge_counts[norm_rem]} "
                    f"cop{'ies' if edge_counts[norm_rem] > 1 else 'y'}, all of which were already removed"
                )
        remaining_counts[norm_rem] -= 1

    # Validate Queries
    if len(queries) != len(removals):
        raise ValueError(
            f"Queries count ({len(queries)}) must match removals count ({len(removals)}) "
            f"as queries are evaluated after each removal"
        )

    for idx, q in enumerate(queries):
        if not isinstance(q, (list, tuple)) or len(q) != 2:
            raise ValueError(f"Query at index {idx} must be a pair [u, v], got {q}")
        u, v = q
        if not isinstance(u, int) or isinstance(u, bool) or not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"Query vertices must be integers, got {q}")
        if u < 0 or u >= n or v < 0 or v >= n:
            raise ValueError(f"Query vertex out of bounds [0, {n - 1}]: {q}")


def load_graph_from_json(filepath: Path | str) -> Dict[str, Any]:
    """Load and validate graph data from a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Validated dictionary containing graph specification.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    validate_graph_data(data)
    return data


def save_graph_to_json(data: Dict[str, Any], filepath: Path | str) -> None:
    """Save graph data to a JSON file after validation.

    Args:
        data: Graph dictionary.
        filepath: Target output file path.
    """
    validate_graph_data(data)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

