"""Synthetic dataset generation module for Offline Connectivity.

Generates reproducible, valid multigraph datasets with edge removal sequences
and corresponding connectivity queries.

Supported topologies:
- Standard benchmark suite: sample, small, medium, large, extra_large
- Adversarial / structured benchmarks:
  1. chain: Long path graph testing deep tree structures.
  2. star: High-degree hub node testing star topologies.
  3. disconnected: Multi-component forest graph with many isolated components.
  4. multigraph: Dense multigraph with substantial edge multiplicity (parallel edges).
  5. all_removed: Complete graph destruction where K = M.
  6. bfs_friendly: Localized queries (distance <= 2) where BFS terminates rapidly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.graph_utils import normalize_edge, save_graph_to_json, validate_graph_data  # type: ignore[import-not-found,import-untyped]
except ImportError:
    from graph_utils import normalize_edge, save_graph_to_json, validate_graph_data  # type: ignore[import-not-found,import-untyped]


def generate_graph_dataset(
    n: int,
    m: int,
    k: int,
    seed: int = 42,
    connected_backbone: bool = True,
    allow_parallel: bool = False,
) -> Dict[str, Any]:
    """Generate a reproducible, valid graph dataset with removals and queries.

    Args:
        n: Number of vertices (vertices will be labeled 0 to n-1).
        m: Number of undirected edges.
        k: Number of removals (0 <= k <= m).
        seed: Random seed for reproducibility.
        connected_backbone: If True, first constructs a spanning tree so that
                           the initial graph has full connectivity.
        allow_parallel: If True, allows multiple parallel edges between the same vertex pair.

    Returns:
        Dictionary adhering to the project schema:
        {"n": int, "edges": list, "removals": list, "queries": list}
    """
    if n < 1:
        raise ValueError(f"Number of vertices n must be >= 1, got {n}")
    max_simple_edges = n * (n - 1) // 2
    if not allow_parallel and m > max_simple_edges:
        raise ValueError(f"Too many edges requested for simple graph: m={m} > {max_simple_edges}")
    if k > m:
        raise ValueError(f"Removals k={k} cannot exceed total edges m={m}")

    rng = random.Random(seed)
    edges_list: List[List[int]] = []
    seen_edges: set[Tuple[int, int]] = set()

    # Step 1: Create spanning tree backbone if requested and n > 1
    if connected_backbone and n > 1:
        nodes = list(range(n))
        rng.shuffle(nodes)
        connected_nodes = [nodes[0]]
        unconnected_nodes = nodes[1:]
        for u in unconnected_nodes:
            v = rng.choice(connected_nodes)
            norm = normalize_edge(u, v)
            edges_list.append([norm[0], norm[1]])
            seen_edges.add(norm)
            connected_nodes.append(u)

    # Step 2: Add remaining edges
    while len(edges_list) < m:
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        norm = normalize_edge(u, v)
        if not allow_parallel:
            if norm in seen_edges:
                continue
            seen_edges.add(norm)
        edges_list.append([norm[0], norm[1]])

    # Step 3: Select k distinct edge instances to remove
    removal_indices = rng.sample(range(len(edges_list)), k)
    removals_list = [edges_list[idx] for idx in removal_indices]

    # Step 4: Generate k connectivity queries
    queries_list: List[List[int]] = []
    for i in range(k):
        r_type = rng.random()
        rem_u, rem_v = removals_list[i]
        if r_type < 0.4:
            queries_list.append([rem_u, rem_v])
        elif r_type < 0.8:
            q_u = rng.randint(0, n - 1)
            q_v = rng.randint(0, n - 1)
            queries_list.append([q_u, q_v])
        else:
            queries_list.append([0, n - 1])

    data: Dict[str, Any] = {
        "n": n,
        "edges": edges_list,
        "removals": removals_list,
        "queries": queries_list,
    }

    validate_graph_data(data)
    return data


def create_sample_dataset() -> Dict[str, Any]:
    """Create the canonical 5-vertex sample graph from the problem statement."""
    return {
        "n": 5,
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
        "removals": [[1, 2], [3, 4], [0, 1]],
        "queries": [[0, 4], [0, 4], [0, 4]],
    }


def generate_chain_dataset(n: int = 4000, k: int = 1500, seed: int = 501) -> Dict[str, Any]:
    """Generate a long chain/path graph (0-1-2-...-n-1)."""
    rng = random.Random(seed)
    edges = [[i, i + 1] for i in range(n - 1)]
    rem_indices = rng.sample(range(len(edges)), k)
    removals = [edges[idx] for idx in rem_indices]
    queries = []
    for i in range(k):
        rem_u, rem_v = removals[i]
        if rng.random() < 0.5:
            queries.append([rem_u, rem_v])
        else:
            queries.append([0, n - 1])
    data = {"n": n, "edges": edges, "removals": removals, "queries": queries}
    validate_graph_data(data)
    return data


def generate_star_dataset(n: int = 4000, k: int = 1500, seed: int = 502) -> Dict[str, Any]:
    """Generate a star graph with hub vertex 0 connected to all leaves 1..n-1."""
    rng = random.Random(seed)
    edges = [[0, i] for i in range(1, n)]
    rem_indices = rng.sample(range(len(edges)), k)
    removals = [edges[idx] for idx in rem_indices]
    queries = []
    for i in range(k):
        rem_leaf = removals[i][1]
        other_leaf = rng.randint(1, n - 1)
        queries.append([rem_leaf, other_leaf])
    data = {"n": n, "edges": edges, "removals": removals, "queries": queries}
    validate_graph_data(data)
    return data


def generate_disconnected_dataset(
    num_comp: int = 50,
    comp_size: int = 80,
    extra_edges: int = 20,
    k: int = 1500,
    seed: int = 503,
) -> Dict[str, Any]:
    """Generate a multi-component forest with many isolated components."""
    rng = random.Random(seed)
    n = num_comp * comp_size
    edges: List[List[int]] = []

    for c in range(num_comp):
        base = c * comp_size
        # Spanning line for component
        for i in range(comp_size - 1):
            edges.append([base + i, base + i + 1])
        # Random extra edges in component
        for _ in range(extra_edges):
            u = base + rng.randint(0, comp_size - 1)
            v = base + rng.randint(0, comp_size - 1)
            if u != v:
                edges.append([min(u, v), max(u, v)])

    rem_indices = rng.sample(range(len(edges)), k)
    removals = [edges[idx] for idx in rem_indices]
    queries: List[List[int]] = []
    for i in range(k):
        # 50% within-component, 50% across-component
        if rng.random() < 0.5:
            c = rng.randint(0, num_comp - 1)
            base = c * comp_size
            queries.append([base + rng.randint(0, comp_size - 1), base + rng.randint(0, comp_size - 1)])
        else:
            c1, c2 = rng.sample(range(num_comp), 2)
            queries.append([c1 * comp_size + rng.randint(0, comp_size - 1), c2 * comp_size + rng.randint(0, comp_size - 1)])

    data = {"n": n, "edges": edges, "removals": removals, "queries": queries}
    validate_graph_data(data)
    return data


def generate_multigraph_dataset(n: int = 500, m: int = 3000, k: int = 1200, seed: int = 504) -> Dict[str, Any]:
    """Generate a dense multigraph with substantial edge multiplicity (parallel edges)."""
    rng = random.Random(seed)
    edges: List[List[int]] = []
    base_pairs = []
    for _ in range(n):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            base_pairs.append(normalize_edge(u, v))

    while len(edges) < m:
        pair = rng.choice(base_pairs)
        edges.append([pair[0], pair[1]])

    rem_indices = rng.sample(range(len(edges)), k)
    removals = [edges[idx] for idx in rem_indices]
    queries = []
    for i in range(k):
        rem_u, rem_v = removals[i]
        queries.append([rem_u, rem_v])

    data = {"n": n, "edges": edges, "removals": removals, "queries": queries}
    validate_graph_data(data)
    return data


def generate_all_removed_dataset(n: int = 1000, m: int = 2000, seed: int = 505) -> Dict[str, Any]:
    """Generate a graph where all edges are removed (K = M)."""
    return generate_graph_dataset(n=n, m=m, k=m, seed=seed, allow_parallel=True)


def generate_bfs_friendly_dataset(n: int = 3000, m: int = 6000, k: int = 1500, seed: int = 506) -> Dict[str, Any]:
    """Generate a dataset where queries target immediate neighbors or distance-1/2 pairs.

    Because target vertices are within local neighborhoods, BFS early-exits in O(1) or O(degree)
    steps, providing an honest baseline comparison where BFS excels.
    """
    rng = random.Random(seed)
    ds = generate_graph_dataset(n=n, m=m, k=k, seed=seed, allow_parallel=True)
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in ds["edges"]:
        adj[u].append(v)
        adj[v].append(u)

    queries: List[List[int]] = []
    for i in range(k):
        rem_u, rem_v = ds["removals"][i]
        if adj[rem_u]:
            target = rng.choice(adj[rem_u])
            queries.append([rem_u, target])
        else:
            queries.append([rem_u, rem_u])

    ds["queries"] = queries
    validate_graph_data(ds)
    return ds


def generate_all_datasets(data_dir: Path | str) -> None:
    """Generate the full suite of 11 benchmark datasets."""
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample
    save_graph_to_json(create_sample_dataset(), out_dir / "sample.json")

    # 2. Small
    save_graph_to_json(generate_graph_dataset(n=100, m=200, k=80, seed=101), out_dir / "small.json")

    # 3. Medium
    save_graph_to_json(generate_graph_dataset(n=1000, m=3000, k=1000, seed=202), out_dir / "medium.json")

    # 4. Large
    save_graph_to_json(generate_graph_dataset(n=5000, m=15000, k=2500, seed=303), out_dir / "large.json")

    # 5. Extra Large
    save_graph_to_json(generate_graph_dataset(n=10000, m=30000, k=3500, seed=404), out_dir / "extra_large.json")

    # 6. Chain (Path)
    save_graph_to_json(generate_chain_dataset(n=4000, k=1500, seed=501), out_dir / "chain.json")

    # 7. Star
    save_graph_to_json(generate_star_dataset(n=4000, k=1500, seed=502), out_dir / "star.json")

    # 8. Disconnected
    save_graph_to_json(generate_disconnected_dataset(num_comp=50, comp_size=80, extra_edges=20, k=1500, seed=503), out_dir / "disconnected.json")

    # 9. Multigraph (Parallel Edges)
    save_graph_to_json(generate_multigraph_dataset(n=500, m=3000, k=1200, seed=504), out_dir / "multigraph.json")

    # 10. All Removed (K = M)
    save_graph_to_json(generate_all_removed_dataset(n=1000, m=2000, seed=505), out_dir / "all_removed.json")

    # 11. BFS Friendly (Localized queries)
    save_graph_to_json(generate_bfs_friendly_dataset(n=3000, m=6000, k=1500, seed=506), out_dir / "bfs_friendly.json")


generate_standard_benchmark_datasets = generate_all_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic graph datasets.")
    parser.add_argument("--outdir", type=str, default="data", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Generate all 11 benchmark datasets")
    parser.add_argument("--standard", action="store_true", help="Generate all datasets")

    args = parser.parse_args()
    generate_all_datasets(args.outdir)
    print(f"Generated all benchmark datasets in {args.outdir}/")
