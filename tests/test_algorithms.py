"""Comprehensive Automated and Randomized Correctness Test Suite.

Covers:
1. Unit tests for DSU (path compression, union by rank/size, bounds).
2. Parallel edge (multigraph) support:
   - The exact problem statement test: n=2, edges=[[0,1],[0,1]], removals=[[0,1]], queries=[[0,1]] -> [True]
   - 3 parallel copies with 1 then 2 removals
   - Parallel edge where the other copy is also removed later
   - Removing more copies than exist raises ValueError
3. Edge case testing:
   - Empty graph (N=0)
   - Single vertex (N=1)
   - Disconnected graph
   - Chain graph (line graph)
   - Cycle graph
   - Complete graph (Kn)
   - All edges removed (K = M)
   - Zero removals (K = 0)
   - Multiple connected components
   - Boundary vertex queries (0 and N-1)
   - Self-connectivity queries (u == v)
   - Chain graph with 10,000 vertices (verifying no recursion depth limit error)
4. Input validation and error handling:
   - Invalid vertex index
   - Removing nonexistent edge
   - Removing more copies of an edge than exist in the multigraph
   - Graph with self-loop
   - Mismatched removals/queries length
   - Accepting valid parallel edges in initial graph
5. Randomized differential fuzz testing:
   - At least 2,000 trials on multigraphs comparing:
     Reverse DSU vs BFS vs DFS vs independent Brute-Force Oracle.
"""

from __future__ import annotations

from collections import Counter
import random
import sys
from pathlib import Path
import unittest

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.dsu import DisjointSetUnion  # type: ignore[import-not-found,import-untyped]
    from src.algorithms import (  # type: ignore[import-not-found,import-untyped]
        reverse_dsu_connectivity,
        bfs_recomputation_connectivity,
        dfs_recomputation_connectivity,
        brute_force_oracle_connectivity,
    )
    from src.graph_utils import validate_graph_data, normalize_edge  # type: ignore[import-not-found,import-untyped]
    from src.generate_data import generate_graph_dataset, create_sample_dataset  # type: ignore[import-not-found,import-untyped]
except ImportError:
    from dsu import DisjointSetUnion  # type: ignore[import-not-found,import-untyped]
    from algorithms import (  # type: ignore[import-not-found,import-untyped]
        reverse_dsu_connectivity,
        bfs_recomputation_connectivity,
        dfs_recomputation_connectivity,
        brute_force_oracle_connectivity,
    )
    from graph_utils import validate_graph_data, normalize_edge  # type: ignore[import-not-found,import-untyped]
    from generate_data import generate_graph_dataset, create_sample_dataset  # type: ignore[import-not-found,import-untyped]


class TestDisjointSetUnion(unittest.TestCase):
    """Unit tests for Disjoint Set Union implementation."""

    def test_initial_state(self):
        dsu = DisjointSetUnion(5)
        self.assertEqual(dsu.num_components, 5)
        for i in range(5):
            self.assertEqual(dsu.find(i), i)
            self.assertTrue(dsu.connected(i, i))

    def test_unions_and_connectivity(self):
        dsu = DisjointSetUnion(6)
        self.assertTrue(dsu.union(0, 1))
        self.assertTrue(dsu.union(1, 2))
        self.assertFalse(dsu.union(0, 2))  # Already in same component
        self.assertEqual(dsu.num_components, 4)

        self.assertTrue(dsu.connected(0, 2))
        self.assertFalse(dsu.connected(0, 3))

        dsu.union(3, 4)
        dsu.union(4, 5)
        self.assertEqual(dsu.num_components, 2)
        self.assertTrue(dsu.connected(3, 5))
        self.assertFalse(dsu.connected(2, 3))

        dsu.union(2, 3)
        self.assertEqual(dsu.num_components, 1)
        self.assertTrue(dsu.connected(0, 5))

    def test_union_by_size(self):
        dsu = DisjointSetUnion(4, use_size=True)
        self.assertTrue(dsu.union(0, 1))
        self.assertTrue(dsu.union(2, 3))
        self.assertTrue(dsu.union(1, 2))
        self.assertEqual(dsu.num_components, 1)
        self.assertTrue(dsu.connected(0, 3))

    def test_deep_chain_iterative_path_compression(self):
        # 10,000 elements chained to ensure zero recursion depth limit issues
        n = 10000
        dsu = DisjointSetUnion(n)
        for i in range(n - 1):
            dsu.union(i, i + 1)
        self.assertEqual(dsu.num_components, 1)
        self.assertTrue(dsu.connected(0, n - 1))

    def test_out_of_bounds(self):
        dsu = DisjointSetUnion(5)
        with self.assertRaises(IndexError):
            dsu.find(-1)
        with self.assertRaises(IndexError):
            dsu.find(5)
        with self.assertRaises(IndexError):
            dsu.connected(0, 10)


class TestParallelEdgesMultigraph(unittest.TestCase):
    """Specific tests for parallel (duplicate) edge handling."""

    def test_exact_assignment_failing_case(self):
        # n=2, edges=[[0,1],[0,1]], removals=[[0,1]], queries=[[0,1]] -> [True]
        n = 2
        edges = [[0, 1], [0, 1]]
        removals = [[0, 1]]
        queries = [[0, 1]]

        res_dsu = reverse_dsu_connectivity(n, edges, removals, queries)
        res_bfs = bfs_recomputation_connectivity(n, edges, removals, queries)
        res_dfs = dfs_recomputation_connectivity(n, edges, removals, queries)
        res_orc = brute_force_oracle_connectivity(n, edges, removals, queries)

        self.assertEqual(res_dsu, [True])
        self.assertEqual(res_bfs, [True])
        self.assertEqual(res_dfs, [True])
        self.assertEqual(res_orc, [True])

    def test_three_parallel_copies(self):
        # 3 parallel copies of (0,1): 1 removal then 2nd removal then 3rd removal
        n = 2
        edges = [[0, 1], [0, 1], [0, 1]]
        removals = [[0, 1], [0, 1], [0, 1]]
        queries = [[0, 1], [0, 1], [0, 1]]

        # After rem 1: 2 copies remain -> True
        # After rem 2: 1 copy remains -> True
        # After rem 3: 0 copies remain -> False
        expected = [True, True, False]

        self.assertEqual(reverse_dsu_connectivity(n, edges, removals, queries), expected)
        self.assertEqual(bfs_recomputation_connectivity(n, edges, removals, queries), expected)
        self.assertEqual(dfs_recomputation_connectivity(n, edges, removals, queries), expected)
        self.assertEqual(brute_force_oracle_connectivity(n, edges, removals, queries), expected)

    def test_parallel_edge_with_later_removal(self):
        # n=3, edges=[[0,1],[0,1],[1,2]], removals=[[0,1],[1,2],[0,1]], queries=[[0,1],[0,2],[0,1]]
        n = 3
        edges = [[0, 1], [0, 1], [1, 2]]
        removals = [[0, 1], [1, 2], [0, 1]]
        queries = [[0, 1], [0, 2], [0, 1]]

        # Step 1: remove (0,1) -> 2nd (0,1) remains, so (0,1) is True
        # Step 2: remove (1,2) -> 0 and 2 disconnected, so (0,2) is False
        # Step 3: remove 2nd (0,1) -> 0 and 1 disconnected, so (0,1) is False
        expected = [True, False, False]

        self.assertEqual(reverse_dsu_connectivity(n, edges, removals, queries), expected)
        self.assertEqual(bfs_recomputation_connectivity(n, edges, removals, queries), expected)
        self.assertEqual(dfs_recomputation_connectivity(n, edges, removals, queries), expected)
        self.assertEqual(brute_force_oracle_connectivity(n, edges, removals, queries), expected)

    def test_removing_more_copies_than_exist_raises_error(self):
        # Edge (0,1) exists 2 times; attempting 3 removals must raise ValueError
        data = {
            "n": 2,
            "edges": [[0, 1], [0, 1]],
            "removals": [[0, 1], [0, 1], [0, 1]],
            "queries": [[0, 1], [0, 1], [0, 1]],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_graph_data(data)
        self.assertIn("already removed", str(ctx.exception))


class TestAlgorithmsEdgeCases(unittest.TestCase):
    """Targeted edge case tests verifying all algorithms agree."""

    def test_problem_statement_sample(self):
        data = create_sample_dataset()
        res_dsu = reverse_dsu_connectivity(data["n"], data["edges"], data["removals"], data["queries"])
        res_bfs = bfs_recomputation_connectivity(data["n"], data["edges"], data["removals"], data["queries"])
        res_dfs = dfs_recomputation_connectivity(data["n"], data["edges"], data["removals"], data["queries"])
        res_orc = brute_force_oracle_connectivity(data["n"], data["edges"], data["removals"], data["queries"])

        expected = [False, False, False]
        self.assertEqual(res_dsu, expected)
        self.assertEqual(res_bfs, expected)
        self.assertEqual(res_dfs, expected)
        self.assertEqual(res_orc, expected)

    def test_empty_graph(self):
        n = 0
        edges = []
        removals = []
        queries = []
        self.assertEqual(reverse_dsu_connectivity(n, edges, removals, queries), [])
        self.assertEqual(bfs_recomputation_connectivity(n, edges, removals, queries), [])
        self.assertEqual(dfs_recomputation_connectivity(n, edges, removals, queries), [])
        self.assertEqual(brute_force_oracle_connectivity(n, edges, removals, queries), [])

    def test_single_vertex(self):
        n = 1
        edges = []
        removals = []
        queries = []
        self.assertEqual(reverse_dsu_connectivity(n, edges, removals, queries), [])
        self.assertEqual(bfs_recomputation_connectivity(n, edges, removals, queries), [])
        self.assertEqual(dfs_recomputation_connectivity(n, edges, removals, queries), [])

    def test_zero_removals(self):
        n = 3
        edges = [[0, 1], [1, 2]]
        removals = []
        queries = []
        self.assertEqual(reverse_dsu_connectivity(n, edges, removals, queries), [])
        self.assertEqual(bfs_recomputation_connectivity(n, edges, removals, queries), [])
        self.assertEqual(dfs_recomputation_connectivity(n, edges, removals, queries), [])

    def test_cycle_graph(self):
        n = 3
        edges = [[0, 1], [1, 2], [0, 2]]
        removals = [[0, 1], [0, 2]]
        queries = [[0, 1], [0, 1]]

        expected = [True, False]
        self.assertEqual(reverse_dsu_connectivity(n, edges, removals, queries), expected)
        self.assertEqual(bfs_recomputation_connectivity(n, edges, removals, queries), expected)
        self.assertEqual(dfs_recomputation_connectivity(n, edges, removals, queries), expected)

    def test_complete_graph_k4(self):
        n = 4
        edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        removals = [[0, 1], [0, 2], [0, 3]]
        queries = [[0, 3], [0, 3], [0, 3]]

        expected = [True, True, False]
        self.assertEqual(reverse_dsu_connectivity(n, edges, removals, queries), expected)
        self.assertEqual(bfs_recomputation_connectivity(n, edges, removals, queries), expected)
        self.assertEqual(dfs_recomputation_connectivity(n, edges, removals, queries), expected)

    def test_all_edges_removed(self):
        n = 3
        edges = [[0, 1], [1, 2]]
        removals = [[0, 1], [1, 2]]
        queries = [[0, 2], [1, 2]]
        expected = [False, False]
        self.assertEqual(reverse_dsu_connectivity(n, edges, removals, queries), expected)
        self.assertEqual(bfs_recomputation_connectivity(n, edges, removals, queries), expected)
        self.assertEqual(dfs_recomputation_connectivity(n, edges, removals, queries), expected)

    def test_disconnected_components(self):
        n = 6
        edges = [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]]
        removals = [[0, 1], [3, 4]]
        queries = [[0, 3], [1, 2]]
        expected = [False, True]
        self.assertEqual(reverse_dsu_connectivity(n, edges, removals, queries), expected)
        self.assertEqual(bfs_recomputation_connectivity(n, edges, removals, queries), expected)
        self.assertEqual(dfs_recomputation_connectivity(n, edges, removals, queries), expected)

    def test_boundary_vertices_and_self_query(self):
        n = 10
        edges = [[i, i + 1] for i in range(9)]
        removals = [[4, 5]]
        queries = [[0, 0]]  # self query must be True
        self.assertEqual(reverse_dsu_connectivity(n, edges, removals, queries), [True])
        self.assertEqual(bfs_recomputation_connectivity(n, edges, removals, queries), [True])
        self.assertEqual(dfs_recomputation_connectivity(n, edges, removals, queries), [True])

    def test_chain_graph_ten_thousand_vertices(self):
        # 10,000 vertices chain, 100 removals; checks that no recursion limits occur
        n = 10000
        edges = [[i, i + 1] for i in range(n - 1)]
        removals = [[500, 501], [1000, 1001], [5000, 5001]]
        queries = [[0, n - 1], [0, 500], [501, 1000]]
        # (0, n-1) disconnected after rem 1 -> False
        # (0, 500) remains connected throughout -> True
        # (501, 1000) remains connected after rem 3 -> True
        res_dsu = reverse_dsu_connectivity(n, edges, removals, queries, validate=False)
        self.assertEqual(res_dsu, [False, True, True])


class TestValidationAndErrorHandling(unittest.TestCase):
    """Test validation errors for invalid inputs according to multigraph specifications."""

    def test_parallel_initial_edges_allowed(self):
        # Parallel edges should now be accepted without error
        data = {
            "n": 3,
            "edges": [[0, 1], [0, 1], [1, 2]],
            "removals": [[0, 1]],
            "queries": [[0, 1]],
        }
        # Should not raise ValueError
        validate_graph_data(data)

    def test_invalid_removal_edge_not_in_graph(self):
        data = {
            "n": 3,
            "edges": [[0, 1], [1, 2]],
            "removals": [[0, 2]],
            "queries": [[0, 1]],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_graph_data(data)
        self.assertIn("does not exist in the graph", str(ctx.exception))

    def test_self_loop_in_edges(self):
        data = {
            "n": 3,
            "edges": [[0, 0]],
            "removals": [],
            "queries": [],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_graph_data(data)
        self.assertIn("Self-loop detected", str(ctx.exception))

    def test_vertex_out_of_bounds(self):
        data = {
            "n": 4,
            "edges": [[0, 4]],
            "removals": [],
            "queries": [],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_graph_data(data)
        self.assertIn("out of bounds", str(ctx.exception))

    def test_mismatched_queries_removals_count(self):
        data = {
            "n": 3,
            "edges": [[0, 1]],
            "removals": [[0, 1]],
            "queries": [[0, 1], [0, 2]],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_graph_data(data)
        self.assertIn("Queries count (2) must match removals count (1)", str(ctx.exception))


class TestRandomizedDifferentialTesting(unittest.TestCase):
    """Fuzz testing: execute Reverse DSU, BFS, DFS, and Brute-Force Oracle across 2000 trials."""

    def test_randomized_trials_multigraph(self):
        rng = random.Random(42)
        num_trials = 2000

        for trial in range(num_trials):
            n = rng.randint(2, 9)
            m = rng.randint(1, 16)
            k = rng.randint(1, m)

            # Generate random multigraph edges (parallel edges allowed)
            edges: list[list[int]] = []
            for _ in range(m):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                while u == v:
                    v = rng.randint(0, n - 1)
                edges.append([min(u, v), max(u, v)])

            # Sample k distinct edge instances to remove
            rem_indices = rng.sample(range(len(edges)), k)
            removals = [edges[idx] for idx in rem_indices]

            # Generate k queries (random vertex pairs, or endpoints of removals, or self-queries)
            queries: list[list[int]] = []
            for i in range(k):
                coin = rng.random()
                if coin < 0.4:
                    queries.append(list(removals[i]))
                elif coin < 0.7:
                    queries.append([rng.randint(0, n - 1), rng.randint(0, n - 1)])
                else:
                    v = rng.randint(0, n - 1)
                    queries.append([v, v])

            # Run all 4 algorithms
            dsu_res = reverse_dsu_connectivity(n, edges, removals, queries, validate=False)
            bfs_res = bfs_recomputation_connectivity(n, edges, removals, queries, validate=False)
            dfs_res = dfs_recomputation_connectivity(n, edges, removals, queries, validate=False)
            orc_res = brute_force_oracle_connectivity(n, edges, removals, queries)

            # Assert all 4 match identically
            self.assertEqual(
                dsu_res,
                bfs_res,
                f"DSU vs BFS mismatch at trial {trial} (N={n}, M={m}, K={k})",
            )
            self.assertEqual(
                bfs_res,
                dfs_res,
                f"BFS vs DFS mismatch at trial {trial} (N={n}, M={m}, K={k})",
            )
            self.assertEqual(
                dsu_res,
                orc_res,
                f"DSU vs Oracle mismatch at trial {trial} (N={n}, M={m}, K={k})",
            )


if __name__ == "__main__":
    unittest.main()
