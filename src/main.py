"""Command-Line Interface (CLI) for Offline Connectivity After Edge Removals.

Supports:
1. Solving custom graph JSON inputs with DSU, BFS, DFS, or both.
2. Generating reproducible synthetic datasets (standard + adversarial topologies).
3. Executing the comprehensive benchmark suite across 11 graph topologies.
4. Verifying correctness via unit test and randomized fuzz test suite execution.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Dict, List

# Ensure package is resolvable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.algorithms import (  # type: ignore[import-not-found,import-untyped]
        reverse_dsu_connectivity,
        bfs_recomputation_connectivity,
        dfs_recomputation_connectivity,
    )
    from src.graph_utils import load_graph_from_json  # type: ignore[import-not-found,import-untyped]
    from src.generate_data import generate_all_datasets  # type: ignore[import-not-found,import-untyped]
except ImportError:
    from algorithms import (  # type: ignore[import-not-found,import-untyped]
        reverse_dsu_connectivity,
        bfs_recomputation_connectivity,
        dfs_recomputation_connectivity,
    )
    from graph_utils import load_graph_from_json  # type: ignore[import-not-found,import-untyped]
    from generate_data import generate_all_datasets  # type: ignore[import-not-found,import-untyped]


def print_banner() -> None:
    banner = """
========================================================================
    OFFLINE CONNECTIVITY AFTER EDGE REMOVALS (MULTIGRAPH SUPPORT)
    Data Structures and Algorithms II - Algorithmic Investigation
========================================================================
    """
    print(banner)


def run_solver(
    input_file: Path | str,
    algorithm: str = "both",
    output_file: str | None = None,
) -> None:
    """Run specified algorithm(s) on an input JSON file and display formatted results."""
    path = Path(input_file)
    print(f"[*] Loading input file: {path}")
    data = load_graph_from_json(path)

    n = data["n"]
    edges = data["edges"]
    removals = data["removals"]
    queries = data["queries"]
    k = len(removals)

    print(f"[*] Graph Summary: N = {n} vertices, M = {len(edges)} edges, K = {k} removals/queries")

    algo_choice = algorithm.lower().strip()

    dsu_res: List[bool] | None = None
    bfs_res: List[bool] | None = None
    dfs_res: List[bool] | None = None

    dsu_time = 0.0
    bfs_time = 0.0
    dfs_time = 0.0

    # 1. Reverse DSU
    if algo_choice in ("dsu", "both", "all"):
        tracemalloc.start()
        t0 = time.perf_counter()
        dsu_res = reverse_dsu_connectivity(n, edges, removals, queries, validate=False)
        t1 = time.perf_counter()
        _, dsu_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        dsu_time = (t1 - t0) * 1000.0
        print(f"[+] Reverse DSU finished in {dsu_time:.3f} ms | Peak memory: {dsu_mem / 1024:.2f} KB")

    # 2. BFS Recomputation
    if algo_choice in ("bfs", "both", "all"):
        tracemalloc.start()
        t0 = time.perf_counter()
        bfs_res = bfs_recomputation_connectivity(n, edges, removals, queries, validate=False)
        t1 = time.perf_counter()
        _, bfs_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        bfs_time = (t1 - t0) * 1000.0
        print(f"[+] BFS Recomputation finished in {bfs_time:.3f} ms | Peak memory: {bfs_mem / 1024:.2f} KB")

    # 3. DFS Recomputation
    if algo_choice in ("dfs", "all"):
        tracemalloc.start()
        t0 = time.perf_counter()
        dfs_res = dfs_recomputation_connectivity(n, edges, removals, queries, validate=False)
        t1 = time.perf_counter()
        _, dfs_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        dfs_time = (t1 - t0) * 1000.0
        print(f"[+] DFS Recomputation finished in {dfs_time:.3f} ms | Peak memory: {dfs_mem / 1024:.2f} KB")

    # Output Display
    if algo_choice in ("both", "all") and dsu_res is not None and bfs_res is not None:
        match = (dsu_res == bfs_res) and (dfs_res is None or dfs_res == dsu_res)
        print(f"\n[*] Correctness Cross-Check: {'PASS (100% Identical)' if match else 'FAIL (Results Differ)'}")
        if dsu_time > 0:
            speedup = bfs_time / dsu_time
            print(f"[*] Performance Ratio: Reverse DSU was {speedup:.2f}x faster than BFS Recomputation")

    # Show detailed table for small query counts (k <= 25)
    if k <= 25:
        headers = ["Step", "Removed Edge", "Query (u, v)"]
        if dsu_res is not None:
            headers.append("DSU")
        if bfs_res is not None:
            headers.append("BFS")
        if dfs_res is not None:
            headers.append("DFS")
        if (dsu_res is not None and bfs_res is not None) or (dsu_res is not None and dfs_res is not None):
            headers.append("Match")

        line_len = 10 + 16 + 16 + (9 * (len(headers) - 3))
        print("\n" + "-" * line_len)
        header_fmt = f"{headers[0]:<6} | {headers[1]:<14} | {headers[2]:<14}"
        for h in headers[3:]:
            header_fmt += f" | {h:<7}"
        print(header_fmt)
        print("-" * line_len)

        for i in range(k):
            rem_str = f"({removals[i][0]}, {removals[i][1]})"
            q_str = f"({queries[i][0]}, {queries[i][1]})"
            row_fmt = f"{i + 1:<6} | {rem_str:<14} | {q_str:<14}"
            if dsu_res is not None:
                row_fmt += f" | {str(dsu_res[i]):<7}"
            if bfs_res is not None:
                row_fmt += f" | {str(bfs_res[i]):<7}"
            if dfs_res is not None:
                row_fmt += f" | {str(dfs_res[i]):<7}"
            if dsu_res is not None and bfs_res is not None:
                row_fmt += f" | {'OK' if dsu_res[i] == bfs_res[i] else 'FAIL':<7}"
            elif dsu_res is not None and dfs_res is not None:
                row_fmt += f" | {'OK' if dsu_res[i] == dfs_res[i] else 'FAIL':<7}"
            print(row_fmt)
        print("-" * line_len)
    else:
        active_answers = dsu_res if dsu_res is not None else (bfs_res if bfs_res is not None else dfs_res)
        if active_answers is not None:
            true_cnt = sum(active_answers)
            print(f"[*] Query Summary (K={k}): {true_cnt} Connected (True), {k - true_cnt} Disconnected (False)")

    # Save output if requested
    if output_file:
        out_dict = {
            "input_file": str(path),
            "n": n,
            "removals_count": k,
            "dsu_answers": dsu_res,
            "bfs_answers": bfs_res,
            "dfs_answers": dfs_res,
            "dsu_time_ms": dsu_time if dsu_res is not None else None,
            "bfs_time_ms": bfs_time if bfs_res is not None else None,
            "dfs_time_ms": dfs_time if dfs_res is not None else None,
        }
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2)
        print(f"[*] Results saved to {out_p}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Offline Connectivity After Edge Removals: Reverse DSU vs BFS Recomputation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to input graph JSON file",
    )
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        choices=["dsu", "bfs", "dfs", "both", "all"],
        default="both",
        help="Algorithm to execute: dsu, bfs, dfs, both, or all (default: both)",
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run comprehensive benchmark suite across 11 graph topologies and generate graphs",
    )
    parser.add_argument(
        "--generate-data", "--generate", "-g",
        dest="generate_data",
        action="store_true",
        help="Generate all 11 benchmark datasets (standard + structured topologies)",
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Execute automated test suite to verify correctness",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save output results as JSON",
    )
    parser.add_argument(
        "--repeats", "-r",
        type=int,
        default=5,
        help="Benchmark timing repetitions per dataset (default: 5)",
    )

    args = parser.parse_args()

    print_banner()

    root_dir = Path(__file__).resolve().parent.parent

    if args.verify:
        import unittest
        print("[*] Running automated test suite...")
        loader = unittest.TestLoader()
        suite = loader.discover(str(root_dir / "tests"))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    if args.generate_data:
        data_dir = root_dir / "data"
        print(f"[*] Generating all 11 datasets in {data_dir}...")
        generate_all_datasets(data_dir)
        print("[+] All 11 datasets generated successfully.")

    if args.benchmark:
        try:
            from src.benchmark import run_benchmark_suite, plot_benchmark_results  # type: ignore[import-not-found,import-untyped]
        except ImportError:
            from benchmark import run_benchmark_suite, plot_benchmark_results  # type: ignore[import-not-found,import-untyped]
        data_dir = root_dir / "data"
        results_dir = root_dir / "results"

        datasets = [
            ("sample", data_dir / "sample.json"),
            ("small", data_dir / "small.json"),
            ("medium", data_dir / "medium.json"),
            ("large", data_dir / "large.json"),
            ("extra_large", data_dir / "extra_large.json"),
            ("chain", data_dir / "chain.json"),
            ("star", data_dir / "star.json"),
            ("disconnected", data_dir / "disconnected.json"),
            ("multigraph", data_dir / "multigraph.json"),
            ("all_removed", data_dir / "all_removed.json"),
            ("bfs_friendly", data_dir / "bfs_friendly.json"),
        ]

        if not (data_dir / "sample.json").is_file():
            print("[*] Benchmark datasets not found. Generating now...")
            generate_all_datasets(data_dir)

        bench_results = run_benchmark_suite(
            # pyrefly: ignore[bad-argument-type]
            datasets,
            results_dir / "benchmark_results.csv",
            repeats=args.repeats,
        )
        plot_benchmark_results(bench_results, results_dir)
        print("[+] Benchmark complete. Graphs and CSV saved in results/.")
        return

    if args.input:
        run_solver(args.input, algorithm=args.algorithm, output_file=args.output)
    elif not args.generate_data and not args.verify:
        sample_path = root_dir / "data" / "sample.json"
        if not sample_path.is_file():
            generate_all_datasets(root_dir / "data")
        print("[*] No arguments provided. Running default demonstration on data/sample.json:")
        run_solver(sample_path, algorithm="both")
        print("\nTip: Run with --help to see all available CLI options.")


if __name__ == "__main__":
    main()
