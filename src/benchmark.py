"""Benchmarking and experimental performance analysis module.

Measures:
1. Runtime (median, mean, min, max, stdev across repeated runs using time.perf_counter).
2. Peak memory allocation (using tracemalloc in isolated memory runs).
3. Correctness status: ensures Reverse DSU answers match BFS baseline.

Datasets evaluated (11 total):
- Standard scaling: Sample, Small, Medium, Large, Extra Large
- Structured & Adversarial:
  - Chain: Deep path topology
  - Star: High-degree hub node
  - Disconnected: Multi-component forest
  - Multigraph: Dense graph with parallel edges
  - All Removed: Total edge destruction (K = M)
  - BFS Friendly: Localized queries where BFS terminates rapidly

Outputs:
- results/benchmark_results.csv
- results/runtime_comparison.png
- results/memory_comparison.png
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import statistics
import sys  
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple

# Ensure package root is resolvable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set writable cache for matplotlib and non-interactive Agg backend (avoids macOS GUI/sandbox warnings)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.algorithms import reverse_dsu_connectivity, bfs_recomputation_connectivity  # type: ignore[import-not-found,import-untyped]
    from src.graph_utils import load_graph_from_json  # type: ignore[import-not-found,import-untyped]
except ImportError:
    from algorithms import reverse_dsu_connectivity, bfs_recomputation_connectivity  # type: ignore[import-not-found,import-untyped]
    from graph_utils import load_graph_from_json  # type: ignore[import-not-found,import-untyped]


def measure_runtime_and_correctness(
    data: Dict[str, Any],
    repeats: int = 5,
) -> Dict[str, Any]:
    """Measure runtime for Reverse DSU and BFS Recomputation across repeated runs.

    Also cross-verifies correctness between the two methods on run 1.

    Args:
        data: Graph dataset dictionary.
        repeats: Number of timing iterations (minimum 5 recommended).

    Returns:
        Dictionary containing timing statistics, memory metrics, and correctness flag.
    """
    n = data["n"]
    edges = data["edges"]
    removals = data["removals"]
    queries = data["queries"]
    m = len(edges)
    k = len(removals)

    # 1. Correctness check on first run
    dsu_ans = reverse_dsu_connectivity(n, edges, removals, queries, validate=False)
    bfs_ans = bfs_recomputation_connectivity(n, edges, removals, queries, validate=False)
    is_correct = (dsu_ans == bfs_ans)
    if not is_correct:
        mismatches = sum(1 for a, b in zip(dsu_ans, bfs_ans) if a != b)
        print(f"[WARNING] Correctness failure! {mismatches} queries differed.")

    # 2. Benchmark DSU Runtime (isolated from memory tracing)
    dsu_times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        reverse_dsu_connectivity(n, edges, removals, queries, validate=False)
        t1 = time.perf_counter()
        dsu_times.append((t1 - t0) * 1000.0)  # Convert to milliseconds

    # 3. Benchmark BFS Runtime
    bfs_times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        bfs_recomputation_connectivity(n, edges, removals, queries, validate=False)
        t1 = time.perf_counter()
        bfs_times.append((t1 - t0) * 1000.0)  # Convert to milliseconds

    # 4. Measure Peak Memory in isolated runs
    tracemalloc.start()
    tracemalloc.reset_peak()
    reverse_dsu_connectivity(n, edges, removals, queries, validate=False)
    _, dsu_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    dsu_peak_kb = dsu_peak_bytes / 1024.0

    tracemalloc.start()
    tracemalloc.reset_peak()
    bfs_recomputation_connectivity(n, edges, removals, queries, validate=False)
    _, bfs_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    bfs_peak_kb = bfs_peak_bytes / 1024.0

    # Compute statistics
    dsu_median = statistics.median(dsu_times)
    dsu_mean = statistics.mean(dsu_times)
    dsu_stdev = statistics.stdev(dsu_times) if len(dsu_times) > 1 else 0.0

    bfs_median = statistics.median(bfs_times)
    bfs_mean = statistics.mean(bfs_times)
    bfs_stdev = statistics.stdev(bfs_times) if len(bfs_times) > 1 else 0.0

    speedup = bfs_median / dsu_median if dsu_median > 0 else 1.0

    return {
        "n": n,
        "m": m,
        "k": k,
        "correct": is_correct,
        "repeats": repeats,
        "dsu_median_ms": round(dsu_median, 4),
        "dsu_mean_ms": round(dsu_mean, 4),
        "dsu_min_ms": round(min(dsu_times), 4),
        "dsu_max_ms": round(max(dsu_times), 4),
        "dsu_stdev_ms": round(dsu_stdev, 4),
        "dsu_peak_kb": round(dsu_peak_kb, 2),
        "bfs_median_ms": round(bfs_median, 4),
        "bfs_mean_ms": round(bfs_mean, 4),
        "bfs_min_ms": round(min(bfs_times), 4),
        "bfs_max_ms": round(max(bfs_times), 4),
        "bfs_stdev_ms": round(bfs_stdev, 4),
        "bfs_peak_kb": round(bfs_peak_kb, 2),
        "speedup_factor": round(speedup, 2),
    }


def run_benchmark_suite(
    datasets: List[Tuple[str, Path | str]],
    output_csv: Path | str,
    repeats: int = 5,
) -> List[Dict[str, Any]]:
    """Execute benchmarks across a list of named dataset files.

    Args:
        datasets: List of tuples (dataset_label, file_path).
        output_csv: Destination CSV file.
        repeats: Timing repetitions per dataset.

    Returns:
        List of benchmark summary dictionaries.
    """
    results: List[Dict[str, Any]] = []

    print(f"\n{'='*82}")
    print(f"{'BENCHMARK SUITE: REVERSE DSU vs BFS RECOMPUTATION':^82}")
    print(f"{'='*82}")
    print(f"{'Dataset':<14} | {'N':>6} | {'M':>7} | {'K':>6} | {'DSU (ms)':>10} | {'BFS (ms)':>10} | {'Speedup':>9} | {'Correct':>7}")
    print(f"{'-'*82}")

    for label, path in datasets:
        p = Path(path)
        if not p.is_file():
            print(f"[SKIP] File not found: {p}")
            continue

        data = load_graph_from_json(p)
        stats = measure_runtime_and_correctness(data, repeats=repeats)
        stats["dataset"] = label
        results.append(stats)

        status_str = "PASS" if stats["correct"] else "FAIL"
        print(
            f"{label:<14} | {stats['n']:>6} | {stats['m']:>7} | {stats['k']:>6} | "
            f"{stats['dsu_median_ms']:>10.3f} | {stats['bfs_median_ms']:>10.3f} | "
            f"{stats['speedup_factor']:>8.1f}x | {status_str:>7}"
        )

    print(f"{'='*82}\n")

    # Save to CSV
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "n", "m", "k", "correct", "repeats",
        "dsu_median_ms", "dsu_mean_ms", "dsu_min_ms", "dsu_max_ms", "dsu_stdev_ms", "dsu_peak_kb",
        "bfs_median_ms", "bfs_mean_ms", "bfs_min_ms", "bfs_max_ms", "bfs_stdev_ms", "bfs_peak_kb",
        "speedup_factor",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[SAVED] Benchmark results written to {csv_path}")
    return results


def plot_benchmark_results(
    results: List[Dict[str, Any]],
    output_dir: Path | str,
) -> Tuple[Path, Path]:
    """Generate high-quality Matplotlib comparison plots from actual benchmark data.

    Plots:
    1. Runtime Comparison (Log scale & Linear scale dual view).
    2. Peak Memory Comparison.

    Args:
        results: Benchmark result dictionaries.
        output_dir: Directory where PNG plots will be saved.

    Returns:
        Tuple of (runtime_plot_path, memory_plot_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [r["dataset"].replace("_", " ").title() for r in results]
    x_indices = list(range(len(labels)))
    dsu_medians = [r["dsu_median_ms"] for r in results]
    bfs_medians = [r["bfs_median_ms"] for r in results]
    dsu_mem = [r["dsu_peak_kb"] for r in results]
    bfs_mem = [r["bfs_peak_kb"] for r in results]

    # --- Plot 1: Runtime Comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    width = 0.38
    x = [i for i in x_indices]
    x_dsu = [i - width / 2 for i in x]
    x_bfs = [i + width / 2 for i in x]

    # Subplot 1: Log scale runtime bar chart
    ax1.bar(x_dsu, dsu_medians, width=width, label="Reverse DSU (O((M+K)·α(N)))", color="#1E3A8A", edgecolor="black", alpha=0.9)
    ax1.bar(x_bfs, bfs_medians, width=width, label="BFS Recomputation (O(K(N+M)))", color="#DC2626", edgecolor="black", alpha=0.9)
    ax1.set_yscale("log")
    ax1.set_xlabel("Dataset Topology", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Median Wall-Clock Runtime (ms) [Log Scale]", fontsize=11, fontweight="bold")
    ax1.set_title("Runtime Comparison across 11 Workloads (Log Scale)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5, rotation=35, ha="right")
    ax1.grid(True, which="both", ls="--", alpha=0.35)
    ax1.legend(fontsize=9.5, loc="upper left")

    # Add speedup annotations on log plot
    for i, r in enumerate(results):
        sp = r["speedup_factor"]
        y_pos = max(dsu_medians[i], bfs_medians[i]) * 1.45
        label_text = f"{sp:.1f}x" if sp >= 1.0 else f"{sp:.2f}x"
        color = "#047857" if sp >= 1.0 else "#B91C1C"
        ax1.text(x[i], y_pos, label_text, ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=color)

    # Subplot 2: Linear scale runtime line chart
    ax2.plot(x, dsu_medians, marker="o", linewidth=2.2, markersize=6.5, color="#1E3A8A", label="Reverse DSU")
    ax2.plot(x, bfs_medians, marker="s", linewidth=2.2, markersize=6.5, color="#DC2626", label="BFS Recomputation")
    ax2.set_xlabel("Dataset Topology", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Median Wall-Clock Runtime (ms) [Linear Scale]", fontsize=11, fontweight="bold")
    ax2.set_title("Runtime Scaling Trend (Linear Scale)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5, rotation=35, ha="right")
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.legend(fontsize=9.5)

    fig.suptitle("Offline Connectivity: Performance and Scalability Analysis Across Graph Topologies", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    runtime_path = out_dir / "runtime_comparison.png"
    plt.savefig(runtime_path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] Runtime plot saved to {runtime_path}")

    # --- Plot 2: Peak Memory Comparison ---
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(x_dsu, dsu_mem, width=width, label="Reverse DSU Peak Heap", color="#0284C7", edgecolor="black", alpha=0.85)
    ax.bar(x_bfs, bfs_mem, width=width, label="BFS Recomputation Peak Heap", color="#EA580C", edgecolor="black", alpha=0.85)

    ax.set_xlabel("Dataset Topology", fontsize=11, fontweight="bold")
    ax.set_ylabel("Peak Dynamic Memory Allocation (Kilobytes)", fontsize=11, fontweight="bold")
    ax.set_title("Peak Dynamic Memory Footprint (tracemalloc) Across Graph Topologies", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=35, ha="right")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.legend(fontsize=10)

    for i in range(len(results)):
        ax.text(x_dsu[i], dsu_mem[i] + max(dsu_mem) * 0.015, f"{dsu_mem[i]:.0f} KB", ha="center", va="bottom", fontsize=7, rotation=30)
        ax.text(x_bfs[i], bfs_mem[i] + max(bfs_mem) * 0.015, f"{bfs_mem[i]:.0f} KB", ha="center", va="bottom", fontsize=7, rotation=30)

    plt.tight_layout()
    memory_path = out_dir / "memory_comparison.png"
    plt.savefig(memory_path, dpi=300)
    plt.close(fig)
    print(f"[SAVED] Memory plot saved to {memory_path}")

    return runtime_path, memory_path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"

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

    bench_results = run_benchmark_suite(datasets, results_dir / "benchmark_results.csv", repeats=5)
    plot_benchmark_results(bench_results, results_dir)

