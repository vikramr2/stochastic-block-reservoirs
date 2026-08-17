"""
This script generates a synthetic planted-partition stochastic block model (SBM) graph
from a small set of parameters, using the graph-tool library.

Given the number of blocks, block size, target average degree k, and mixing
parameter mu, edge probabilities are derived as:

    within-block probability: p = (1 - mu) * k / (block_size - 1)
    between-block probability: q = mu * k / ((num_blocks - 1) * block_size)

Usage:
    python gen_planted_sbm.py -n <num_blocks> -s <block_size> -k <avg_degree> -m <mu> -o <output_directory> [-v]

Arguments:
    -n, --num_blocks: Number of blocks/communities.
    -s, --block_size: Number of nodes per block (all blocks have equal size).
    -k, --avg_degree: Target average node degree.
    -m, --mu: Mixing parameter in [0, 1]; fraction of edges that go between blocks.
    -o, --output_directory: Directory to save the generated graph.
    -v, --verbose: Enable verbose output with performance statistics and progress information.

Example:
    python gen_planted_sbm.py -n 10 -s 100 -k 8 -m 0.3 -o output_directory -v

Author: Vikram Ramavarapu + Claude
"""

import os
import time
from pathlib import Path
from typing import Set, Tuple

import graph_tool.all as gt  # type: ignore[import]
import numpy as np
import typer

from gen_sbm import (
    create_logger,
    extract_edges_threaded,
    monitor_resources,
    save_generated_graph,
)


def compute_edge_probabilities(num_blocks: int, block_size: int, k: float, mu: float) -> tuple:
    """
    Compute the within-block and between-block edge probabilities.

    Args:
        num_blocks: Number of blocks/communities.
        block_size: Number of nodes per block.
        k: Target average node degree.
        mu: Mixing parameter in [0, 1]; fraction of edges that go between blocks.

    Returns:
        Tuple of (p, q): within-block and between-block edge probabilities.
    """
    p = (1 - mu) * k / (block_size - 1)
    q = mu * k / ((num_blocks - 1) * block_size)
    return p, q


def build_block_probability_matrix(num_blocks: int, block_size: int, p: float, q: float) -> np.ndarray:
    """
    Build the num_blocks x num_blocks edge propensity matrix for a planted-partition SBM.

    graph_tool.generate_sbm expects probs[r, s] to be the *expected number of
    edges* between groups r and s (twice that number when r == s, for an
    undirected graph), not a per-pair probability. So the within/between
    block probabilities are converted to expected edge counts using the
    number of node pairs in each block/block-pair.

    Args:
        num_blocks: Number of blocks/communities.
        block_size: Number of nodes per block.
        p: Within-block edge probability.
        q: Between-block edge probability.

    Returns:
        np.ndarray: Symmetric matrix of expected edge counts between groups.
    """
    within_pairs = block_size * (block_size - 1) / 2
    between_pairs = block_size * block_size

    probs = np.full((num_blocks, num_blocks), q * between_pairs)
    np.fill_diagonal(probs, 2 * p * within_pairs)
    return probs


def generate_planted_sbm(
        num_blocks: int,
        block_size: int,
        k: float,
        mu: float,
        n_threads: int = -1,
        logger=None) -> Set[Tuple[int, int]]:
    """
    Generate a planted-partition SBM graph and return its edge list.

    Args:
        num_blocks: Number of blocks/communities.
        block_size: Number of nodes per block (all blocks have equal size).
        k: Target average node degree.
        mu: Mixing parameter in [0, 1]; fraction of edges that go between blocks.
        n_threads: Number of threads to use (-1 for all available cores).
        logger: Logger function for output (defaults to a silent logger).

    Returns:
        Set of (source, target) node-index edges, self-loop and multi-edge free.
    """
    if logger is None:
        logger = create_logger(False)

    if n_threads <= 0:
        n_threads = os.cpu_count() or 4

    if num_blocks < 2:
        raise ValueError("num_blocks must be at least 2")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    if not (0.0 <= mu <= 1.0):
        raise ValueError("mu must be between 0 and 1")

    p, q = compute_edge_probabilities(num_blocks, block_size, k, mu)
    logger(f"Computed within-block probability p={p:.6f}, between-block probability q={q:.6f}")

    if not (0.0 <= p <= 1.0):
        logger(f"Warning: within-block probability p={p:.6f} is out of [0, 1] range", always=True)
    if not (0.0 <= q <= 1.0):
        logger(f"Warning: between-block probability q={q:.6f} is out of [0, 1] range", always=True)

    probs_matrix = build_block_probability_matrix(num_blocks, block_size, p, q)

    num_nodes = num_blocks * block_size
    cluster_assignment = np.repeat(np.arange(num_blocks), block_size)
    node_idx_set = np.arange(num_nodes)

    logger(f"Total nodes: {num_nodes}")

    sbm_start_time = time.time()
    logger("Generating planted SBM graph...")
    N = gt.generate_sbm(cluster_assignment, probs_matrix)
    logger(f"SBM generation completed in {time.time() - sbm_start_time:.2f} seconds")
    logger(f"Raw graph: {N.num_vertices()} vertices, {N.num_edges()} edges (includes self-loops/multi-edges)")

    # generate_sbm samples a Poisson multigraph, which can include self-loops
    # and parallel edges; strip those at the graph-tool level before extraction.
    gt.remove_self_loops(N)
    gt.remove_parallel_edges(N)
    logger(f"Simplified graph: {N.num_vertices()} vertices, {N.num_edges()} edges")
    monitor_resources(logger)

    return extract_edges_threaded(N, node_idx_set, n_threads, logger)


def main(
        num_blocks: int = typer.Option(..., "--num_blocks", "-n"),
        block_size: int = typer.Option(..., "--block_size", "-s"),
        k: float = typer.Option(..., "--avg_degree", "-k"),
        mu: float = typer.Option(..., "--mu", "-m"),
        output_dir: str = typer.Option("", "--output_directory", "-o"),
        n_threads: int = typer.Option(-1, "--jobs", "-j"),
        verbose: bool = typer.Option(False, "--verbose", "-v")):
    """
    Generate a planted-partition SBM graph from num_blocks, block_size, average
    degree k, and mixing parameter mu.

    Args:
        num_blocks: Number of blocks/communities.
        block_size: Number of nodes per block (all blocks have equal size).
        k: Target average node degree.
        mu: Mixing parameter in [0, 1]; fraction of edges that go between blocks.
        output_dir: Output directory to save the generated graph.
        n_threads: Number of threads to use (-1 for all available cores).
        verbose: Whether to enable verbose output.
    """
    total_start_time = time.time()
    logger = create_logger(verbose)

    if n_threads <= 0:
        n_threads = os.cpu_count() or 4

    logger(f"Starting planted SBM generation with {n_threads} threads")
    logger(f"num_blocks={num_blocks}, block_size={block_size}, k={k}, mu={mu}")
    monitor_resources(logger)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    out_edge_file = output_path / 'syn_planted_sbm.tsv'

    N_edge_list = generate_planted_sbm(num_blocks, block_size, k, mu, n_threads, logger)

    save_generated_graph(N_edge_list, str(out_edge_file), n_threads, logger)

    print(f"Generated graph saved to: {out_edge_file}")
    logger(f"Total execution time: {time.time() - total_start_time:.2f} seconds", always=True)
    monitor_resources(logger)


if __name__ == "__main__":
    typer.run(main)
