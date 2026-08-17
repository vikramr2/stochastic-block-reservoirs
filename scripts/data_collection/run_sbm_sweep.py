"""
Sweeps the planted-partition SBM generator (gen_planted_sbm.py) over a grid of
average degree k and mixing parameter mu, saving one edge list per (k, mu)
combination. Intended to produce a batch of synthetic graphs for building and
comparing graph reservoirs across connectivity/mixing regimes.

Usage:
    python run_sbm_sweep.py -n <num_blocks> -s <block_size> -o <output_directory> \
        [-k "2,4,8,16,32,64"] [-m "0.0,0.1,...,1.0"] [-v]

Arguments:
    -n, --num_blocks: Number of blocks/communities.
    -s, --block_size: Number of nodes per block (all blocks have equal size).
    -o, --output_directory: Directory under which per-combination subfolders are saved.
    -k, --k-values: Comma-separated average degree values to sweep (default: log-spaced 2-64).
    -m, --mu-values: Comma-separated mixing parameter values to sweep (default: 0.0-1.0 step 0.1).
    -j, --jobs: Number of parallel jobs per graph generation (default: number of CPU cores).
    -v, --verbose: Enable verbose output with performance statistics and progress information.

Example:
    python run_sbm_sweep.py -n 10 -s 100 -o sweeps -v
    python run_sbm_sweep.py -n 10 -s 100 -o sweeps_mu_fine -k "6" -m "0.0,0.05,0.1,...,1.0" -v

Author: Vikram Ramavarapu + Claude
"""

import time
from pathlib import Path

import typer

from sbm.gen_sbm import create_logger, monitor_resources, save_generated_graph
from sbm.gen_planted_sbm import generate_planted_sbm

# Log-spaced average degrees; stays well under the p<=1/q<=1 ceiling implied
# by block_size=100 across the full mu range.
DEFAULT_K_VALUES = "2,4,8,16,32,64"

# Mixing parameter from 0 (fully assortative) to 1 (fully mixed), step 0.1.
DEFAULT_MU_VALUES = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"


def parse_values(csv: str) -> list:
    """Parse a comma-separated string of numbers into a sorted list of floats."""
    return sorted(float(v.strip()) for v in csv.split(",") if v.strip())


def main(
        num_blocks: int = typer.Option(..., "--num_blocks", "-n"),
        block_size: int = typer.Option(..., "--block_size", "-s"),
        output_dir: str = typer.Option(..., "--output_directory", "-o"),
        k_values: str = typer.Option(DEFAULT_K_VALUES, "--k-values", "-k"),
        mu_values: str = typer.Option(DEFAULT_MU_VALUES, "--mu-values", "-m"),
        n_threads: int = typer.Option(-1, "--jobs", "-j"),
        verbose: bool = typer.Option(False, "--verbose", "-v")):
    """
    Generate a planted-partition SBM graph for every (k, mu) combination in
    the k_values x mu_values grid, saving each to its own subdirectory.

    Args:
        num_blocks: Number of blocks/communities.
        block_size: Number of nodes per block (all blocks have equal size).
        output_dir: Output directory under which per-combination subfolders are saved.
        k_values: Comma-separated average degree values to sweep.
        mu_values: Comma-separated mixing parameter values to sweep.
        n_threads: Number of threads to use per graph (-1 for all available cores).
        verbose: Whether to enable verbose output.
    """
    import os

    if n_threads <= 0:
        n_threads = os.cpu_count() or 4

    K_VALUES = parse_values(k_values)
    MU_VALUES = parse_values(mu_values)

    logger = create_logger(verbose)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    total_runs = len(K_VALUES) * len(MU_VALUES)
    print(f"Sweeping {len(K_VALUES)} k values x {len(MU_VALUES)} mu values = {total_runs} graphs")
    print(f"k values: {K_VALUES}")
    print(f"mu values: {MU_VALUES}")

    sweep_start = time.time()
    run_idx = 0

    for k in K_VALUES:
        for mu in MU_VALUES:
            run_idx += 1
            run_start = time.time()

            run_dir = output_path / f"k{k}_mu{mu:.2f}"
            run_dir.mkdir(exist_ok=True, parents=True)
            out_edge_file = run_dir / "syn_planted_sbm.tsv"

            print(f"[{run_idx}/{total_runs}] Generating k={k}, mu={mu:.2f} -> {out_edge_file}")

            edge_list = generate_planted_sbm(num_blocks, block_size, k, mu, n_threads, logger)
            save_generated_graph(edge_list, str(out_edge_file), n_threads, logger)

            print(f"[{run_idx}/{total_runs}] Done in {time.time() - run_start:.2f}s "
                  f"({len(edge_list)} edges)")
            monitor_resources(logger)

    print(f"Sweep completed in {time.time() - sweep_start:.2f} seconds")


if __name__ == "__main__":
    typer.run(main)
