"""
Evaluates every SBM graph produced by run_sbm_sweep.py as a continuous-rate
attractor (Hopfield-style) network, storing random binary patterns via
SBM-masked Hebbian weights and measuring recall accuracy from noisy cues.
Intended to test whether SBM mixing parameter (mu) affects associative
memory / pattern-recall performance, replicated over multiple pattern/seed
draws per (k, mu) combination.

Usage:
    python run_hopfield_sweep.py -i sweeps_mu_fine -o hopfield_mu_results.csv \
        [--num-nodes 1000] [--num-patterns 5] [--flip-fraction 0.2] \
        [--n-seeds 5] [--seed 42]

With --n-seeds > 1, each (k, mu) graph is evaluated once per pattern/cue seed
(0, 1, ..., n_seeds-1), and the output CSV has one row per (k, mu, seed)
replicate so mean/std/SEM can be computed downstream.

Author: Vikram Ramavarapu + Claude
"""

import csv
import re
import time
from pathlib import Path

import typer

from models.sbm_hopfield import evaluate_sbm_hopfield

RUN_DIR_PATTERN = re.compile(r"^k(?P<k>[\d.]+)_mu(?P<mu>[\d.]+)$")


def find_sweep_runs(sweep_dir: Path):
    """
    Discover (k, mu, edge_file) tuples from a run_sbm_sweep.py output directory.

    Args:
        sweep_dir: Directory containing k{k}_mu{mu} subdirectories.

    Returns:
        List of (k, mu, edge_file_path) tuples, sorted by (k, mu).
    """
    runs = []
    for run_dir in sweep_dir.iterdir():
        if not run_dir.is_dir():
            continue
        match = RUN_DIR_PATTERN.match(run_dir.name)
        if not match:
            continue
        edge_file = run_dir / "syn_planted_sbm.tsv"
        if not edge_file.exists():
            continue
        k = float(match.group("k"))
        mu = float(match.group("mu"))
        runs.append((k, mu, edge_file))

    runs.sort(key=lambda r: (r[0], r[1]))
    return runs


def main(
        sweep_dir: str = typer.Option(..., "--input_directory", "-i"),
        output_csv: str = typer.Option("hopfield_mu_results.csv", "--output_csv", "-o"),
        num_nodes: int = typer.Option(1000, "--num-nodes"),
        num_patterns: int = typer.Option(5, "--num-patterns"),
        flip_fraction: float = typer.Option(0.2, "--flip-fraction"),
        synaptic_gain: float = typer.Option(2000.0, "--synaptic-gain"),
        settle_time_ms: float = typer.Option(200.0, "--settle-time-ms"),
        n_seeds: int = typer.Option(5, "--n-seeds"),
        seed: int = typer.Option(42, "--seed")):
    """
    Evaluate every SBM graph in sweep_dir as a Hopfield-style attractor
    network, measuring recall accuracy from corrupted cues.

    Args:
        sweep_dir: Directory containing k{k}_mu{mu} subdirectories from run_sbm_sweep.py.
        output_csv: Path to write the (k, mu, seed, accuracy) results CSV.
        num_nodes: Number of nodes in each SBM graph (network size).
        num_patterns: Number of random patterns to store per replicate.
        flip_fraction: Fraction of units corrupted in each recall cue.
        synaptic_gain: Scaling applied to recurrent synaptic input.
        settle_time_ms: Simulation duration per recall trial, in ms.
        n_seeds: Number of pattern/cue seeds to average per (k, mu) combination.
        seed: Base random seed.
    """
    runs = find_sweep_runs(Path(sweep_dir))
    if not runs:
        raise typer.BadParameter(f"No k*_mu* runs found under {sweep_dir}")

    print(f"Found {len(runs)} SBM graphs under {sweep_dir}")

    results = []
    sweep_start = time.time()
    total_runs = len(runs) * n_seeds
    run_idx = 0

    for k, mu, edge_file in runs:
        for seed_offset in range(n_seeds):
            run_idx += 1
            run_seed = seed + seed_offset
            run_start = time.time()
            accuracy, _ = evaluate_sbm_hopfield(
                str(edge_file), num_nodes,
                num_patterns=num_patterns, flip_fraction=flip_fraction,
                synaptic_gain=synaptic_gain, settle_time_ms=settle_time_ms,
                seed=run_seed,
            )
            elapsed = time.time() - run_start
            print(f"[{run_idx}/{total_runs}] k={k}, mu={mu:.2f}, seed={run_seed}: "
                  f"accuracy={accuracy:.4f} ({elapsed:.2f}s)")
            results.append({"k": k, "mu": mu, "seed": run_seed, "accuracy": accuracy})

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "mu", "seed", "accuracy"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Sweep completed in {time.time() - sweep_start:.2f}s")
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    typer.run(main)
