"""
Evaluates every SBM graph produced by run_sbm_sweep.py as a reservoir on
MNIST digit classification, to study how the SBM parameters k (average
degree) and mu (mixing parameter) affect classification accuracy.

For each sweeps/k{k}_mu{mu}/syn_planted_sbm.tsv graph, builds a reservoir
whose recurrent weights follow the graph's connectivity, trains a ridge
readout on mean-pooled reservoir states over a shared MNIST train/test
split, and records test accuracy. Results are written to a CSV.

Usage:
    python run_mnist_sweep.py -i sweeps -o mnist_sweep_results.csv \
        [--num-nodes 1000] [--train-size 5000] [--test-size 1000] \
        [--n-seeds 1] [--seed 42]

With --n-seeds > 1, each (k, mu) graph is evaluated once per reservoir weight
seed (0, 1, ..., n_seeds-1), and the output CSV has one row per (k, mu, seed)
replicate so mean/std/SEM can be computed downstream.

Author: Vikram Ramavarapu + Claude
"""

import csv
import re
import time
from pathlib import Path

import numpy as np
import typer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from models.sbm_reservoir import evaluate_sbm_reservoir, threshold_activation

RUN_DIR_PATTERN = re.compile(r"^k(?P<k>[\d.]+)_mu(?P<mu>[\d.]+)$")


def load_mnist(train_size: int, test_size: int, seed: int):
    """
    Load and subsample MNIST, returning a shared train/test split.

    Args:
        train_size: Number of training images to sample.
        test_size: Number of test images to sample.
        seed: Random seed for the stratified split.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels).
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    images = mnist.data.astype(np.float64) / 255.0
    labels = mnist.target.astype(int)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels,
        train_size=train_size, test_size=test_size,
        random_state=seed, stratify=labels,
    )
    return train_images, train_labels, test_images, test_labels


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
        output_csv: str = typer.Option("mnist_sweep_results.csv", "--output_csv", "-o"),
        num_nodes: int = typer.Option(1000, "--num-nodes"),
        train_size: int = typer.Option(5000, "--train-size"),
        test_size: int = typer.Option(1000, "--test-size"),
        sr: float = typer.Option(0.9, "--spectral-radius"),
        ridge: float = typer.Option(1.0, "--ridge"),
        threshold: bool = typer.Option(False, "--threshold/--tanh"),
        n_seeds: int = typer.Option(1, "--n-seeds"),
        seed: int = typer.Option(42, "--seed")):
    """
    Evaluate every SBM graph in sweep_dir as a reservoir on MNIST classification.

    Args:
        sweep_dir: Directory containing k{k}_mu{mu} subdirectories from run_sbm_sweep.py.
        output_csv: Path to write the (k, mu, seed, accuracy) results CSV.
        num_nodes: Number of nodes in each SBM graph (reservoir size).
        train_size: Number of MNIST training images to use.
        test_size: Number of MNIST test images to use.
        sr: Target spectral radius for reservoir weights.
        ridge: L2 regularization strength for the Ridge readout.
        threshold: Use the threshold-like activation (Rodriguez et al. 2019) instead of tanh.
        n_seeds: Number of reservoir weight seeds to average per (k, mu) combination.
        seed: Base random seed for MNIST split and reservoir weight sampling.
    """
    activation = threshold_activation if threshold else None
    runs = find_sweep_runs(Path(sweep_dir))
    if not runs:
        raise typer.BadParameter(f"No k*_mu* runs found under {sweep_dir}")

    print(f"Found {len(runs)} SBM graphs under {sweep_dir}")
    print(f"Loading MNIST (train={train_size}, test={test_size})...")
    train_images, train_labels, test_images, test_labels = load_mnist(train_size, test_size, seed)

    results = []
    sweep_start = time.time()
    total_runs = len(runs) * n_seeds
    run_idx = 0

    for k, mu, edge_file in runs:
        for seed_offset in range(n_seeds):
            run_idx += 1
            run_seed = seed + seed_offset
            run_start = time.time()
            accuracy, _ = evaluate_sbm_reservoir(
                str(edge_file), num_nodes,
                train_images, train_labels, test_images, test_labels,
                sr=sr, ridge=ridge, activation=activation, seed=run_seed,
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
