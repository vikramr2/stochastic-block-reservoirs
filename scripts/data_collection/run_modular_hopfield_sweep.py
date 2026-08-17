"""
Evaluates modular Hopfield-style networks (sbm_hopfield_modular.py) across a
grid of bridge_gain (inter-module coupling strength) and mu (SBM bridge
density, from sweeps_mu_fine/), measuring whether a single cued block's
recall accuracy improves from light coupling to other (uncued) blocks versus
an isolated single-module baseline of the same size.

Usage:
    python run_modular_hopfield_sweep.py -i sweeps_mu_fine -o modular_hopfield_results.csv \
        [--num-blocks 10] [--block-size 100] [--num-patterns 3] \
        [--flip-fraction 0.2] [--bridge-gains "0.0,0.1,0.3,0.5,1.0,2.0"] \
        [--n-seeds 5] [--seed 42]

Author: Vikram Ramavarapu + Claude
"""

import csv
import re
import time
from pathlib import Path

import typer

from models.sbm_hopfield_modular import evaluate_isolated_module_recall, evaluate_single_module_recall

RUN_DIR_PATTERN = re.compile(r"^k(?P<k>[\d.]+)_mu(?P<mu>[\d.]+)$")

DEFAULT_BRIDGE_GAINS = "0.0,0.1,0.3,0.5,1.0,2.0"


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


def parse_values(csv_str: str) -> list:
    """Parse a comma-separated string of numbers into a sorted list of floats."""
    return sorted(float(v.strip()) for v in csv_str.split(",") if v.strip())


def main(
        sweep_dir: str = typer.Option(..., "--input_directory", "-i"),
        output_csv: str = typer.Option("modular_hopfield_results.csv", "--output_csv", "-o"),
        num_blocks: int = typer.Option(10, "--num-blocks"),
        block_size: int = typer.Option(100, "--block-size"),
        num_patterns: int = typer.Option(3, "--num-patterns"),
        flip_fraction: float = typer.Option(0.2, "--flip-fraction"),
        bridge_gains: str = typer.Option(DEFAULT_BRIDGE_GAINS, "--bridge-gains"),
        cued_block: int = typer.Option(0, "--cued-block"),
        n_seeds: int = typer.Option(5, "--n-seeds"),
        seed: int = typer.Option(42, "--seed")):
    """
    Evaluate modular Hopfield recall accuracy across a (mu, bridge_gain) grid,
    plus an isolated single-module baseline for each mu/seed combination.

    Args:
        sweep_dir: Directory containing k{k}_mu{mu} subdirectories from run_sbm_sweep.py.
        output_csv: Path to write the (k, mu, bridge_gain, seed, accuracy) results CSV.
        num_blocks: Number of blocks in each SBM graph.
        block_size: Number of nodes per block.
        num_patterns: Number of patterns stored per block.
        flip_fraction: Fraction of the cued block's units corrupted per trial.
        bridge_gains: Comma-separated list of bridge_gain values to sweep
            (bridge_gain=0.0 is treated as the isolated-module baseline).
        cued_block: Which block index to cue and score.
        n_seeds: Number of pattern/cue seeds to average per (mu, bridge_gain) combination.
        seed: Base random seed.
    """
    runs = find_sweep_runs(Path(sweep_dir))
    if not runs:
        raise typer.BadParameter(f"No k*_mu* runs found under {sweep_dir}")

    # At mu=1.0 the planted-partition formula puts zero probability on
    # within-block edges (p = (1-mu)*k/(block_size-1) = 0), so every block
    # has no internal connectivity at all -- there is no "isolated module"
    # to evaluate a within-block-only baseline against. Skip it rather than
    # crash or silently fabricate a baseline for an undefined case.
    skipped = [r for r in runs if r[1] >= 1.0]
    runs = [r for r in runs if r[1] < 1.0]
    if skipped:
        print(f"Skipping mu>=1.0 runs (no within-block edges, isolated baseline undefined): "
              f"{[r[1] for r in skipped]}")

    gains = parse_values(bridge_gains)
    print(f"Found {len(runs)} SBM graphs under {sweep_dir}")
    print(f"bridge_gain values: {gains}")

    results = []
    sweep_start = time.time()
    total_runs = len(runs) * len(gains) * n_seeds
    run_idx = 0

    for k, mu, edge_file in runs:
        for seed_offset in range(n_seeds):
            run_seed = seed + seed_offset

            # Isolated baseline, once per (mu, seed) -- doesn't depend on bridge_gain.
            iso_acc, _ = evaluate_isolated_module_recall(
                str(edge_file), num_blocks, block_size,
                num_patterns=num_patterns, flip_fraction=flip_fraction,
                cued_block=cued_block, seed=run_seed,
            )

            for gain in gains:
                run_idx += 1
                run_start = time.time()
                if gain == 0.0:
                    accuracy = iso_acc
                else:
                    accuracy, _ = evaluate_single_module_recall(
                        str(edge_file), num_blocks, block_size,
                        num_patterns=num_patterns, flip_fraction=flip_fraction,
                        bridge_gain=gain, cued_block=cued_block, seed=run_seed,
                    )
                elapsed = time.time() - run_start
                print(f"[{run_idx}/{total_runs}] mu={mu:.2f}, bridge_gain={gain}, "
                      f"seed={run_seed}: accuracy={accuracy:.4f} ({elapsed:.2f}s)")
                results.append({
                    "k": k, "mu": mu, "bridge_gain": gain, "seed": run_seed,
                    "accuracy": accuracy, "isolated_baseline": iso_acc,
                })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["k", "mu", "bridge_gain", "seed", "accuracy", "isolated_baseline"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Sweep completed in {time.time() - sweep_start:.2f}s")
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    typer.run(main)
