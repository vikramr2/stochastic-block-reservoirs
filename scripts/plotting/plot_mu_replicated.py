"""
Plots mean MNIST classification accuracy (+/- SEM) across replicated seeds
for a fixed-k, swept-mu reservoir experiment, comparing threshold-like vs.
tanh activation functions (as produced by run_mnist_sweep.py --n-seeds > 1).

Usage:
    python plot_mu_replicated.py \
        --threshold-csv mnist_mu_fine_threshold_replicated.csv \
        --tanh-csv mnist_mu_fine_tanh_replicated.csv \
        -o mu_replicated.png

Author: Vikram Ramavarapu + Claude
"""

import matplotlib.pyplot as plt
import pandas as pd
import typer


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-(k, mu) replicate accuracies into mean and standard error.

    Args:
        df: DataFrame with columns k, mu, seed, accuracy.

    Returns:
        DataFrame with columns mu, mean, sem, n, sorted by mu.
    """
    grouped = df.groupby("mu")["accuracy"].agg(["mean", "std", "count"]).reset_index()
    grouped["sem"] = grouped["std"] / grouped["count"] ** 0.5
    grouped["sem"] = grouped["sem"].fillna(0.0)
    return grouped.sort_values("mu")


def main(
        threshold_csv: str = typer.Option(..., "--threshold-csv"),
        tanh_csv: str = typer.Option(..., "--tanh-csv"),
        output_file: str = typer.Option("mu_replicated.png", "--output_file", "-o")):
    """
    Plot mean +/- SEM accuracy vs. mu for threshold-like and tanh reservoirs.

    Args:
        threshold_csv: CSV of replicated results with the threshold activation.
        tanh_csv: CSV of replicated results with the tanh activation.
        output_file: Path to save the rendered figure.
    """
    threshold_df = summarize(pd.read_csv(threshold_csv))
    tanh_df = summarize(pd.read_csv(tanh_csv))

    n_seeds = int(threshold_df["count"].max())

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.errorbar(
        threshold_df["mu"], threshold_df["mean"], yerr=threshold_df["sem"],
        marker="o", markersize=5, capsize=3, linewidth=1.8,
        label="threshold-like activation", color="#2a78d6",
    )
    ax.errorbar(
        tanh_df["mu"], tanh_df["mean"], yerr=tanh_df["sem"],
        marker="s", markersize=5, capsize=3, linewidth=1.8,
        label="tanh (default)", color="#eb6834",
    )

    ax.set_xlabel("mu (mixing parameter)")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"MNIST accuracy vs. mixing parameter (mean +/- SEM, n={n_seeds} seeds/point)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    print(f"Saved plot to {output_file}")

    peak_thresh = threshold_df.loc[threshold_df["mean"].idxmax()]
    peak_tanh = tanh_df.loc[tanh_df["mean"].idxmax()]
    print(f"Threshold peak: mu={peak_thresh['mu']:.2f}, "
          f"accuracy={peak_thresh['mean']:.4f} +/- {peak_thresh['sem']:.4f}")
    print(f"Tanh peak: mu={peak_tanh['mu']:.2f}, "
          f"accuracy={peak_tanh['mean']:.4f} +/- {peak_tanh['sem']:.4f}")


if __name__ == "__main__":
    typer.run(main)
