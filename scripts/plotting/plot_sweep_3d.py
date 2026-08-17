"""
Plots MNIST classification accuracy from run_mnist_sweep.py's results as a 3D
surface over average degree (k) and mixing parameter (mu).

Usage:
    python plot_sweep_3d.py -i mnist_sweep_results.csv -o sweep_3d.png

Author: Vikram Ramavarapu + Claude
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)


def main(
        input_csv: str = typer.Option("mnist_sweep_results.csv", "--input_csv", "-i"),
        output_file: str = typer.Option("sweep_3d.png", "--output_file", "-o"),
        log_k: bool = typer.Option(True, "--log-k/--linear-k")):
    """
    Render a 3D surface plot of accuracy over (k, mu) from a sweep results CSV.

    Args:
        input_csv: Path to the CSV with columns k, mu, accuracy.
        output_file: Path to save the rendered figure.
        log_k: Plot the k axis on a log2 scale (k values are log-spaced).
    """
    df = pd.read_csv(input_csv)

    k_values = np.sort(df["k"].unique())
    mu_values = np.sort(df["mu"].unique())

    # Build a (k, mu) -> accuracy grid for plot_surface.
    accuracy_grid = df.pivot(index="k", columns="mu", values="accuracy").loc[k_values, mu_values].to_numpy()

    mu_grid, k_grid = np.meshgrid(mu_values, k_values)
    k_plot = np.log2(k_grid) if log_k else k_grid

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        k_plot, mu_grid, accuracy_grid,
        cmap="viridis", edgecolor="k", linewidth=0.3, alpha=0.9,
    )

    # Overlay the actual sample points.
    ax.scatter(
        k_plot.ravel(), mu_grid.ravel(), accuracy_grid.ravel(),
        color="black", s=12, depthshade=True,
    )

    ax.set_xlabel("k (average degree)" + (" [log2]" if log_k else ""))
    ax.set_ylabel("mu (mixing parameter)")
    ax.set_zlabel("test accuracy")
    ax.set_title("MNIST classification accuracy across SBM reservoir parameters")

    if log_k:
        ax.set_xticks(np.log2(k_values))
        ax.set_xticklabels([str(int(k)) for k in k_values])

    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="accuracy")

    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    print(f"Saved 3D sweep plot to {output_file}")


if __name__ == "__main__":
    typer.run(main)
