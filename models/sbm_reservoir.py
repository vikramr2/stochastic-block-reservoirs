"""
Builds reservoirpy Reservoir nodes from synthetic SBM graphs (as produced by
gen_planted_sbm.py / run_sbm_sweep.py) and evaluates them on MNIST digit
classification via ridge regression on mean-pooled reservoir states.

Each MNIST image is fed to the reservoir as a 28-timestep sequence (one row
of the image per timestep). The reservoir states over those 28 timesteps are
mean-pooled into a single feature vector, and a Ridge readout is trained on
one-hot labels (argmax at inference) to perform 10-way digit classification.

Author: Vikram Ramavarapu + Claude
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from reservoirpy.nodes import Reservoir, Ridge
from scipy.sparse import csr_matrix  # type: ignore[import]


def load_sbm_edge_list(edge_file: str, num_nodes: int) -> csr_matrix:
    """
    Load an SBM edge list (as saved by gen_planted_sbm.py) into a sparse
    binary adjacency matrix.

    Args:
        edge_file: Path to a whitespace/tab-separated edge list file.
        num_nodes: Total number of nodes in the graph.

    Returns:
        csr_matrix: Symmetric binary adjacency matrix of shape (num_nodes, num_nodes).
    """
    rows, cols = [], []
    with open(edge_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            source, target = line.split("\t")
            rows.append(int(source))
            cols.append(int(target))

    # Mirror edges to make the matrix symmetric (edge list stores each edge once).
    all_rows = rows + cols
    all_cols = cols + rows
    data = np.ones(len(all_rows), dtype=np.float64)

    return csr_matrix((data, (all_rows, all_cols)), shape=(num_nodes, num_nodes))


def threshold_activation(z: np.ndarray, a: float = 1.0, b: float = 1.0, c: float = 1.0,
                          k: float = 10.0, d: float = 0.0) -> np.ndarray:
    """
    Threshold-like (steep logistic) activation function, matching Eq. 3 of
    Rodriguez, Izquierdo & Ahn (2019), "Optimal modularity and memory
    capacity of neural reservoirs". Unlike tanh, this only excites a node
    once enough input has accumulated (a "complex contagion" dynamic), which
    the paper found to be necessary for community structure to affect
    reservoir performance -- tanh/linear reservoirs showed no such effect.

    Args:
        z: Pre-activation input.
        a, b, c, k, d: Shape parameters (paper defaults: a=b=c=1, k=10, d=0).

    Returns:
        Activated output, same shape as z.
    """
    return a / (b + np.exp(-k * (z - c))) - d


def build_reservoir_weights(adjacency: csr_matrix, seed: int = None) -> csr_matrix:
    """
    Convert a binary SBM adjacency matrix into signed reservoir weights.

    Each existing edge is assigned a weight drawn from a standard normal
    distribution; non-edges remain zero. reservoirpy rescales the resulting
    matrix to the target spectral radius when passed as `W` to a Reservoir.

    Args:
        adjacency: Sparse binary adjacency matrix.
        seed: Random seed for weight sampling.

    Returns:
        csr_matrix: Sparse matrix with signed random weights on existing edges.
    """
    rng = np.random.default_rng(seed)
    adjacency = adjacency.tocoo()
    weights = rng.standard_normal(size=adjacency.data.shape[0])
    return csr_matrix((weights, (adjacency.row, adjacency.col)), shape=adjacency.shape)


def images_to_sequences(images: np.ndarray) -> list:
    """
    Convert a batch of flattened 28x28 MNIST images into row-scan sequences.

    Args:
        images: Array of shape (num_images, 784), pixel values in [0, 255] or [0, 1].

    Returns:
        List of arrays, each of shape (28, 28): one sequence per image, where
        each timestep is one row of the image (28-dim vector).
    """
    num_images = images.shape[0]
    return [images[i].reshape(28, 28) for i in range(num_images)]


def compute_pooled_states(reservoir: Reservoir, sequences: list, workers: int = -1) -> np.ndarray:
    """
    Run the reservoir over each image sequence and mean-pool states over time.

    Args:
        reservoir: A reservoirpy Reservoir node (already configured with W).
        sequences: List of (timesteps, input_dim) arrays, one per image.
        workers: Number of parallel workers for reservoir.run (-1 for all).

    Returns:
        np.ndarray of shape (num_images, reservoir_units): mean-pooled states.
    """
    all_states = reservoir.run(sequences, workers=workers)
    return np.array([states.mean(axis=0) for states in all_states])


def build_sbm_reservoir(
        edge_file: str,
        num_nodes: int,
        sr: float = 0.9,
        input_scaling: float = 1.0,
        input_connectivity: float = 0.1,
        lr: float = 1.0,
        activation=None,
        seed: int = None) -> Reservoir:
    """
    Build a reservoirpy Reservoir whose recurrent weight matrix W is derived
    from an SBM graph's adjacency structure.

    Args:
        edge_file: Path to the SBM edge list file.
        num_nodes: Total number of nodes in the graph (reservoir size).
        sr: Target spectral radius after rescaling W.
        input_scaling: Scaling applied to the input weights Win.
        input_connectivity: Connectivity fraction for the (default random) Win.
        lr: Leak rate.
        activation: Reservoir activation function (defaults to reservoirpy's tanh).
        seed: Random seed for weight sampling and reservoir initialization.

    Returns:
        Reservoir: A reservoirpy Reservoir node with SBM-derived connectivity.
    """
    adjacency = load_sbm_edge_list(edge_file, num_nodes)
    W = build_reservoir_weights(adjacency, seed=seed)

    kwargs = dict(
        units=num_nodes,
        sr=sr,
        lr=lr,
        input_scaling=input_scaling,
        input_connectivity=input_connectivity,
        W=W,
        input_dim=28,
        seed=seed,
    )
    if activation is not None:
        kwargs["activation"] = activation

    return Reservoir(**kwargs)


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    One-hot encode integer labels.

    Args:
        labels: Array of shape (num_samples,) with integer class labels.
        num_classes: Number of classes.

    Returns:
        np.ndarray of shape (num_samples, num_classes).
    """
    encoded = np.zeros((labels.shape[0], num_classes))
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def evaluate_sbm_reservoir(
        edge_file: str,
        num_nodes: int,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        test_images: np.ndarray,
        test_labels: np.ndarray,
        sr: float = 0.9,
        input_scaling: float = 1.0,
        input_connectivity: float = 0.1,
        lr: float = 1.0,
        ridge: float = 1.0,
        activation=None,
        seed: int = None,
        workers: int = -1) -> Tuple[float, np.ndarray]:
    """
    Build an SBM-derived reservoir, train a ridge readout on mean-pooled
    states, and evaluate classification accuracy on held-out MNIST digits.

    Args:
        edge_file: Path to the SBM edge list file.
        num_nodes: Total number of nodes in the graph (reservoir size).
        train_images: Training images, shape (n_train, 784).
        train_labels: Training labels, shape (n_train,), integers 0-9.
        test_images: Test images, shape (n_test, 784).
        test_labels: Test labels, shape (n_test,), integers 0-9.
        sr: Target spectral radius after rescaling W.
        input_scaling: Scaling applied to the input weights Win.
        input_connectivity: Connectivity fraction for the (default random) Win.
        lr: Leak rate.
        ridge: L2 regularization strength for the Ridge readout.
        activation: Reservoir activation function (defaults to reservoirpy's tanh).
        seed: Random seed for weight sampling and reservoir initialization.
        workers: Number of parallel workers for reservoir.run (-1 for all).

    Returns:
        Tuple of (test_accuracy, confusion-relevant predictions array).
    """
    reservoir = build_sbm_reservoir(
        edge_file, num_nodes, sr=sr, input_scaling=input_scaling,
        input_connectivity=input_connectivity, lr=lr, activation=activation, seed=seed,
    )

    train_sequences = images_to_sequences(train_images)
    test_sequences = images_to_sequences(test_images)

    train_states = compute_pooled_states(reservoir, train_sequences, workers=workers)
    reservoir.reset()
    test_states = compute_pooled_states(reservoir, test_sequences, workers=workers)

    readout = Ridge(ridge=ridge)
    readout.fit(train_states, one_hot(train_labels))

    predictions = readout.run(test_states)
    predicted_labels = np.argmax(predictions, axis=1)

    accuracy = float(np.mean(predicted_labels == test_labels))
    return accuracy, predicted_labels
