"""
Builds a continuous-rate attractor network (a Brian2-simulated analogue of a
classical Hopfield network) wired using an SBM graph's adjacency structure,
with Hebbian-derived synaptic weights encoding a set of stored binary
patterns. Recurrent connectivity is masked to only the edges present in the
SBM graph (rather than a fully-connected Hebbian weight matrix), letting us
study how community structure (mu) and sparsity (k) affect associative
memory / pattern recall.

A continuous-rate model (dr/dt = (-r + tanh(x))/tau, x = recurrent input +
cue) is used rather than spiking LIF units: LIF + raw Hebbian weights has no
stable regime for pattern completion without careful excitation/inhibition
balancing (recurrent drive is either too weak to matter or causes runaway
firing) -- the continuous-rate model is the standard, well-behaved way to
study Hopfield-style attractor dynamics and was validated first before
considering a spiking extension.

Author: Vikram Ramavarapu + Claude
"""

from typing import List, Tuple

import numpy as np
from brian2 import (
    NeuronGroup, Synapses, StateMonitor, prefs,
    ms, run, start_scope,
)
from scipy.sparse import csr_matrix  # type: ignore[import]

# Avoid repeated failed Cython compile attempts (x86_64 conda env on arm64
# macOS has a compiler/architecture mismatch); numpy codegen is slower but
# always available and avoids per-run compile overhead across a sweep.
prefs.codegen.target = "numpy"


def load_sbm_edge_list(edge_file: str, num_nodes: int) -> csr_matrix:
    """
    Load an SBM edge list (as saved by gen_planted_sbm.py) into a sparse
    binary adjacency matrix.

    Args:
        edge_file: Path to a tab-separated edge list file.
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

    all_rows = rows + cols
    all_cols = cols + rows
    data = np.ones(len(all_rows), dtype=np.float64)

    return csr_matrix((data, (all_rows, all_cols)), shape=(num_nodes, num_nodes))


def generate_patterns(num_patterns: int, num_units: int, seed: int = None) -> np.ndarray:
    """
    Generate random bipolar (+1/-1) patterns to store in the network.

    Args:
        num_patterns: Number of patterns to generate.
        num_units: Number of units per pattern (network size).
        seed: Random seed.

    Returns:
        np.ndarray of shape (num_patterns, num_units), entries in {-1, +1}.
    """
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(num_patterns, num_units))


def hebbian_weights(patterns: np.ndarray, adjacency: csr_matrix) -> np.ndarray:
    """
    Compute Hebbian outer-product weights from stored patterns, masked to
    only the edges present in the SBM adjacency matrix.

    Args:
        patterns: Array of shape (num_patterns, num_units), entries in {-1, +1}.
        adjacency: Sparse binary adjacency matrix of shape (num_units, num_units).

    Returns:
        np.ndarray of shape (num_units, num_units): masked Hebbian weight matrix.
    """
    num_units = patterns.shape[1]
    W = (patterns.T @ patterns) / num_units
    np.fill_diagonal(W, 0.0)
    mask = adjacency.toarray() > 0
    return W * mask


def corrupt_pattern(pattern: np.ndarray, flip_fraction: float, seed: int = None) -> np.ndarray:
    """
    Return a noisy copy of a pattern with a fraction of units randomly flipped.

    Args:
        pattern: Array of shape (num_units,), entries in {-1, +1}.
        flip_fraction: Fraction of units to flip.
        seed: Random seed.

    Returns:
        np.ndarray: Corrupted copy of the pattern.
    """
    rng = np.random.default_rng(seed)
    corrupted = pattern.copy()
    num_flips = int(round(flip_fraction * len(pattern)))
    flip_idx = rng.choice(len(pattern), size=num_flips, replace=False)
    corrupted[flip_idx] *= -1
    return corrupted


def run_rate_recall(
        weights: np.ndarray,
        cue_pattern: np.ndarray,
        settle_time_ms: float = 200.0,
        tau_ms: float = 10.0,
        synaptic_gain: float = 2000.0,
) -> np.ndarray:
    """
    Simulate a continuous-rate attractor network cued with a (possibly noisy)
    pattern and return each unit's steady-state rate.

    Each unit's continuous firing rate r follows dr/dt = (-r + tanh(x)) / tau,
    where x = synaptic_gain * (recurrent Hebbian input). The cue sets the
    initial rates r(0) = cue_pattern and is not held on afterward (matching
    classical Hopfield recall: a brief cue sets the start state, then pure
    recurrent dynamics settle to the nearest attractor). Holding the cue on
    for the whole simulation would let it dominate x indefinitely, since
    Hebbian recurrent input on a sparse SBM graph is naturally small in
    magnitude (~0.01-0.02) relative to a unit-strength cue -- using the cue
    only as an initial condition avoids needing an implausibly large gain to
    ever let recurrence override it.

    Args:
        weights: (num_units, num_units) SBM-masked Hebbian weight matrix.
        cue_pattern: (num_units,) bipolar cue pattern in {-1, +1}.
        settle_time_ms: Total simulation duration in ms.
        tau_ms: Rate time constant in ms.
        synaptic_gain: Scaling applied to recurrent synaptic input.

    Returns:
        np.ndarray of shape (num_units,): final rate r for each unit.
    """
    start_scope()
    num_units = weights.shape[0]

    eqs = """
    dr/dt = (-r + tanh(x)) / tau : 1
    x = synaptic_gain * x_rec : 1
    x_rec : 1
    tau : second
    """
    G = NeuronGroup(num_units, eqs, method="euler",
                     namespace={"synaptic_gain": synaptic_gain})
    G.tau = tau_ms * ms
    G.r = cue_pattern

    pre_idx, post_idx = np.nonzero(weights)
    syn_weights = weights[pre_idx, post_idx]

    # Continuously recompute the recurrent drive (not event-driven): a
    # "summed variable" gives x_rec = sum_j w_ij * r_j at every timestep,
    # matching classical Hopfield's continuous update rule.
    S = Synapses(G, G, model="w : 1\nx_rec_post = w * r_pre : 1 (summed)")
    S.connect(i=pre_idx, j=post_idx)
    S.w = syn_weights

    rate_mon = StateMonitor(G, "r", record=True)
    run(settle_time_ms * ms)

    return rate_mon.r[:, -1]


def score_recall(rates: np.ndarray, target_pattern: np.ndarray) -> float:
    """
    Convert steady-state rates to a recalled bipolar pattern (sign) and score
    similarity against the target pattern.

    Args:
        rates: (num_units,) steady-state rates from run_rate_recall, in [-1, 1].
        target_pattern: (num_units,) bipolar pattern in {-1, +1} to compare against.

    Returns:
        float: fraction of units matching the target pattern (0 to 1).
    """
    recalled = np.where(rates > 0, 1.0, -1.0)
    return float(np.mean(recalled == target_pattern))


def evaluate_sbm_hopfield(
        edge_file: str,
        num_nodes: int,
        num_patterns: int = 5,
        flip_fraction: float = 0.2,
        seed: int = None,
        **sim_kwargs) -> Tuple[float, List[float]]:
    """
    Build a continuous-rate attractor network on an SBM graph, store random
    patterns via masked Hebbian weights, and evaluate recall accuracy from
    noisy cues.

    Args:
        edge_file: Path to the SBM edge list file.
        num_nodes: Total number of nodes in the graph (network size).
        num_patterns: Number of random patterns to store.
        flip_fraction: Fraction of units to corrupt in each cue.
        seed: Random seed for patterns and corruption.
        **sim_kwargs: Extra keyword arguments passed to run_rate_recall.

    Returns:
        Tuple of (mean_recall_accuracy, list of per-pattern recall accuracies).
    """
    adjacency = load_sbm_edge_list(edge_file, num_nodes)
    patterns = generate_patterns(num_patterns, num_nodes, seed=seed)
    weights = hebbian_weights(patterns, adjacency)

    accuracies = []
    for i, pattern in enumerate(patterns):
        cue = corrupt_pattern(pattern, flip_fraction, seed=None if seed is None else seed + i)
        rates = run_rate_recall(weights, cue, **sim_kwargs)
        accuracies.append(score_recall(rates, pattern))

    return float(np.mean(accuracies)), accuracies
