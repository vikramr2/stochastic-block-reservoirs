"""
Builds a MODULAR continuous-rate attractor network: each SBM block stores
its own independent set of local patterns (its own energy landscape), and
inter-block (bridge) edges carry a separate, gain-scaled Hebbian coupling
that pairs each block's patterns by index (block A's pattern p <-> block B's
pattern p), rather than one pattern spread across all units.

This tests a different hypothesis than sbm_hopfield.py's whole-network
recall: does light inter-module coupling (controlled by bridge_gain and the
SBM mixing parameter mu) help a SINGLE cued module recall its own pattern
more accurately than an isolated module of the same size, by letting
uncued modules' bridge-driven activity feed back useful context? This is
the "communicating modules" alternative to post-hoc ensembling (see
papers/coevolution.pdf, where sub-networks are combined only after
independent evolution/inference).

Author: Vikram Ramavarapu + Claude
"""

from typing import Dict, List, Tuple

import numpy as np
from brian2 import (
    NeuronGroup, Synapses, StateMonitor, prefs,
    ms, run, start_scope,
)
from scipy.sparse import csr_matrix  # type: ignore[import]

from models.sbm_hopfield import corrupt_pattern, load_sbm_edge_list, run_rate_recall, score_recall

prefs.codegen.target = "numpy"


def block_of(node_idx: np.ndarray, block_size: int) -> np.ndarray:
    """
    Map node indices to their block index, given the fixed block_size layout
    used by gen_planted_sbm.py (block b occupies nodes [b*block_size, (b+1)*block_size)).

    Args:
        node_idx: Array of node indices.
        block_size: Number of nodes per block.

    Returns:
        np.ndarray of block indices, same shape as node_idx.
    """
    return node_idx // block_size


def generate_block_patterns(
        num_blocks: int, block_size: int, num_patterns: int, seed: int = None
) -> Dict[int, np.ndarray]:
    """
    Generate an independent set of random bipolar patterns for each block.

    Args:
        num_blocks: Number of blocks.
        block_size: Number of nodes per block.
        num_patterns: Number of patterns stored per block (same count in every block).
        seed: Random seed.

    Returns:
        Dict mapping block index -> array of shape (num_patterns, block_size).
    """
    rng = np.random.default_rng(seed)
    return {
        b: rng.choice([-1.0, 1.0], size=(num_patterns, block_size))
        for b in range(num_blocks)
    }


def build_modular_weights(
        block_patterns: Dict[int, np.ndarray],
        adjacency: csr_matrix,
        num_blocks: int,
        block_size: int,
        bridge_gain: float,
) -> np.ndarray:
    """
    Build the full weight matrix for a modular Hopfield-style network: each
    block's own patterns determine its within-block Hebbian weights at full
    strength, and bridge (between-block) weights are computed by pairing
    each block's patterns by index with its neighbor's, scaled by bridge_gain.

    Args:
        block_patterns: Dict mapping block index -> (num_patterns, block_size) array.
        adjacency: Sparse binary adjacency matrix over all num_blocks*block_size units.
        num_blocks: Number of blocks.
        block_size: Number of nodes per block.
        bridge_gain: Scaling applied to inter-block Hebbian weights (0 = no coupling).

    Returns:
        np.ndarray of shape (N, N), N = num_blocks * block_size: full weight matrix.
    """
    num_units = num_blocks * block_size
    mask = adjacency.toarray() > 0
    W = np.zeros((num_units, num_units))

    for b in range(num_blocks):
        sl = slice(b * block_size, (b + 1) * block_size)
        patterns_b = block_patterns[b]
        W_local = (patterns_b.T @ patterns_b) / block_size
        np.fill_diagonal(W_local, 0.0)
        W[sl, sl] = W_local

    for b1 in range(num_blocks):
        for b2 in range(b1 + 1, num_blocks):
            sl1 = slice(b1 * block_size, (b1 + 1) * block_size)
            sl2 = slice(b2 * block_size, (b2 + 1) * block_size)
            patterns_1 = block_patterns[b1]
            patterns_2 = block_patterns[b2]
            # Pair patterns by index: bridge weight w_ij = mean_p(pattern_1_p[i] * pattern_2_p[j]).
            W_bridge = (patterns_1.T @ patterns_2) / block_size * bridge_gain
            W[sl1, sl2] = W_bridge
            W[sl2, sl1] = W_bridge.T

    return W * mask


def evaluate_single_module_recall(
        edge_file: str,
        num_blocks: int,
        block_size: int,
        num_patterns: int = 5,
        flip_fraction: float = 0.2,
        bridge_gain: float = 0.1,
        cued_block: int = 0,
        seed: int = None,
        **sim_kwargs) -> Tuple[float, List[float]]:
    """
    Cue a single block with a corrupted local pattern (other blocks start at
    rest) in a modular network, and measure that block's own recall accuracy
    after the full network's recurrent dynamics settle.

    Args:
        edge_file: Path to the SBM edge list file.
        num_blocks: Number of blocks.
        block_size: Number of nodes per block.
        num_patterns: Number of patterns stored per block.
        flip_fraction: Fraction of the cued block's units to corrupt.
        bridge_gain: Scaling applied to inter-block Hebbian weights.
        cued_block: Which block index to cue and score.
        seed: Random seed for patterns and corruption.
        **sim_kwargs: Extra keyword arguments passed to run_rate_recall.

    Returns:
        Tuple of (mean recall accuracy over the cued block's patterns,
        list of per-pattern recall accuracies).
    """
    num_units = num_blocks * block_size
    adjacency = load_sbm_edge_list(edge_file, num_units)
    block_patterns = generate_block_patterns(num_blocks, block_size, num_patterns, seed=seed)
    weights = build_modular_weights(block_patterns, adjacency, num_blocks, block_size, bridge_gain)

    sl = slice(cued_block * block_size, (cued_block + 1) * block_size)
    accuracies = []
    for p_idx in range(num_patterns):
        target_local = block_patterns[cued_block][p_idx]
        cue_local = corrupt_pattern(target_local, flip_fraction,
                                     seed=None if seed is None else seed + p_idx)

        full_cue = np.zeros(num_units)
        full_cue[sl] = cue_local

        rates = run_rate_recall(weights, full_cue, **sim_kwargs)
        accuracies.append(score_recall(rates[sl], target_local))

    return float(np.mean(accuracies)), accuracies


def evaluate_isolated_module_recall(
        edge_file: str,
        num_blocks: int,
        block_size: int,
        num_patterns: int = 5,
        flip_fraction: float = 0.2,
        cued_block: int = 0,
        seed: int = None,
        **sim_kwargs) -> Tuple[float, List[float]]:
    """
    Baseline: evaluate recall for a single block in ISOLATION (no other
    blocks, no bridges at all) using only its own within-block SBM edges and
    its own patterns. Equivalent to build_modular_weights with bridge_gain=0
    restricted to just the cued block's own units, but simulated as a
    standalone smaller network so it isn't diluted/slowed by other blocks'
    (inert) dynamics.

    Args:
        edge_file: Path to the SBM edge list file.
        num_blocks: Number of blocks (used only to slice out the block's own edges).
        block_size: Number of nodes per block.
        num_patterns: Number of patterns stored in this block.
        flip_fraction: Fraction of units to corrupt in each cue.
        cued_block: Which block index to isolate and evaluate.
        seed: Random seed for patterns and corruption.
        **sim_kwargs: Extra keyword arguments passed to run_rate_recall.

    Returns:
        Tuple of (mean recall accuracy, list of per-pattern recall accuracies).
    """
    num_units = num_blocks * block_size
    adjacency = load_sbm_edge_list(edge_file, num_units)
    block_patterns = generate_block_patterns(num_blocks, block_size, num_patterns, seed=seed)

    sl = slice(cued_block * block_size, (cued_block + 1) * block_size)
    local_adjacency = adjacency.toarray()[sl, sl]
    patterns_b = block_patterns[cued_block]
    W_local = (patterns_b.T @ patterns_b) / block_size
    np.fill_diagonal(W_local, 0.0)
    W_local = W_local * (local_adjacency > 0)

    accuracies = []
    for p_idx in range(num_patterns):
        target = patterns_b[p_idx]
        cue = corrupt_pattern(target, flip_fraction, seed=None if seed is None else seed + p_idx)
        rates = run_rate_recall(W_local, cue, **sim_kwargs)
        accuracies.append(score_recall(rates, target))

    return float(np.mean(accuracies)), accuracies
