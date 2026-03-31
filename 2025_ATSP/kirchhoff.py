# algorithms/kirchhoff.py
import numpy as np
from typing import Dict, Tuple

def build_laplacian(n: int, edges: Dict[Tuple[int, int], float]) -> np.ndarray:
    L = np.zeros((n, n), dtype=np.float64)
    for (i, j), weight in edges.items():
        if i > j:
            i, j = j, i
        L[i, j] -= weight
        L[j, i] -= weight
        L[i, i] += weight
        L[j, j] += weight
    return L

def _pseudo_inverse(L: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(L)
    threshold = eigvals.max() * 1e-10
    nonzero = eigvals > threshold
    L_pinv = np.zeros_like(L)
    for k in np.where(nonzero)[0]:
        v = eigvecs[:, k]
        L_pinv += np.outer(v, v) / eigvals[k]
    return L_pinv

_pseudo_inverse_diagonal = _pseudo_inverse

def compute_all_marginals(
    n: int,
    edges: Dict[Tuple[int, int], float],
) -> Dict[Tuple[int, int], float]:
    norm = {(min(i, j), max(i, j)): w for (i, j), w in edges.items()}

    L = build_laplacian(n, norm)
    L_pinv = _pseudo_inverse(L)

    marginals = {}
    for (i, j), lam_e in norm.items():
        r_eff = float(L_pinv[i, i] + L_pinv[j, j] - 2.0 * L_pinv[i, j])
        marginals[(i, j)] = float(lam_e * r_eff)

    return marginals

def compute_edge_marginal(
    n: int,
    edges: Dict[Tuple[int, int], float],
    edge: Tuple[int, int],
) -> float:
    marginals = compute_all_marginals(n, edges)
    i, j = min(edge), max(edge)
    return marginals.get((i, j), 0.0)

def contract_edge(
    n: int,
    edges: Dict[Tuple[int, int], float],
    edge: Tuple[int, int],
) -> Dict[Tuple[int, int], float]:
    i, j = edge
    if i > j:
        i, j = j, i
    new_edges: Dict[Tuple[int, int], float] = {}
    for (u, v), weight in edges.items():
        if u > v:
            u, v = v, u
        if (u, v) == (i, j):
            continue
        if u == j:
            u = i
        if v == j:
            v = i
        if u == v:
            continue
        if u > j:
            u -= 1
        if v > j:
            v -= 1
        if u > v:
            u, v = v, u
        new_edges[(u, v)] = new_edges.get((u, v), 0.0) + weight
    return new_edges

def count_spanning_trees(n: int, edges: Dict[Tuple[int, int], float]) -> float:
    if n <= 1:
        return 1.0
    L = build_laplacian(n, edges)
    return abs(np.linalg.det(L[1:, 1:]))
