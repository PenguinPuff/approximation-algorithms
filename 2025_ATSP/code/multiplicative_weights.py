# algorithms/multiplicative_weights.py
import numpy as np
from math import exp, log as ln
from typing import Dict, Tuple

from algorithms.kirchhoff import (
    build_laplacian,
    _pseudo_inverse_diagonal,
    compute_all_marginals,
)

def find_gamma_multiplicative_weights(
    n: int,
    z_star: Dict[Tuple[int, int], float],
    epsilon: float = 0.2,
    max_iterations: int = 10000,
    verbose: bool = True,
) -> Dict[Tuple[int, int], float]:
    z_star = {(min(i, j), max(i, j)): v for (i, j), v in z_star.items()}

    edges   = list(z_star.keys())
    z_vals  = np.array([z_star[e] for e in edges])
    hi      = (1.0 + epsilon) * z_vals      # violation threshold
    step    = ln(1.0 + epsilon / 2.0)       # fixed MW step size

    gamma = {edge: 0.0 for edge in edges}

    if verbose:
        print(
            f"  MW: n={n}, edges={len(edges)}, epsilon={epsilon}, "
            f"max_z*={z_vals.max():.4f}, step={step:.4f}"
        )

    for iteration in range(1, max_iterations + 1):
        lambda_weights = {e: exp(gamma[e]) for e in edges}
        L     = build_laplacian(n, lambda_weights)
        L_pinv = _pseudo_inverse_diagonal(L)
        q_vals = np.empty(len(edges))
        for k, (i, j) in enumerate(edges):
            r_eff     = float(L_pinv[i, i] + L_pinv[j, j] - 2.0 * L_pinv[i, j])
            q_vals[k] = lambda_weights[(i, j)] * max(r_eff, 0.0)

        over      = q_vals > hi
        n_violated = int(over.sum())

        if verbose and (iteration % 100 == 0 or iteration == 1):
            under_count = int((q_vals < z_vals / (1.0 + epsilon)).sum())
            print(
                f"  MW iter {iteration:5d}: {n_violated} over-threshold"
                f" (of {len(edges)})  [under={under_count} for info only]"
            )

        if n_violated == 0:
            if verbose:
                print(f"  MW converged after {iteration} iterations.")
            break

        gamma_arr = np.array([gamma[e] for e in edges])
        gamma_arr[over] -= step
        for k, e in enumerate(edges):
            gamma[e] = float(gamma_arr[k])

    else:
        if verbose:
            print(f"  MW did not converge after {max_iterations} iterations (using best gamma so far).")

    return gamma

def verify_marginals(
    n: int,
    gamma: Dict[Tuple[int, int], float],
    z_star: Dict[Tuple[int, int], float],
    verbose: bool = True,
) -> bool:
    z_star = {(min(i, j), max(i, j)): v for (i, j), v in z_star.items()}
    gamma  = {(min(i, j), max(i, j)): v for (i, j), v in gamma.items()}

    lambda_weights = {e: exp(gamma.get(e, 0.0)) for e in z_star}
    marginals      = compute_all_marginals(n, lambda_weights)

    errors = []
    for edge, z_e in z_star.items():
        q_e     = marginals.get(edge, 0.0)
        err     = abs(q_e - z_e)
        rel_err = err / z_e if z_e > 1e-9 else err
        errors.append((edge, q_e, z_e, err, rel_err))

    errors.sort(key=lambda x: x[3], reverse=True)

    if verbose:
        print("\n" + "=" * 70)
        print("MARGINAL VERIFICATION")
        print("=" * 70)
        print(f"{'Edge':<14} {'q(e)':<10} {'z*(e)':<10} {'|error|':<10} {'rel%':<10}")
        print("-" * 70)
        for edge, q_e, z_e, err, rel_err in errors[:10]:
            print(f"{str(edge):<14} {q_e:<10.6f} {z_e:<10.6f} {err:<10.6f} {rel_err:<10.2%}")

        errs     = [e[3] for e in errors]
        rel_errs = [e[4] for e in errors]
        print(f"\n  Max |error|:    {max(errs):.6f}")
        print(f"  Avg |error|:    {np.mean(errs):.6f}")
        print(f"  Median |error|: {np.median(errs):.6f}")

        within_20 = sum(1 for r in rel_errs if r < 0.20)
        within_30 = sum(1 for r in rel_errs if r < 0.30)
        m = len(errors)
        print(f"  Within 20% rel: {within_20}/{m} ({100*within_20/m:.1f}%)")
        print(f"  Within 30% rel: {within_30}/{m} ({100*within_30/m:.1f}%)")

    within_20_frac = sum(1 for e in errors if e[4] < 0.20) / max(len(errors), 1)
    return within_20_frac >= 0.70
