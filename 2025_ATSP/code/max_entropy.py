# algorithms/max_entropy.py
import time
import math
from typing import Dict, List, Tuple
import numpy as np

def symmetrize_lp_solution(
    x_star: Dict[Tuple[int, int], float],
    n: int,
) -> Dict[Tuple[int, int], float]:
    factor = (n - 1) / n
    z_star: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            val = factor * (x_star.get((i, j), 0.0) + x_star.get((j, i), 0.0))
            if val > 1e-9:
                z_star[(i, j)] = val
    return z_star

def _earliest_arrival(
    path: List[int],
    C: List[List[float]],
    service: List[float],
    r_times: List[float],
    start_at: float = 0.0,
) -> List[float]:
    if not path:
        return []
    t = max(r_times[path[0]], start_at)
    arr = [t]
    for k in range(1, len(path)):
        i, j = path[k - 1], path[k]
        t = max(r_times[j], t + service[i] + C[i][j])
        arr.append(t)
    return arr

def _compute_tw_status(
    tour: List[int],
    arrival_times: List[float],
    d_times: List[float],
) -> List[str]:
    INF = 10**9
    status = []
    for node, t in zip(tour, arrival_times):
        if d_times[node] >= INF:
            status.append("no_tw")
        elif t <= d_times[node] + 1e-6:
            status.append("ok")
        else:
            status.append("violate")
    return status

def solve_with_max_entropy(
    nodes: List[str],
    C: List[List[float]],
    service: List[float],
    r_times: List[float],
    d_times: List[float],
    departure_sec: float,
    route_id: str,
    samples: int = None,
    gamma_epsilon: float = 0.2,
    gamma_max_iter: int = 10000,
    seed: int = 0,
) -> Dict:
    from lp.held_karp import solve_held_karp
    from algorithms.multiplicative_weights import (
        find_gamma_multiplicative_weights,
        verify_marginals,
    )
    from algorithms.lambda_random_tree import sample_lambda_random_tree_robust
    from algorithms.tree_conversion import (
        orient_tree_optimally,
        compute_degree_imbalance,
        make_eulerian_graph,
        euler_tour,
        shortcut_to_tour,
    )
    from algorithms.min_cost_balance import min_cost_balance

    n = len(nodes)
    INF = 10**9

    if samples is None:
        samples = 2 * math.ceil(math.log(n))

    print(f"\n{'='*70}", flush=True)
    print(f"  O(log n / log log n)-approximation  (Asadpour et al.)")
    print(f"  Route: {route_id}  |  Nodes: {n}  |  Samples: {samples}")
    print(f"  epsilon={gamma_epsilon}  max_iter={gamma_max_iter}")
    print(f"{'='*70}", flush=True)

    wall_start = time.perf_counter()

    # [1] Held-Karp LP  (extreme point via crossover)
    print("\n[1] Solving Held-Karp LP relaxation...")
    t0 = time.perf_counter()
    x_star, lp_bound = solve_held_karp(C, verbose=False)
    lp_time = time.perf_counter() - t0
    print(f"    LP lower bound (OPT_HK) = {lp_bound:.2f}  [{lp_time:.1f}s]")

    # [2] Symmetrise  (eq. 3.5)
    t0 = time.perf_counter()
    z_star = symmetrize_lp_solution(x_star, n)
    sym_time = time.perf_counter() - t0

    if z_star:
        max_z = max(z_star.values())
        avg_z = sum(z_star.values()) / len(z_star)
        print(f"\n[2] Symmetrised z*: {len(z_star)} edges  [{sym_time*1000:.0f}ms]")
        print(f"    max z*={max_z:.4f}  avg z*={avg_z:.4f}")
        if max_z > 1.0 + 1e-6:
            raise RuntimeError(
                f"max z*={max_z:.6f} > 1. Held-Karp LP did not return an extreme "
                "point solution. Ensure Crossover != 0 in held_karp.py."
            )

    # [3] Multiplicative weights -> gamma
    print(f"\n[3] Finding gamma via multiplicative weights (eps={gamma_epsilon})...")
    t0 = time.perf_counter()
    gamma = find_gamma_multiplicative_weights(
        n=n,
        z_star=z_star,
        epsilon=gamma_epsilon,
        max_iterations=gamma_max_iter,
        verbose=True,
    )
    gamma_time = time.perf_counter() - t0
    print(f"    Done in {gamma_time:.1f}s")

    # [4] Verify marginals
    print("\n[4] Verifying marginals...")
    t0 = time.perf_counter()
    marginals_ok = verify_marginals(n, gamma, z_star, verbose=True)
    verify_time = time.perf_counter() - t0
    print(f"    Verification done in {verify_time:.2f}s")

    # [5] Sample lambda-random spanning trees
    print(f"\n[5] Sampling {samples} lambda-random trees...")
    lambda_weights = {e: math.exp(g) for e, g in gamma.items()}

    best_tour = None
    best_cost = INF
    sample_times = []

    for k in range(samples):
        ts = time.perf_counter()

        tree = sample_lambda_random_tree_robust(n, lambda_weights, seed=seed + k)

        oriented_arcs, _ = orient_tree_optimally(tree, C)
        imbalance = compute_degree_imbalance(oriented_arcs, n)

        try:
            augmentation_arcs = min_cost_balance(C, imbalance, verbose=False)
        except Exception:
            augmentation_arcs = [(j, i) for (i, j) in tree]

        all_arcs = oriented_arcs + augmentation_arcs
        adj = make_eulerian_graph(all_arcs, n)
        path = euler_tour(adj, start=0)
        tour = shortcut_to_tour(path)

        cost = sum(C[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
        sample_times.append(time.perf_counter() - ts)

        if cost < best_cost:
            best_cost = cost
            best_tour = tour

        if (k + 1) % max(1, samples // 5) == 0 or k == 0:
            print(
                f"    Sampled {k+1}/{samples} trees  "
                f"(best so far: {best_cost:.2f},  "
                f"this sample: {sample_times[-1]*1000:.0f}ms)"
            , flush=True)

    sampling_time = sum(sample_times)
    avg_sample_ms = (sampling_time / samples * 1000) if samples else 0

    performance_ratio = best_cost / lp_bound if lp_bound > 0 else float("inf")
    gap_pct = ((best_cost - lp_bound) / lp_bound * 100) if lp_bound > 0 else float("inf")

    arr = _earliest_arrival(best_tour, C, service, r_times, departure_sec)
    tw_status = _compute_tw_status(best_tour, arr, d_times)

    total_time = time.perf_counter() - wall_start

    print(f"\n{'='*70}", flush=True)
    print(f"  RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  LP lower bound (OPT_HK):         {lp_bound:>12.2f}")
    print(f"  Tour cost:                        {best_cost:>12.2f}")
    print(f"  Performance w.r.t. Held-Karp:                     {performance_ratio:>12.4f}x")
    print(f"  gap w.r.t. Held-Karp:                   {gap_pct:>11.1f}%")
    if n > 2:
        theoretical = math.log(n) / math.log(math.log(n))
        print(f"  O(log n / log log n) bound:       {theoretical:>12.2f}x")
    print(f"{'='*70}", flush=True)
    print(f"  Runtimes", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  [1] LP (Held-Karp):               {lp_time:>10.2f}s")
    print(f"  [2] Symmetrization:               {sym_time*1000:>10.1f}ms")
    print(f"  [3] Multiplicative weights:       {gamma_time:>10.2f}s")
    print(f"  [4] Marginal verification:        {verify_time:>10.2f}s")
    print(f"  [5] Tree sampling (# {samples:2d}):         {sampling_time:>10.2f}s"
          f"  (avg {avg_sample_ms:.0f}ms/tree)")
    print(f"  {'─'*40}")
    print(f"  Total wall time:                  {total_time:>10.2f}s")
    print(f"{'='*70}\n")

    return {
        "status": "MAX_ENTROPY",
        "algo": "max_entropy",
        "objective": float(best_cost),
        "lp_bound": float(lp_bound),
        "performance_ratio": float(performance_ratio),
        "optimality_gap_percent": float(gap_pct),
        "seq_idx": best_tour,
        "arrival_times": arr,
        "tw_status": tw_status,
        "time_lp_sec": lp_time,
        "time_symmetrise_sec": sym_time,
        "time_gamma_sec": gamma_time,
        "time_verify_sec": verify_time,
        "time_sampling_sec": sampling_time,
        "time_total_sec": total_time,
        "samples": samples,
        "marginals_ok": marginals_ok,
    }
