# algorithms/lp_guided_mst.py
# LP-Guided MST heuristic (in the report, a simple heuristic):
#   1. solve Held-Karp LP -> x*
#   2. extract symmetrized edge weights from x*
#   3. sample Z > (2 log n) spanning trees via lp-bias Kruskal
#   4. for each tree: orient, balance degrees, Euler tour, shortcut
#   5. return best tour
import random
import time
from typing import Dict, List, Tuple

def _extract_lp_edge_weights(
    x_star: Dict[Tuple[int, int], float], n: int
) -> Dict[Tuple[int, int], float]:
    edge_weights: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            w = x_star.get((i, j), 0.0) + x_star.get((j, i), 0.0)
            if w > 1e-9:
                edge_weights[(i, j)] = w
    return edge_weights

def _lp_biased_kruskal(
    n: int,
    C: List[List[float]],
    lp_edge_weights: Dict[Tuple[int, int], float],
    seed: int = 0,
    bias_strength: float = 0.5,
) -> List[Tuple[int, int]]:
    """
    Kruskal's MST with costs biased toward LP-supported edges:
        perturbed_cost(i,j) = cost(i,j) * (1 - bias_strength * lp_weight(i,j))
    Higher bias_strength -> stronger preference for high-LP-weight edges.
    """
    rng = random.Random(seed)
    INF = 10**9

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            return True
        return False

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            cost = min(C[i][j], C[j][i])
            if cost >= INF:
                continue
            lp_w = lp_edge_weights.get((i, j), 0.0)
            bias = 1.0 - bias_strength * min(lp_w, 1.0)
            perturbed = cost * bias * (1.0 + rng.uniform(-1e-6, 1e-6))
            edges.append((perturbed, i, j))

    edges.sort()
    tree = []
    for _, i, j in edges:
        if union(i, j):
            tree.append((i, j))
            if len(tree) == n - 1:
                break

    return tree

def solve_with_lp_guided_thin_tree(
    nodes: List[str],
    C: List[List[float]],
    service: List[float],
    r_times: List[float],
    d_times: List[float],
    departure_sec: float,
    route_id: str,
    samples: int = 20,
    seed: int = 0,
    bias_strength: float = 0.5,
) -> Dict:
    from lp.held_karp import solve_held_karp
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

    print(f"\n{'='*70}")
    print(f"  LP-Guided MST (a simple heuristic)")
    print(f"  Route: {route_id}  |  Nodes: {n}  |  Samples: {samples}")
    print(f"  bias_strength={bias_strength}")
    print(f"{'='*70}")

    wall_start = time.perf_counter()

    # Step 1: Held-Karp LP
    t0 = time.perf_counter()
    x_star, lp_bound = solve_held_karp(C, verbose=False)
    lp_time = time.perf_counter() - t0
    print(f"    LP lower bound = {lp_bound:.2f}  [{lp_time:.1f}s]")

    # Step 2: LP edge weights
    lp_edge_weights = _extract_lp_edge_weights(x_star, n)

    # Step 3: Sample trees
    best_tour = None
    best_cost = INF
    sample_times = []

    for k in range(samples):
        ts = time.perf_counter()
        
        tree = _lp_biased_kruskal(
            n=n, C=C,
            lp_edge_weights=lp_edge_weights,
            seed=seed + k,
            bias_strength=bias_strength,
        )

        oriented_arcs, _ = orient_tree_optimally(tree, C)
        imbalance = compute_degree_imbalance(oriented_arcs, n)

        try:
            aug = min_cost_balance(C, imbalance, verbose=False)
        except Exception:
            aug = [(j, i) for (i, j) in tree]

        all_arcs = oriented_arcs + aug
        adj = make_eulerian_graph(all_arcs, n)
        path = euler_tour(adj, start=0)
        tour = shortcut_to_tour(path)

        cost = sum(C[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
        sample_times.append(time.perf_counter() - ts)
        
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    sampling_time = sum(sample_times)
    performance_ratio = best_cost / lp_bound if lp_bound > 0 else float("inf")
    gap_pct = ((best_cost - lp_bound) / lp_bound * 100) if lp_bound > 0 else float("inf")
    
    total_time = time.perf_counter() - wall_start

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  LP lower bound:                   {lp_bound:>12.2f}")
    print(f"  Best tour cost:                   {best_cost:>12.2f}")
    print(f"  Performance w.r.t. Held-Karp:     {performance_ratio:>12.4f}x")
    print(f"  Gap w.r.t. Held-Karp:             {gap_pct:>11.1f}%")
    print(f"{'='*70}")
    print(f"  Runtimes")
    print(f"{'='*70}")
    print(f"  LP (Held-Karp):                   {lp_time:>10.2f}s")
    print(f"  Tree sampling ({samples:2d}):              {sampling_time:>10.2f}s")
    print(f"  {'─'*40}")
    print(f"  Total wall time:                  {total_time:>10.2f}s")
    print(f"{'='*70}\n")

    # Compute arrival times and TW status
    def earliest_arrival(path, C, service, r_times, start_at=0.0):
        if not path:
            return []
        t = max(r_times[path[0]], start_at)
        arr = [t]
        for k in range(1, len(path)):
            i, j = path[k - 1], path[k]
            t = max(r_times[j], t + service[i] + C[i][j])
            arr.append(t)
        return arr

    arr = earliest_arrival(best_tour, C, service, r_times, departure_sec)

    tw_status = []
    for node, t in zip(best_tour, arr):
        if d_times[node] >= INF:
            tw_status.append("no_tw")
        elif t <= d_times[node] + 1e-6:
            tw_status.append("ok")
        else:
            tw_status.append("violate")

    return {
        "status": "HEURISTIC",
        "algo": "lp_guided_thin_tree",
        "objective": float(best_cost),
        "lp_bound": float(lp_bound),
        "performance_ratio": float(performance_ratio),
        "optimality_gap_percent": float(gap_pct),
        "seq_idx": best_tour,
        "arrival_times": arr,
        "tw_status": tw_status,
        "time_lp_sec": lp_time,
        "time_sampling_sec": sampling_time,
        "time_total_sec": total_time,
        "samples": samples,
    }
