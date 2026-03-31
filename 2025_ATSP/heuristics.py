# algorithms/heuristics.py
import numpy as np
import time
from typing import List, Tuple, Dict
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

# preprocess arcs wlog and use them first, keep adding arcs so that a Hamiltonian tour is found.
def greedy_algorithm(C: np.ndarray) -> Tuple[List[int], float]:
    n = len(C)
    arcs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                arcs.append((float(C[i, j]), i, j))
    arcs.sort()

    in_deg  = [0] * n
    out_deg = [0] * n
    parent  = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def can_union(a, b):
        return find(a) != find(b)

    def do_union(a, b):
        parent[find(b)] = find(a)

    selected = []

    for cost, i, j in arcs:
        if len(selected) >= n:
            break
        if out_deg[i] >= 1 or in_deg[j] >= 1:
            continue
        if can_union(i, j) or len(selected) == n - 1:
            if not can_union(i, j) and len(selected) < n - 1:
                continue
            selected.append((i, j))
            out_deg[i] += 1
            in_deg[j]  += 1
            if can_union(i, j):
                do_union(i, j)

    if len(selected) < n:
        return nearest_neighbor(C, start=0)

    tour      = _arcs_to_tour(selected, n)
    tour_cost = sum(C[tour[k]][tour[k + 1]] for k in range(len(tour) - 1))
    return tour, tour_cost

# at each node, select the successor node in the sequence as the closest one
def nearest_neighbor(C: np.ndarray, start: int = 0) -> Tuple[List[int], float]:
    n         = len(C)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    cur  = start

    while unvisited:
        nxt = min(unvisited, key=lambda j: C[cur, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    tour.append(start)
    cost = sum(C[tour[k]][tour[k + 1]] for k in range(len(tour) - 1))
    return tour, cost

# start at a single node, keep adding nodes in between such that the tour increase is minimum.
def cheapest_insertion(C: np.ndarray) -> Tuple[List[int], float]:
    n         = len(C)
    tour      = [0, 0]
    remaining = set(range(1, n))

    while remaining:
        best_node, best_pos, best_delta = None, None, float("inf")
        for node in remaining:
            for pos in range(len(tour) - 1):
                a, b  = tour[pos], tour[pos + 1]
                delta = C[a, node] + C[node, b] - C[a, b]
                if delta < best_delta:
                    best_delta = delta
                    best_node  = node
                    best_pos   = pos + 1
        tour.insert(best_pos, best_node)
        remaining.remove(best_node)

    cost = sum(C[tour[k]][tour[k + 1]] for k in range(len(tour) - 1))
    return tour, cost


def assignment_lower_bound(C: np.ndarray) -> float:
    """
    Compute the assignment relaxation lower bound for TSP.

    Solves the linear assignment problem on the cost matrix with the diagonal set to infinity.
    Every TSP tour is a valid assignment, so the optimal assignment cost is a lower bound on the optimal TSP cost.

    Returns the assignment bound (sum of costs of matched arcs).
    """
    n       = len(C)
    C_work  = C.astype(float).copy()
    np.fill_diagonal(C_work, np.inf)

    # linear_sum_assignment cannot handle inf; replace with a large finite value
    big     = float(np.nanmax(C_work[C_work < np.inf])) * n * 10 + 1
    C_work  = np.where(np.isinf(C_work), big, C_work)

    row_ind, col_ind = linear_sum_assignment(C_work)
    bound = float(C_work[row_ind, col_ind].sum())

    # If the solution used any "big" arcs the bound is meaningless
    if bound >= big:
        return 0.0
    return bound


def repeated_assignment(C: np.ndarray) -> Tuple[List[int], float, float]:
    n = len(C)

    assign_bound = assignment_lower_bound(C)

    node_groups: List[List[int]] = [[i] for i in range(n)]
    D       = C.copy().astype(float)
    cur_n   = n
    selected_arcs: List[Tuple[int, int]] = []
    max_rounds = int(np.ceil(np.log2(n))) + 1

    for _round in range(max_rounds):
        if cur_n <= 1:
            break

        D_assign = D.copy()
        np.fill_diagonal(D_assign, np.inf)
        row_ind, col_ind = linear_sum_assignment(D_assign)
        assignment = {r: c for r, c in zip(row_ind, col_ind)}

        visited = [False] * cur_n
        cycles: List[List[int]] = []

        for start in range(cur_n):
            if visited[start]:
                continue
            cycle = []
            cur   = start
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = assignment[cur]
            cycles.append(cycle)

        for cycle in cycles:
            for k in range(len(cycle)):
                ci     = cycle[k]
                cj     = cycle[(k + 1) % len(cycle)]
                orig_i = node_groups[ci][0]
                orig_j = node_groups[cj][0]
                selected_arcs.append((orig_i, orig_j))

        if len(cycles) == 1:
            break

        new_groups: List[List[int]] = []
        for cycle in cycles:
            merged: List[int] = []
            for ci in cycle:
                merged.extend(node_groups[ci])
            new_groups.append(merged)

        node_groups = new_groups
        cur_n       = len(node_groups)

        D_new = np.full((cur_n, cur_n), np.inf)
        for ni in range(cur_n):
            for nj in range(cur_n):
                if ni == nj:
                    continue
                best = np.inf
                for orig_i in node_groups[ni]:
                    for orig_j in node_groups[nj]:
                        if C[orig_i, orig_j] < best:
                            best = C[orig_i, orig_j]
                D_new[ni, nj] = best
        D = D_new

    tour      = _tour_procedure(n, selected_arcs, C)
    tour_cost = sum(C[tour[k]][tour[k + 1]] for k in range(len(tour) - 1))
    return tour, tour_cost, assign_bound

def _tour_procedure(
    n: int,
    arcs: List[Tuple[int, int]],
    C: np.ndarray,
) -> List[int]:
    if not arcs:
        return list(range(n)) + [0]

    adj: Dict[int, List[int]] = defaultdict(list)
    for (i, j) in arcs:
        adj[i].append(j)

    changed = True
    while changed:
        changed = False
        for v in range(n):
            if len(adj[v]) > 1:
                w2 = adj[v].pop(1)
                u2 = None
                for src, dsts in adj.items():
                    if v in dsts and src != v:
                        if dsts.count(v) > 0:
                            u2 = src
                            break
                if u2 is not None:
                    adj[u2].remove(v)
                    adj[u2].append(w2)
                else:
                    adj[v].insert(1, w2)
                    break
                changed = True
                break

    next_node = {v: adj[v][0] for v in range(n) if adj[v]}
    if len(next_node) < n:
        return nearest_neighbor(C, start=0)[0]

    tour  = [0]
    cur   = next_node.get(0)
    steps = 0
    while cur != 0 and cur is not None and steps < n:
        tour.append(cur)
        cur = next_node.get(cur)
        steps += 1
    tour.append(0)

    if len(tour) != n + 1:
        return nearest_neighbor(C, start=0)[0]
    return tour


def _arcs_to_tour(arcs: List[Tuple[int, int]], n: int) -> List[int]:
    if not arcs:
        return list(range(n)) + [0]

    nxt   = {}
    for (i, j) in arcs:
        nxt[i] = j

    start = arcs[0][0]
    tour  = [start]
    cur   = nxt.get(start)
    while cur != start and cur is not None:
        tour.append(cur)
        cur = nxt.get(cur)
    tour.append(start)
    return tour

def solve_atsp_frieze(
    C: np.ndarray,
    algorithm: str = "nearest_neighbor",
    **kwargs,
) -> Dict:
    start_time = time.time()

    if algorithm == "greedy":
        tour, cost = greedy_algorithm(C)
        status = "GREEDY"
        result = {"status": status, "algo": algorithm,
                  "objective": float(cost), "seq_idx": tour}

    elif algorithm == "nearest_neighbor":
        tour, cost = nearest_neighbor(C, start=kwargs.get("start", 0))
        status = "NEAREST_NEIGHBOR"
        result = {"status": status, "algo": algorithm,
                  "objective": float(cost), "seq_idx": tour}

    elif algorithm == "cheapest_insertion":
        tour, cost = cheapest_insertion(C)
        status = "CHEAPEST_INSERTION"
        result = {"status": status, "algo": algorithm,
                  "objective": float(cost), "seq_idx": tour}

    elif algorithm == "repeated_assignment":
        tour, cost, assign_bound = repeated_assignment(C)
        status = "REPEATED_ASSIGNMENT"
        perf_ratio = (cost / assign_bound) if assign_bound > 0 else float("inf")
        gap_pct    = ((cost - assign_bound) / assign_bound * 100) if assign_bound > 0 else float("inf")
        result = {
            "status":                  status,
            "algo":                    algorithm,
            "objective":               float(cost),
            "seq_idx":                 tour,
            "lp_bound":                float(assign_bound),
            "assignment_bound":        float(assign_bound),
            "performance_ratio":       float(perf_ratio),
            "optimality_gap_percent":  float(gap_pct),
        }

    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")

    result["n_nodes"]          = len(C)
    result["computation_time"] = time.time() - start_time
    return result
