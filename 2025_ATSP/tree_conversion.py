# algorithms/tree_conversion.py
from typing import List, Tuple

INF = 10**9

def orient_tree_optimally(
    tree: List[Tuple[int, int]],
    C: List[List[float]],
) -> Tuple[List[Tuple[int, int]], float]:
    arcs = []
    total_cost = 0.0
    for (i, j) in tree:
        if C[i][j] <= C[j][i]:
            arcs.append((i, j))
            total_cost += C[i][j]
        else:
            arcs.append((j, i))
            total_cost += C[j][i]
    return arcs, total_cost

def compute_degree_imbalance(
    oriented_arcs: List[Tuple[int, int]],
    n: int,
) -> List[int]:
    imbalance = [0] * n
    for (i, j) in oriented_arcs:
        imbalance[i] += 1  # extra out
        imbalance[j] -= 1  # extra in
    return imbalance

def make_eulerian_graph(
    arcs: List[Tuple[int, int]],
    n: int,
) -> List[List[int]]:
    adj = [[] for _ in range(n)]
    for (i, j) in arcs:
        adj[i].append(j)
    return adj

def euler_tour(adj: List[List[int]], start: int = 0) -> List[int]:
    local_adj = [list(nbrs) for nbrs in adj]
    stack = [start]
    path = []
    while stack:
        v = stack[-1]
        if local_adj[v]:
            u = local_adj[v].pop()
            stack.append(u)
        else:
            path.append(stack.pop())
    return path[::-1]

def shortcut_to_tour(path: List[int]) -> List[int]:
    seen = set()
    tour = []
    for v in path:
        if v not in seen:
            tour.append(v)
            seen.add(v)
    if tour and tour[-1] != tour[0]:
        tour.append(tour[0])
    return tour
