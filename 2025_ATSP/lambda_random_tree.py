# algorithms/lambda_random_tree.py
import random
import numpy as np
from typing import Dict, Tuple, List
from algorithms.kirchhoff import compute_edge_marginal, contract_edge, build_laplacian

def sample_lambda_random_tree(
    n: int,
    lambda_weights: Dict[Tuple[int, int], float],
    seed: int = 0
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    tree_edges = []
    current_n = n
    current_edges = lambda_weights.copy()

    edges_to_process = list(lambda_weights.keys())
    rng.shuffle(edges_to_process)

    for edge in edges_to_process:
        if len(tree_edges) == n - 1:
            break

        if edge not in current_edges:
            continue

        p_e = compute_edge_marginal(current_n, current_edges, edge)

        p_e = max(0.0, min(1.0, p_e))

        if rng.random() < p_e:
            tree_edges.append(edge)
            current_edges = contract_edge(current_n, current_edges, edge)
            current_n -= 1
        else:
            del current_edges[edge]

    if len(tree_edges) < n - 1:
        tree_edges = complete_tree_greedy(n, lambda_weights, tree_edges)
    return tree_edges

def sample_lambda_random_tree_robust(
    n: int,
    lambda_weights: Dict[Tuple[int, int], float],
    seed: int = 0,
    max_attempts: int = 10
) -> List[Tuple[int, int]]:
    for attempt in range(max_attempts):
        try:
            tree = sample_lambda_random_tree(n, lambda_weights, seed=seed + attempt)

            if len(tree) == n - 1 and is_connected(n, tree):
                return tree

        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            continue
    return sample_weighted_kruskal(n, lambda_weights, seed)

def sample_weighted_kruskal(
    n: int,
    lambda_weights: Dict[Tuple[int, int], float],
    seed: int = 0
) -> List[Tuple[int, int]]:
    import math
    rng = random.Random(seed)
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
    for (i, j), weight in lambda_weights.items():
        u = rng.random()
        gumbel = -math.log(-math.log(u + 1e-10) + 1e-10)

        log_weight = math.log(weight + 1e-10)
        priority = -(log_weight + gumbel)

        edges.append((priority, i, j))

    edges.sort()
    tree = []

    for _, i, j in edges:
        if union(i, j):
            tree.append((i, j))
            if len(tree) == n - 1:
                break
    return tree

def complete_tree_greedy(
    n: int,
    all_edges: Dict[Tuple[int, int], float],
    partial_tree: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
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

    tree_edges = list(partial_tree)
    for (i, j) in tree_edges:
        union(i, j)

    remaining_edges = [(w, i, j) for (i, j), w in all_edges.items()
                       if (i, j) not in tree_edges]
    remaining_edges.sort(reverse=True)

    for _, i, j in remaining_edges:
        if len(tree_edges) == n - 1:
            break
        if union(i, j):
            tree_edges.append((i, j))
    return tree_edges

def is_connected(n: int, edges: List[Tuple[int, int]]) -> bool:
    if len(edges) != n - 1:
        return False

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j) in edges:
        ra, rb = find(i), find(j)
        if ra != rb:
            parent[rb] = ra

    root = find(0)
    return all(find(i) == root for i in range(n))

def sample_lambda_random_tree_wilson(
    n: int,
    lambda_weights: Dict[Tuple[int, int], float],
    seed: int = 0
) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    adj = {i: [] for i in range(n)}
    for (i, j), weight in lambda_weights.items():
        adj[i].append((j, weight))
        adj[j].append((i, weight))

    for i in range(n):
        total = sum(w for _, w in adj[i])
        if total > 0:
            adj[i] = [(j, w/total) for j, w in adj[i]]

    in_tree = {0}
    tree_edges = []

    for start in range(1, n):
        if start in in_tree:
            continue

        path = [start]
        current = start

        while current not in in_tree:
            if not adj[current]:
                break
            neighbors, weights = zip(*adj[current])
            current = rng.choices(neighbors, weights=weights)[0]

            if current in path:
                idx = path.index(current)
                path = path[:idx+1]
            else:
                path.append(current)

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            tree_edges.append((min(u,v), max(u,v)))
            in_tree.add(path[i])
        in_tree.add(path[-1])

    return tree_edges
