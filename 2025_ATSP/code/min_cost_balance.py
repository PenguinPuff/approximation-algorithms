#algorithms/min_cost_balance.py
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple

def min_cost_balance(
    C: List[List[float]],
    imbalance: List[int],
    time_limit: float = 10.0,
    verbose: bool = False
) -> List[Tuple[int, int]]:
    n = len(imbalance)
    model = gp.Model("min_cost_circulation")

    if not verbose:
        model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit

    f = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            f[i, j] = model.addVar(
                lb=0,
                vtype=GRB.INTEGER,
                obj=C[i][j],
                name=f"f_{i}_{j}"
            )

    for i in range(n):
        model.addConstr(
            gp.quicksum(f[i, j] for j in range(n) if j != i) -
            gp.quicksum(f[j, i] for j in range(n) if j != i)
            == -imbalance[i],
            name=f"balance_{i}"
        )

    model.ModelSense = GRB.MINIMIZE
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError("Min-cost circulation failed")

    arcs = []
    for (i, j), var in f.items():
        k = int(round(var.X))
        if k > 0:
            arcs.extend([(i, j)] * k)
    return arcs
