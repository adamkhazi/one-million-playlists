import sys
from collections import *
from ortools.constraint_solver import pywrapcp

import sys
import os
import json
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
from datetime import datetime
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath('../'))
from data import Data

d = Data()
tracks = d.getTrackFeatures()

tracks = tracks[:100]

# Create the solver.
solver = pywrapcp.Solver("Problem")

#
# data
#
N = 100

# the items for each bid
items = [
    [0, 1],  # A,B
    [0, 2],  # A, C
    [1, 3],  # B,D
    [1, 2, 3],  # B,C,D
    [0]  # A
]
# collect the bids for each item
items_t = defaultdict(list)

# [items_t.setdefault(j,[]).append(i) for i in range(N) for j in items[i] ]
# nicer:
[items_t[j].append(i) for i in range(N) for j in items[i]]

bid_amount = [10, 20, 30, 40, 14]

#
# declare variables
#
X = [solver.BoolVar("x%i" % i) for i in range(N)]
obj = solver.IntVar(0, 100, "obj")

#
# constraints
#
solver.Add(obj == solver.ScalProd(X, bid_amount))
for item in items_t:
    solver.Add(solver.Sum([X[bid] for bid in items_t[item]]) <= 1)

# objective
objective = solver.Maximize(obj, 1)

#
# solution and search
#
solution = solver.Assignment()
solution.Add(X)
solution.Add(obj)

# db: DecisionBuilder
db = solver.Phase(X, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)

solver.NewSearch(db, [objective])
num_solutions = 0
while solver.NextSolution():
    print("X:", [X[i].Value() for i in range(N)])
    print("obj:", obj.Value())
    print()
num_solutions += 1

solver.EndSearch()

print()
print("num_solutions:", num_solutions)
print("failures:", solver.Failures())
print("branches:", solver.Branches())
print("WallTime:", solver.WallTime())