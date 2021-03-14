from pulp import *


def payoff(x, y):
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return 0


K = 10  # number of battlefields
A = 100  # number of troops of A
B = 100  # number of troops of B

m = LpProblem("model", sense=LpMinimize)

u = LpVariable("u")
d = LpVariable("l", 0, A)
k = LpVariable("k", 1, K-1)
j = LpVariable("j", 0, A)
i = LpVariable("i", 0, j)

