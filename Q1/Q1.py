#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Logistic Map Function
def logistic_map(x, r):
    return r * x * (1 - x)

# (a) Fixed Points and Stability
def fixed_points(r):
    roots = fsolve(lambda x: x - r * x * (1 - x), [0, 1])
    stability = [abs(r * (1 - 2 * x)) for x in roots]
    return roots, stability

r_values = [1, 2, 3, 4]
for r in r_values:
    fp, st = fixed_points(r)
    print(f"r = {r}, Fixed Points: {fp}, Stability: {st}")

# (b) Dynamic Programming - Iterating the Logistic Map
def iterate_logistic_map(r, x0, n_iter=100):
    x_vals = np.zeros(n_iter)
    x_vals[0] = x0
    for i in range(1, n_iter):
        x_vals[i] = logistic_map(x_vals[i - 1], r)
    return x_vals

r_values = [2, 3, 3.5, 3.8, 4.0]
x0 = 0.2
for r in r_values:
    x_series = iterate_logistic_map(r, x0)
    plt.plot(x_series, label=f"r={r}")
plt.xlabel("Iterations")
plt.ylabel("x_n")
plt.title("Logistic Map Iterations for Different r Values")
plt.legend()
plt.show()

# (c) Different Initial Conditions
initial_conditions = [0.1, 0.3, 0.5]
r = 3.5
for x0 in initial_conditions:
    x_series = iterate_logistic_map(r, x0)
    plt.plot(x_series, label=f"x0={x0}")
plt.xlabel("Iterations")
plt.ylabel("x_n")
plt.title(f"Logistic Map for r={r} with Different Initial Conditions")
plt.legend()
plt.show()

# (d) Bifurcation Diagram
r_values = np.linspace(2.4, 4.0, 600)
n_iter = 1000
last = 200
bifurcation_x = []
bifurcation_r = []

for r in r_values:
    x = 0.2
    for _ in range(n_iter):
        x = logistic_map(x, r)
    for _ in range(last):
        x = logistic_map(x, r)
        bifurcation_r.append(r)
        bifurcation_x.append(x)

plt.scatter(bifurcation_r, bifurcation_x, s=0.1, color='black')
plt.xlabel("r (Control Parameter)")
plt.ylabel("x_n (Population)")
plt.title("Bifurcation Diagram of the Logistic Map")
plt.show()

# (e) Scaling in Bifurcation
def modified_logistic_map(x, r, gamma):
    return r * x * (1 - x**gamma)

gamma_values = np.linspace(0.5, 1.5, 100)
bifurcation_points = []

for gamma in gamma_values:
    r = 3.0
    x = 0.2
    for _ in range(n_iter):
        x = modified_logistic_map(x, r, gamma)
    bifurcation_points.append(x)

plt.plot(gamma_values, bifurcation_points, marker='o')
plt.xlabel("Gamma")
plt.ylabel("First Bifurcation Point")
plt.title("First Bifurcation Point vs. Gamma")
plt.show()
