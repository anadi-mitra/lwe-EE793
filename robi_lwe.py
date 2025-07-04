import os
import numpy as np
import pyomo.environ as pyo
import time

# Set Gurobi path
os.environ['GUROBI_HOME'] = '/home/anadi/wd/lwe/gurobi1202/linux64'
os.environ['PATH'] += os.pathsep + '/home/anadi/wd/lwe/gurobi1202/linux64/bin'

# Generate error vector
def generate_error_vector(q, size, sigma_scale=0.25):
    bounds = (-q/4, q/4)
    mu = 0
    sigma = q/4 * sigma_scale
    e = np.empty(size)
    remaining = size
    while remaining > 0:
        samples = np.random.normal(mu, sigma, size=remaining)
        valid = samples[(samples >= bounds[0]) & (samples <= bounds[1])]
        num_valid = len(valid)
        if num_valid > 0:
            e[size - remaining : size - remaining + num_valid] = valid
            remaining -= num_valid
    return e

def generate_lwe_samples(m, n, q=10):
    A = np.random.randint(0, q, size=(m, n))
    s = np.random.randint(0, 2, size=(n, 1))
    As = np.matmul(A, s)
    e = generate_error_vector(q, m)
    e = np.expand_dims(e, axis=1)
    As_e = (As + e) % q
    b_ = As_e % q
    return A, s, e, As_e, b_

# LWE parameters
m, n, q = 20, 20, 29
A, s_true, e_true, As_e, b_prime = generate_lwe_samples(m, n, q)

# Pyomo model
model = pyo.ConcreteModel()
model.I = pyo.Set(initialize=range(m))
model.J = pyo.Set(initialize=range(n))
model.s = pyo.Var(model.J, domain=pyo.Binary)
model.e = pyo.Var(model.I, bounds=(-q/4, q/4))
model.u = pyo.Var(model.I, domain=pyo.NonNegativeReals)
model.v = pyo.Var(model.I, domain=pyo.NonNegativeReals)
model.h = pyo.Var(model.I, within=pyo.Integers)

def absolute_value_constraint(model, i):
    return (
            sum(A[i][j] * model.s[j] for j in model.J)
            + (model.u[i] - model.v[i])
            == b_prime[i] + model.h[i]*q
    )
model.abs_constraints = pyo.Constraint(model.I, rule=absolute_value_constraint)

def objective_rule(model):
    return sum(model.u[i] + model.v[i] for i in model.I)
model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Use Gurobi
solver = pyo.SolverFactory('gurobi')
solver.options['Threads'] = 12  # Use multi-threading (adjust core count)

start_time = time.time()
results_solver = solver.solve(model, tee=True)  # tee=True to show Gurobi output
solve_time = time.time() - start_time

# Output results
print(f"\nOptimal objective value: {pyo.value(model.obj):.2f}")
print("s =", ' '.join(str(int(pyo.value(model.s[j]))) for j in model.J))
e_vec = [round(pyo.value(model.u[i]) - pyo.value(model.v[i]), 2) for i in model.I]
print("e =", e_vec)
print("Time Taken: ", solve_time)
