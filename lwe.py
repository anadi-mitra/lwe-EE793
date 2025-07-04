# !pip install -q pyomo
# !apt install -y -q coinor-cbc


import numpy as np
import pyomo.environ as pyo
import time

def generate_error_vector(q, size, sigma_scale=0.25):
    """
    Generates error vector with elements from truncated Gaussian distribution.

    Args:
        q (int): Prime modulus defining range [-q/4, q/4]
        size (int): Number of elements in vector
        sigma_scale (float): Sigma as fraction of q (default: q/4/1.0 = q/4)
                             Adjust to control distribution spread (0.25 = q/4)

    Returns:
        np.ndarray: Error vector of shape (size,)
    """
    bounds = (-q/4, q/4)
    mu = 0  # Centered at zero
    sigma = q/4 * sigma_scale  # Default: sigma = q/16

    # Preallocate array
    e = np.empty(size)
    remaining = size

    while remaining > 0:
        # Generate batch of candidates
        samples = np.random.normal(mu, sigma, size=remaining)

        # Filter valid samples within bounds
        valid = samples[(samples >= bounds[0]) & (samples <= bounds[1])]
        num_valid = len(valid)

        if num_valid > 0:
            # Fill valid samples into result array
            e[size - remaining : size - remaining + num_valid] = valid
            remaining -= num_valid

    return e

def generate_lwe_samples(m, n, q=10):
    """
    Generates random LWE samples (A, s) where:
    - A: m×n matrix of random integers (0 to max_int)
    - s: n×1 binary vector (0 or 1 elements)

    Args:
        m (int): Number of equations/rows
        n (int): Secret dimension/columns
        max_int (int): Maximum value for matrix elements (default: 10)

    Returns:
        tuple: (A matrix, s vector) as NumPy arrays
    """
    # Generate random integer matrix
    A = np.random.randint(0, q, size=(m, n))

    # Generate binary secret vector
    s = np.random.randint(0, 2, size=(n, 1))

    # b = A * s
    As = np.matmul(A, s)
    # print(As)
    e = generate_error_vector(q, m)
    # print(e.reshape(1,-1))
    # #print(e.T.shape)
    # print(e.transpose())
    e = np.expand_dims(e, axis = 1)
    # print(e)
    As_e = (As + e) % q
    #As_e = As
    # print(As_e)
    # b' = b mod q
    b_ = As_e % q
    return A, s, e, As_e, b_

m, n, q = 20, 20, 29
A, s_true, e_true, As_e, b_prime = generate_lwe_samples(m, n, q)


# Create Pyomo model
model = pyo.ConcreteModel()

# Index sets
model.I = pyo.Set(initialize=range(m))  # Row indices
model.J = pyo.Set(initialize=range(n))  # Column indices

# Decision variables
model.s = pyo.Var(model.J, domain=pyo.Binary)  # Binary secret vector
model.e = pyo.Var(model.I, bounds=(-q/4, q/4)) # Error vector (bounded)

# Auxiliary variables for linearizing absolute value
model.u = pyo.Var(model.I, domain=pyo.NonNegativeReals)  # Positive part
model.v = pyo.Var(model.I, domain=pyo.NonNegativeReals)  # Negative part
model.h = pyo.Var(model.I, within=pyo.Integers)

# Constraints
def absolute_value_constraint(model, i):
    return (
        sum(A[i][j] * model.s[j] for j in model.J)
        + (model.u[i] - model.v[i])
        == b_prime[i] + model.h[i]*q
    )
model.abs_constraints = pyo.Constraint(model.I, rule=absolute_value_constraint)

def as_e_constraint(model, i):
    return sum(A[i][j] * model.s[j] for j in model.J) + model.e[i] == As_e[i]
# model.as_e_constr = pyo.Constraint(model.I, rule=as_e_constraint)

# model.abs_constraints.add(sum(A[i][j] * model.s[j] for j in model.J) + model.e[i] == As_e[i][j])

# Objective: Minimize sum of absolute differences
def objective_rule(model):
    return sum(model.u[i] + model.v[i] for i in model.I)
model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Solve with CBC solver
solver = pyo.SolverFactory('cbc')
start_time = time.time()
results_solver = solver.solve(model, tee=False)
solve_time = time.time() - start_time

# Output results
print(f"Optimal objective value: {pyo.value(model.obj):.2f}\n")

print("Secret vector (s):")
# print(model.s)
for j in model.J:
    print(f"s[{j}] = {int(pyo.value(model.s[j]))}")

print("\nError vector (e):")
for i in model.I:
    print(f"e[{i}] = {pyo.value(model.u[i] - model.v[i]):.2f}")

print("Time Taken: ", solve_time)
