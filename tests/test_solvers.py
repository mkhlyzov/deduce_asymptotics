import numpy as np
import pytest

from deduce_asymptotics.solvers import SOLVERS_ALL, SOLVERS_EXTRA
from deduce_asymptotics.solvers import (
    Solver, Constant, Log, Linear, LinearLog, Quadratic, QuadraticLog, Cubic,
    CubicLog, Exponential, Polynomial, PolynomialLog
)


PARAMS_MIN_VALUE = 1e-7
PARAMS_MAX_VALUE = 10


def lognuniform(low=0, high=1, size=None, base=np.e):
    return np.power(base, np.random.uniform(low, high, size))

def get_random_param(min=PARAMS_MIN_VALUE, max=PARAMS_MAX_VALUE):
    # return np.random.uniform(min, max)
    return lognuniform(low=np.log10(min), high=np.log10(max), base=10)

def randomize_values(y, denom=100):
    return y
    # return y + np.random.normal(
    #     0, (y - np.min(y) + 1e-7) / (np.max(y) - np.min(y) + 1e-7) / denom)
    # return y * np.random.normal(
    #     1, (y - np.min(y) + 1e-7) / (np.max(y) - np.min(y) + 1e-7) / denom)

solvers_and_functions = [
    (Constant,      lambda x, a   : np.ones_like(x) * a),
    (Log,           lambda x, a, b: a * np.log(x) + b),
    (Linear,        lambda x, a, b: a * x + b),
    (LinearLog,     lambda x, a, b: a * x * np.log(x) + b),
    (Quadratic,     lambda x, a, b: a * x**2 + b),
    (QuadraticLog,  lambda x, a, b: a * x**2 * np.log(x) + b),
    (Cubic,         lambda x, a, b: a * x**3 + b),
    (CubicLog,      lambda x, a, b: a * x**3 * np.log(x) + b),
    (Exponential,   lambda x, a, b, c: a * np.exp(b * x) + c),
    # (Polynomial,    lambda x, a, b, c: a * x**b + c),
    # (PolynomialLog, lambda x, a, b, c, d: a * x**b * np.log(x)**c + d),
]

@pytest.mark.parametrize('num_points', [50, 100, 1000])
@pytest.mark.parametrize('x_lb', [1])
@pytest.mark.parametrize('x_ub', [2, 4, 6])
@pytest.mark.parametrize('solver_class, func', solvers_and_functions)
def test_solvers(num_points, x_lb, x_ub, solver_class, func, num_tries=10):
    mistakes_total = 0
    num_params = len(solver_class().params)
    for i in range(num_tries):
        x = np.logspace(x_lb, x_ub, num=num_points, dtype=np.float64)
        params = [get_random_param() for _ in range(num_params)]
        y = func(x, *params)
        y = randomize_values(y)

        solver = solver_class()
        solver.fit(x, y)
        
        mistakes_current = np.isclose(params, solver.params, rtol=0.05).sum()
        if mistakes_current > 0:
            if not np.isclose(y, solver(x)).all():
                mistakes_total += 1
        assert mistakes_total <= num_tries / 10