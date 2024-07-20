import time

import numpy as np
import pytest

import deduce_asymptotics
from deduce_asymptotics import deduce
from deduce_asymptotics.solvers import SOLVERS_ALL, SOLVERS_EXTRA
from deduce_asymptotics.solvers import (
    Solver, Constant, Log, Linear, LinearLog, Quadratic, QuadraticLog, Cubic,
    CubicLog, Exponential, Polynomial, PolynomialLog
)


def get_random_array(n):
    return np.random.standard_normal(n)

def get_random_sorted_array(n):
    x = get_random_array(n)
    return np.sort(x)

def constant_function(x):
    for i in range(1000):
        i**2

def log_function(x):
    i = 1
    a = 0
    while i < len(x):
        a += x[i] ** 2
        i *= 2

def quadratic_function(x):
    for i in range(len(x)):
        for j in range(i, len(x)):
            k = x[i] * x[j]

def quadratic_log_function(x):
    for i in range(len(x)):
        sorted(x)

def cubic_function(x):
    for i in range(0, len(x), 1):
        for j in range(0, len(x), 3):
            for k in range(0, len(x), 5):
                m = x[i] * x[j] * x[k]


def exponential_function(x):
    n = len(x)
    for i in range(int(1.1**n)):
        time.sleep(1e-7)


solvers_and_functions = [
    (Constant,      constant_function),
    (Log,           log_function),
    (Linear,        np.sum),
    (Linear,        np.argmax),
    (LinearLog,     np.sort),
    (Quadratic,     quadratic_function),
    (QuadraticLog,  quadratic_log_function),
    (Cubic,         cubic_function),
    # (Exponential,   exponential_function),
]

@pytest.mark.parametrize('solver_class, func', solvers_and_functions)
def test_default_deduce(solver_class, func, num_tries=5):
    mistakes_total = 0
    for i in range(num_tries):
        solver, ns, ts = deduce_asymptotics.deduce_asymptotics.deduce_helper(
            func, get_random_array, 10, 10)
        if not isinstance(solver, solver_class):
            mistakes_total += 1
        assert mistakes_total <= 1    


def test_default_deduce_exponential(
    solver_class=Exponential, func=exponential_function, num_tries=5
):
    mistakes_total = 0
    for i in range(num_tries):
        solver, ns, ts = deduce_asymptotics.deduce_asymptotics.deduce_helper(
            func, get_random_array, time_budget=10, num_samples=10,
            step=lambda n: n + 2, start=2)
        if not isinstance(solver, solver_class):
            mistakes_total += 1
        assert mistakes_total <= 0
