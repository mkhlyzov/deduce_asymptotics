from collections.abc import Callable
from typing import Iterable
from numbers import Number

import numpy as np
from scipy.optimize import minimize, differential_evolution, shgo, dual_annealing, direct

from .utils import suppress_warnings


REPR_PRECISION = 4  # digits of precision for __repr__


class Solver(object):
    name: str
    params: np.ndarray[float]
    bounds: Iterable[tuple[Number, Number]]

    def __init__(self, params: np.ndarray[float] = None):
        if params is not None:
            self.params = params
        if not hasattr(self, 'bounds'):
            self.bounds = [[0, 10]] * len(self.params)

    def __call__(self, x) -> float:
        raise NotImplementedError
    
    def loss(self, x, y) -> float:
        return self._loss(self.params, x, y)

    def _loss(self, params, x, y) -> float:
        if not hasattr(x, '__len__'):
            x = np.array([x], dtype=float)
            y = np.array([y], dtype=float)
        old_params = self.params
        self.params = params
        y_hat = self(x)

        
        N = 10 * len(x) // len(set(x))
        w = np.arange(len(y) + N, N, -1, dtype=float) ** -1
        # w = np.ones(len(y))
        w /= np.sum(w)

        loss_1 = np.dot(w, np.abs(y - y_hat) / y)
        loss_2 = np.dot(w, (y - y_hat) ** 2)
        loss_3 = np.mean(params ** 2)
        loss = loss_1 * 0.1 + loss_2
        self.params = old_params
        return loss
    
    def _update_bounds(self) -> int:
        hits = 0
        for p, b in zip(self.params, self.bounds):
            if isinstance(b, tuple):
                # Can't update tuples
                continue
            lower, upper = b
            if np.isclose(p, upper):
                upper = upper * 10
                hits += 1
            if lower > 0 and np.isclose(p, lower, atol=lower / 100):
                lower = lower / 10
                hits += 1
        return hits
    
    @suppress_warnings
    def fit(self, x: np.ndarray[float], y: np.ndarray[float]) -> None:
        while True:
            self.fit_genetic(x, y)
            if self._update_bounds() == 0:
                break
        return self
    
    def fit_minimize(self, x: np.ndarray[float], y: np.ndarray[float]) -> None:
        result = minimize(self._loss, self.params, args=(x, y))
        self.params = result.x
        return self
    
    def fit_genetic(self, x: np.ndarray[float], y: np.ndarray[float]) -> None:
        result = differential_evolution(self._loss, self.bounds, args=(x, y))
        self.params = result.x
        return self
    
    def __repr__(self):
        raise NotImplementedError
    

class Constant(Solver):
    name = "O(1)"
    params = np.array([1.]) * 1e-5
    def __call__(self, x):
        return self.params[0]
    def __repr__(self):
        return f"{self.params[0]:.{REPR_PRECISION}g}"


class Log(Solver):
    name = "O(log n)"
    params = np.array([1., 1.]) * 1e-5
    def __call__(self, x):
        return self.params[0] * np.log(x) + self.params[1]
    def __repr__(self):
        return f"{self.params[0]:.{REPR_PRECISION}g} * log(x) + {self.params[1]:.{REPR_PRECISION}g}"


class Linear(Solver):
    name = "O(n)"
    params = np.array([1., 1.]) * 1e-5
    def __call__(self, x):
        return self.params[0] * x + self.params[1]
    def __repr__(self):
        return f"{self.params[0]:.{REPR_PRECISION}g} * x + {self.params[1]:.{REPR_PRECISION}g}"


class LinearLog(Solver):
    name = "O(n log n)"
    params = np.array([1., 1.]) * 1e-5
    def __call__(self, x):
        return self.params[0] * x * np.log(x) + self.params[1]
    def __repr__(self):
        return f"{self.params[0]:.{REPR_PRECISION}g} * x * log(x) + {self.params[1]:.{REPR_PRECISION}g}"


class Quadratic(Solver):
    name = "O(n^2)"
    params = np.array([1., 1.]) * 1e-5
    def __call__(self, x):
        return self.params[0] * x ** 2 + self.params[1]
    def __repr__(self):
        return f"{self.params[0]:.{REPR_PRECISION}g} * x^2 + {self.params[1]:.{REPR_PRECISION}g}"


class QuadraticLog(Solver):
    name = "O(n^2 log n)"
    params = np.array([1., 1.]) * 1e-5
    def __call__(self, x):
        return self.params[0] * (x ** 2) * np.log(x) + self.params[1]
    def __repr__(self):
        return f"{self.params[0]:.{REPR_PRECISION}g} * x^2 * log(x) + {self.params[1]:.{REPR_PRECISION}g}"


class Cubic(Solver):
    name = "O(n^3)"
    params = np.array([1., 1.]) * 1e-5
    def __call__(self, x):
        return self.params[0] * x ** 3 + self.params[1]
    def __repr__(self):
        return f"{self.params[0]:.{REPR_PRECISION}g} * x^3 + {self.params[1]:.{REPR_PRECISION}g}"


class CubicLog(Solver):
    name = "O(n^3 log n)"
    params = np.array([1., 1.]) * 1e-5
    def __call__(self, x):
        return self.params[0] * x ** 3 * np.log(x) + self.params[1]
    def __repr__(self):
        return f"{self.params[0]:.{REPR_PRECISION}g} * x^3 * log(x) + {self.params[1]:.{REPR_PRECISION}g}"


class Exponential(Solver):
    name = "O(e^n)"
    params = np.array([1., 1., 1.]) * 1e-5
    bounds = [[0, 10], [0, 0.1], [0, 10]]
    def __call__(self, x):
        return self.params[0] * np.exp(self.params[1] * x) + self.params[2]
    def __repr__(self):
        return f"{self.params[0]:.{REPR_PRECISION}g} * exp({self.params[1]:.{REPR_PRECISION}g} * x) + {self.params[2]:.{REPR_PRECISION}g}"


class Polynomial(Solver):
    name = "O(n^p)"
    params = np.array([1., 1., 1.]) * 1e-5
    def __call__(self, x):
        return self.params[0] * (x ** self.params[1]) + self.params[2]
    def __repr__(self):
        return f"{self.params[0]:.{REPR_PRECISION}g} * x^{self.params[1]:.{REPR_PRECISION}g} + {self.params[2]:.{REPR_PRECISION}g}"


class PolynomialLog(Solver):
    name = "O(n^p log^d n)"
    params = np.array([1., 1., 1., 1.]) * 1e-5
    def __call__(self, x):
        return self.params[0] * (x ** self.params[1]) * (np.log(x) ** self.params[2]) + self.params[3]
    def __repr__(self):
        return f"{self.params[0]:.{REPR_PRECISION}g} * x^{self.params[1]:.{REPR_PRECISION}g} * log^{self.params[2]:.{REPR_PRECISION}g} x + {self.params[3]:.{REPR_PRECISION}g}"


SOLVERS_ALL = [
    Constant,
    Log,
    Linear,
    LinearLog,
    Quadratic,
    QuadraticLog,
    Cubic,
    CubicLog,
    Exponential,
]

SOLVERS_EXTRA = [
    Polynomial,
    PolynomialLog,
]

def main():
    import time
    x = np.linspace(1, 100, 1000, dtype=np.float64)
    a, b = 3.82, 14.51
    y = a * x * np.log(x) + b + np.random.normal(0, 1, len(x))
    print(a, b)
    solver = LinearLog()
    t0 = time.perf_counter()
    solver.fit_minimize(x, y)
    print(solver, time.perf_counter() - t0)
    solver = LinearLog()
    t0 = time.perf_counter()
    solver.fit(x, y)
    print(solver, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
    