from datetime import datetime
import time
from typing import Callable, Any, Tuple, List, Iterable

import matplotlib.pyplot as plt
import numpy as np

from .solvers import Solver, SOLVERS_ALL, SOLVERS_EXTRA
from .utils import suppress_warnings


class Deducer(object):
    function: Callable[(...), Any]
    build_input: Callable[[int], Any]

    _data: dict[float, list[float]]
    best_solver: Solver = None
    solvers: list[Solver] = None
    losses: list[float] = None

    def __init__(self,
        function: Callable[(...), Any],
        build_input: Callable[[int], Any],
    ):
        self.function = function
        self.build_input = build_input

        self._data = {}
        self.best_solver = None
        self.solvers = []
        self.losses = []

    @property
    def data(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        if len(self._data) == 0:
            x, y = (), ()
        else:
            data = [(x, y) for x in sorted(self._data) for y in self._data[x]]
            x, y = zip(*data)
        x, y = np.array(x, dtype=float), np.array(y, dtype=float)
        return x, y
    
    @property
    def xs(self) -> np.ndarray[float]:
        xs = sorted(self._data)
        return np.array(xs, dtype=float)
    
    @property
    def ys(self) -> np.ndarray[float]:
        ys = [np.mean(self._data[x]) for x in sorted(self._data)]
        return np.array(ys, dtype=float)

    @property
    def errors(self) -> np.ndarray[float]:
        errors = [np.std(self._data[x]) for x in sorted(self._data)]
        return np.array(errors, dtype=float)
    
    def measure_runtime(self, n: int):
        data = self.build_input(n)
        t0 = time.perf_counter()
        _ = self.function(data)
        dt = time.perf_counter() - t0

        if n not in self._data:
            self._data[n] = []
        self._data[n].append(dt)

        return dt

    def get_next_n(self, n: int) -> int:
        n = int(n)
        if not hasattr(self, "_step"):
            self._step = lambda n: int(n * 1.1)
        return max(self._step(n), n + 1)
    
    def collect(self,
        time_budget: float = 10,    # seconds
        num_samples: int = 10,
        start: int|None = None,
        verbose=False,
    ) -> None:
        """Measures runtime of funciton for different input sizes."""
        verboseprint = print if verbose else lambda *a, **k: None
        verboseprint(f"{datetime.now():%H:%M:%S} Collecting data for '{self.function.__name__}'...")
        time_start = time.time()
        iteration = len(self._data)
        n = (start if start is not None else
            2 if iteration == 0 else
            self.get_next_n(np.max(self.xs))
        )
        while time.time() - time_start < time_budget:
            for _ in range(num_samples):  # Repeat several times to account for randomness
                self.measure_runtime(n)  # this call automatycally saves measurements
            
            avg_runtime = np.mean(self._data[n])
            std_runtime = np.std(self._data[n])
            verboseprint((
                f"{datetime.now():%H:%M:%S} Iteration {iteration:3}. Input length: {n}, "
                f"Avg time: {avg_runtime:.4n} ± {std_runtime:.4n} seconds"
            ))
            n = self.get_next_n(n)
            iteration += 1

    def get_solver(self, solver_class: type[Solver]) -> tuple[Solver, int]:
        for i, s in enumerate(self.solvers):
            if isinstance(s, solver_class):
                return s, i

        solver = solver_class()
        self.solvers.append(solver)
        self.losses.append(float('inf'))
        return solver, len(self.solvers) - 1
    
    @suppress_warnings
    def fit(self,
        solver_classes: list[type[Solver]] = SOLVERS_ALL,
        verbose=False,
    ) -> 'Deducer':
        """Fits solvers from 'solver_classes' to previosly collected data."""
        verboseprint = print if verbose else lambda *a, **k: None
        verboseprint(f"{datetime.now():%H:%M:%S} Starting the fit...")
        verboseprint(f"{datetime.now():%H:%M:%S} Potential candidates: {[s.name for s in solver_classes]}")
        t0 = time.perf_counter()
        X, Y = self.data

        if len(X) <= 10:
            test_horizon = 1
        elif len(X) <= 15:
            test_horizon = len(X) - 10
        else:
            test_horizon = 5

        # test_horizon = 1

        for solver_class in solver_classes:
            solver, idx = self.get_solver(solver_class)

            w = 1
            # w = 0
            # h = len(X) // len(set(X))
            # for i in range(test_horizon, 0, -1):
            #     solver.fit(X[:-i*h], Y[:-i*h])
            #     w += solver.loss(X[-i*h:], Y[-i*h:])

            solver.fit_genetic_l2(X, Y)
            loss = solver.loss(X, Y)
            loss *= w
            if np.isnan(loss):
                loss = np.inf
            # solver.fit_genetic_l1(X, Y)
            self.solvers[idx] = solver
            self.losses[idx] = loss
            verboseprint(f"{datetime.now():%H:%M:%S} Solver {solver.name:15}  loss = {loss:6.5n}")

        self.best_solver = self.solvers[np.argmin(self.losses)]
        verboseprint(f"{datetime.now():%H:%M:%S} Optimization took: {time.perf_counter() - t0:.4n} (sec)")
        verboseprint(f"{datetime.now():%H:%M:%S} Best fit: {self.best_solver.name} with parameters {self.best_solver.params}")
        return self

    def report(self) -> None:
        if self.solvers is None:
            print("No data to report. Run 'collect' or 'deduce' first.")
            return
        for s, loss in zip(self.solvers, self.losses):
            print(f'{s.name:15} =   {str(s):50}:   loss={loss:8.4n},   error={s.loss(*self.data):8.4n}')

    def plot(self) -> None:
        if len(self._data) == 0:
            print("No data to display. Run 'collect' or 'deduce' first.")
            return
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        X, Y = self.data

        x_ = np.linspace(X[0], X[-1], len(self._data) * 50, dtype=float)
        for ax in [ax1, ax2]:
            # ax.errorbar(self.xs, self.ys, yerr=self.errors, fmt='o',
            #             label="Measured times", capsize=5, color='blue')
            ax.plot(X, Y, '.', alpha=0.9, label="Measured times", color='blue')
            if self.best_solver is not None:
                ax.plot(
                    x_, self.best_solver(x_), alpha=0.9,
                    label=f"Best fit: {self.best_solver.name}   = {self.best_solver}", color='red'
                )
            ax.set_xlabel("Input size (n)")
            ax.set_ylabel("Runtime (seconds)")
            ax.legend()

        ax1.set_title(f"Runtime of function {self.function.__name__} (Regular Scale)")
        ax2.set_title(f"Runtime of function {self.function.__name__} (Logarithmic Scale)")
        ax2.set_xscale('log')
        ax2.set_yscale('log')

        plt.show()

    def plot_err(self) -> None:
        if len(self._data) == 0:
            print("No data to display. Run 'collect' or 'deduce' first.")
            return
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        X, Y = self.xs, self.ys

        x_ = np.linspace(X[0], X[-1], len(X) * 50, dtype=float)
        for ax in [ax1, ax2]:
            ax.errorbar(X, Y, yerr=self.errors, fmt='.', alpha=0.9,
                        label="Measured times", capsize=5, color='blue')
            if self.best_solver is not None:
                ax.plot(
                    x_, self.best_solver(x_), alpha=0.9,
                    label=f"Best fit: {self.best_solver.name}   = {self.best_solver}", color='red'
                )
            ax.set_xlabel("Input size (n)")
            ax.set_ylabel("Runtime (seconds)")
            ax.legend()

        ax1.set_title(f"Runtime of function {self.function.__name__} (Regular Scale)")
        ax2.set_title(f"Runtime of function {self.function.__name__} (Logarithmic Scale)")
        ax2.set_xscale('log')
        ax2.set_yscale('log')

        plt.show()

    def deduce(self,
        time_budget: float = 10,    # seconds
        num_samples: int = 10,
        step: Callable[[int], int] = lambda n: int(n * 1.1),
        start: int|None = None,
        extras: bool=False,
        verbose=False,
    ) -> 'Deducer':
        self._step = step
        self.collect(time_budget, num_samples, start, verbose)
        solver_classes = SOLVERS_ALL if not extras else SOLVERS_EXTRA
        self.fit(solver_classes, verbose)
        # self.report()
        # self.plot()
        return self
