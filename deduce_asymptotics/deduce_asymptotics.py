import logging
import time
from typing import Callable, Any, Tuple, List, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning

from .solvers import Solver, SOLVERS_ALL, SOLVERS_EXTRA
from .utils import suppress_warnings


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logging.basicConfig(format='%(asctime)s - %(message)s')


def measure_runtime(f: Callable, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    f(*args, **kwargs)
    t1 = time.perf_counter()
    return t1 - t0


def collect_data(
    f: Callable[..., Any],
    build_input: Callable[[int], Any],
    time_budget: float,
    num_samples: int,
    step: Callable,
    start: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logging.info(f"Collecting data for {f.__name__}...")
    n_values = []
    times = []
    errors = []

    n = start
    time_start = time.perf_counter()
    iteration = 0
    while time.perf_counter() - time_start < time_budget:
        n_values.append(n)
        single_run_times = []
        for _ in range(num_samples):  # Repeat several times to account for randomness
            input_data = build_input(n)
            runtime = measure_runtime(f, input_data)
            single_run_times.append(runtime)
        
        avg_runtime = np.mean(single_run_times)
        std_runtime = np.std(single_run_times)
        times.append(avg_runtime)
        errors.append(std_runtime)
        logging.info(f"Iteration {iteration:3}. Input length: {n}, Avg time: {avg_runtime:.4g} ± {std_runtime:.4g} seconds")

        n = max(step(n), n + 1)
        iteration += 1

    return np.array(n_values), np.array(times), np.array(errors)


@suppress_warnings
def fit_time_complexity(n_values, t_values, solver_classes=SOLVERS_ALL) -> Solver:
    logging.info(f"Starting the fit...")
    logging.info(f"Potential candidates: {[s.name for s in solver_classes]}")
    t0 = time.perf_counter()

    solvers = []
    losses = []
    X, Y = np.array(n_values, dtype=float), np.array(t_values, dtype=float)

    if len(X) <= 10:
        test_horizon = 1
    elif len(X) <= 15:
        test_horizon = len(X) - 10
    else:
        test_horizon = 5

    # test_horizon = 1

    for solver_class in solver_classes:
        solver = solver_class()

        w = 0
        for i in range(test_horizon, 0, -1):
            solver.fit(X[:-i], Y[:-i])
            w += solver.loss(X[-i], Y[-i])

        solver.fit(X, Y)
        loss = solver.loss(X, Y)
        loss *= w
        if np.isnan(loss):
            loss = np.inf
        solvers.append(solver)
        losses.append(loss)
        logging.info(f"Solver {solver.name:15}  loss = {loss:5g}")

    best_solver = solvers[np.argmin(losses)]

    logging.info(f"Optimization took: {time.perf_counter() - t0:4g} (sec)")
    logging.info(f"Best fit: {best_solver.name} with parameters {best_solver.params}")
    return best_solver


def plot_data(
    x: np.ndarray,
    y: np.ndarray,
    errors: np.ndarray = None,
    solver: Solver = None,
    f_name: str = None,
) -> None:
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    if errors is None:
        errors = np.zeros_like(y)

    x_ = np.linspace(x[0], x[-1], len(x) * 20, dtype=float)
    for ax in [ax1, ax2]:
        ax.errorbar(x, y, yerr=errors, fmt='o', label="Measured times", capsize=5, color='blue')
        if solver is not None:
            ax.plot(
                x_, solver(x_),
                label=f"Best fit: {solver.name}   = {solver}", color='red'
            )
        ax.set_xlabel("Input size (n)")
        ax.set_ylabel("Runtime (seconds)")
        ax.legend()

    ax1.set_title(f"Runtime of function {f_name} (Regular Scale)")
    ax2.set_title(f"Runtime of function {f_name} (Logarithmic Scale)")
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.show()


def deduce(
    f: Callable[..., Any],
    build_input: Callable[[int], Any],
    time_budget: float = 10.,
    num_samples: int = 10,
    step: Callable = lambda n: int(n * 1.1),
    start: int = 64,
    extra: bool = False
) -> None:
    n_values, times, errors = collect_data(f, build_input, time_budget, num_samples, step, start)
    solver_classes = SOLVERS_ALL if not extra else SOLVERS_EXTRA + SOLVERS_ALL
    best_fit = fit_time_complexity(n_values, times, solver_classes)
    plot_data(n_values, times, errors, best_fit, f.__name__)
    print(f"Time complexity of the function {f.__name__} is {best_fit.name}")
    print(f'Time = {best_fit} (sec)')
    return best_fit, n_values, times


def deduce_helper(
    f: Callable[..., Any],
    build_input: Callable[[int], Any],
    time_budget: float = 10.,
    num_samples: int = 10,
    step: Callable = lambda n: int(n * 1.1),
    start: int = 64,
    extra: bool = False
) -> None:
    n_values, times, errors = collect_data(f, build_input, time_budget, num_samples, step, start)
    solver_classes = SOLVERS_ALL if not extra else SOLVERS_EXTRA + SOLVERS_ALL
    best_fit = fit_time_complexity(n_values, times, solver_classes)
    return best_fit, n_values, times


class Deducer(object):
    function: Callable[(...), Any]
    build_input: Callable[[int], Any]

    xs: list[float] = None
    ys: list[float] = None
    errors: list[float] = None
    best_solver: Solver = None
    solvers: list[Solver] = None
    losses: list[float] = None

    def __init__(self,
        function: Callable[(...), Any],
        build_input: Callable[[int], Any],
    ):
        self.function = function
        self.build_input = build_input

        self.xs = []
        self.ys = []
        self.errors = []
        self.best_solver = None
        self.solvers = []
        self.losses = []
    
    def measure_runtime(self, n: int):
        data = self.build_input(n)
        t0 = time.perf_counter()
        _ = self.function(data)
        return time.perf_counter() - t0      

    def get_next_n(self, n: int) -> int:
        n = int(n)
        if not hasattr(self, "_step"):
            self._step = lambda n: int(n * 1.1)
        return max(self._step(n), n + 1)
    
    def collect(self,
        time_budget: float = 10,    # seconds
        num_samples: int = 10,
    ) -> None:
        """Measures runtime of funciton for different input sizes."""
        logging.info(f"Collecting data for {self.function.__name__}...")
        time_start = time.perf_counter()
        iteration = len(self.xs)
        n = 2 if iteration == 0 else self.get_next_n(self.xs[-1])
        while time.perf_counter() - time_start < time_budget:
            self.xs.append(n)
            single_run_times = []
            for _ in range(num_samples):  # Repeat several times to account for randomness
                runtime = self.measure_runtime(n)
                single_run_times.append(runtime)
            
            avg_runtime = np.mean(single_run_times)
            std_runtime = np.std(single_run_times)
            self.ys.append(avg_runtime)
            self.errors.append(std_runtime)
            logging.info(f"Iteration {iteration:3}. Input length: {n}, Avg time: {avg_runtime:.4n} ± {std_runtime:.4n} seconds")

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
    ) -> 'Deducer':
        """Fits solvers from 'solver_classes' to previosly collected data."""
        logging.info(f"Starting the fit...")
        logging.info(f"Potential candidates: {[s.name for s in solver_classes]}")
        t0 = time.perf_counter()
        X, Y = np.array(self.xs, dtype=float), np.array(self.ys, dtype=float)

        if len(X) <= 10:
            test_horizon = 1
        elif len(X) <= 15:
            test_horizon = len(X) - 10
        else:
            test_horizon = 5

        # test_horizon = 1

        for solver_class in solver_classes:
            solver, idx = self.get_solver(solver_class)

            w = 0
            for i in range(test_horizon, 0, -1):
                solver.fit(X[:-i], Y[:-i])
                w += solver.loss(X[-i], Y[-i])

            solver.fit(X, Y)
            loss = solver.loss(X, Y)
            loss *= w
            if np.isnan(loss):
                loss = np.inf
            self.solvers[idx] = solver
            self.losses[idx] = loss
            logging.info(f"Solver {solver.name:15}  loss = {loss:6.5n}")

        self.best_solver = self.solvers[np.argmin(self.losses)]
        logging.info(f"Optimization took: {time.perf_counter() - t0:.4n} (sec)")
        logging.info(f"Best fit: {self.best_solver.name} with parameters {self.best_solver.params}")
        return self

    def report(self) -> None:
        if self.solvers is None:
            print("No data to report. Run 'collect' or 'deduce' first.")
            return
        for s, loss in zip(self.solvers, self.losses):
            print(f'{s.name:15} =   {str(s):50}:   loss={loss:8.4n},   error={s.loss(np.array(self.xs), np.array(self.ys)):8.4n}')

    def plot(self) -> None:
        if self.xs is None:
            print("No data to display. Run 'collect' or 'deduce' first.")
            return
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        x_ = np.linspace(self.xs[0], self.xs[-1], len(self.xs) * 20, dtype=float)
        for ax in [ax1, ax2]:
            ax.errorbar(self.xs, self.ys, yerr=self.errors, fmt='o',
                        label="Measured times", capsize=5, color='blue')
            if self.best_solver is not None:
                ax.plot(
                    x_, self.best_solver(x_),
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
        extras: bool=False,
    ) -> 'Deducer':
        self._step = step
        self.collect(time_budget, num_samples)
        solver_classes = SOLVERS_ALL if not extras else SOLVERS_EXTRA
        self.fit(solver_classes)
        # self.report()
        # self.plot()
        return self
