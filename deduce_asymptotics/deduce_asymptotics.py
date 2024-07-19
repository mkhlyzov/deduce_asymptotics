import logging
import time
from typing import Callable, Any, Tuple, List, Iterable
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

# Suppress specific OptimizeWarnings
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .functions import COMPLEXITIES


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

        # logging.info(f"n = {n}, runtime = {avg_runtime} ± {std_runtime}")
        logging.info(f"Iteration {iteration:3}. Input length: {n}, Avg time: {avg_runtime:.4g} ± {std_runtime:.4g} seconds")

        n = step(n)
        iteration += 1

    return np.array(n_values), np.array(times), np.array(errors)


def fit_time_complexity(n_values, t_values):
    logging.info(f"Starting the fit...")
    # logging.info(f"Values: {(n, y) for n, y in zip(n_values, y_values)}")
    logging.info(f"Potential candidates: {[c.name for c in COMPLEXITIES]}")

    best_fit = None
    best_fit_name = None
    min_error = float('inf')

    for complexity in COMPLEXITIES:
        try:
            popt, _ = curve_fit(complexity._callable, n_values, t_values, maxfev=10000)
            fitted_times = complexity(n_values, *popt)
            error = np.mean((t_values - fitted_times) ** 2)
            if error < min_error:
                min_error = error
                best_fit = (complexity, popt)
        except RuntimeError:
            continue

    logging.info(f"Best fit: {best_fit[0].name} with parameters {best_fit[1]}")
    return best_fit


def plot_data(
    n_values: np.ndarray,
    times: np.ndarray,
    errors: np.ndarray,
    best_fit: Tuple[Callable, List[float]],
    f_name: str
) -> None:
    # Create subplots
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Regular scale plot
    ax1.errorbar(n_values, times, yerr=errors, fmt='o', label="Measured times", capsize=5, color='blue')
    x = np.linspace(n_values[0], n_values[-1], 1_000)
    ax1.plot(
        x, best_fit[0](x, *best_fit[1]),
        label=f"Best fit: {best_fit[0].name}   = {best_fit[0].repr(*best_fit[1])}", color='red'
    )
    ax1.set_xlabel("Input size (n)")
    ax1.set_ylabel("Runtime (seconds)")
    ax1.legend()
    ax1.set_title(f"Runtime of function {f_name} (Regular Scale)")

    # Logarithmic scale plot
    ax2.errorbar(n_values, times, yerr=errors, fmt='o', label="Measured times", capsize=5, color='blue')
    ax2.plot(
        x, best_fit[0](x, *best_fit[1]),
        label=f"Best fit: {best_fit[0].name}   = {best_fit[0].repr(*best_fit[1])}", color='red'
    )
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel("Input size (n)")
    ax2.set_ylabel("Runtime log(seconds)")
    ax2.legend()
    ax2.set_title(f"Runtime of function {f_name} (Logarithmic Scale)")

    plt.show()


def deduce(
    f: Callable[..., Any],
    build_input: Callable[[int], Any],
    time_budget: float = 10.,
    num_samples: int = 10,
    step: Callable = lambda n: int(n * 1.1),
    start: int = 64,
) -> None:
    n_values, times, errors = collect_data(f, build_input, time_budget, num_samples, step, start)
    best_fit = fit_time_complexity(n_values, times)
    plot_data(n_values, times, errors, best_fit, f.__name__)
    print(f"Time complexity of the function {f.__name__} is {best_fit[0].name}")
    print(f'Time = {best_fit[0].repr(*best_fit[1])} (sec)')
