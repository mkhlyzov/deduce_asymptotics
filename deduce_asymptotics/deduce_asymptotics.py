import logging
import time
from typing import Callable, Any, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def measure_runtime(f: Callable, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    f(*args, **kwargs)
    t1 = time.perf_counter()
    return t1 - t0


def fit_time_complexity(n_values, y_values):
    logging.info(f"Starting the fit...")
    # logging.info(f"Values: {(n, y) for n, y in zip(n_values, y_values)}")
    complexities = {
        "O(1)":         lambda n, a   : a * np.ones_like(n),
        "O(log n)":     lambda n, a, b: a * np.log(n) + b,
        "O(n)":         lambda n, a, b: a * n + b,
        "O(n log n)":   lambda n, a: a * n * np.log(n),
        "O(n^2)":       lambda n, a, b: a * n ** 2 + b,
        "O(n^2 log n)": lambda n, a, b: a * n ** 2 * np.log(n) + b,
        "O(n^3)":       lambda n, a, b: a * n ** 3 + b,
        "O(n^3 log n)": lambda n, a, b: a * n ** 3 * np.log(n) + b,
        "O(e^n)":       lambda n, a, b: a * np.exp(n) + b,
    }
    logging.info(f"Potential candidates: {complexities.keys()}")

    best_fit = None
    best_fit_name = None
    min_error = float('inf')

    for name, complexity in complexities.items():
        try:
            popt, _ = curve_fit(complexity, n_values, y_values, maxfev=10000)
            fitted_times = complexity(n_values, *popt)
            error = np.mean((y_values - fitted_times) ** 2)
            if error < min_error:
                min_error = error
                best_fit = (complexity, popt)
                best_fit_name = name
        except RuntimeError:
            continue

    logging.info(f"Best fit: {best_fit_name} with parameters {best_fit[1]}")
    return best_fit, best_fit_name


def collect_data(
    build_input: Callable[[int], Any],
    f: Callable[..., Any],
    time_budget: float,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logging.info(f"Collecting data for {f.__name__}...")
    n_values = []
    times = []
    errors = []

    n = 32
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

        n *= 2  # Exponentially increasing n
        iteration += 1

    return np.array(n_values), np.array(times), np.array(errors)


def plot_data(
    n_values: np.ndarray,
    times: np.ndarray,
    errors: np.ndarray,
    best_fit: Tuple[Callable, List[float]],
    best_fit_name: str,
    f_name: str
) -> None:
    # Create subplots
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Regular scale plot
    ax1.errorbar(n_values, times, yerr=errors, fmt='o', label="Measured times", capsize=5, color='blue')
    x = np.linspace(n_values[0], n_values[-1], 1_000)
    ax1.plot(x, best_fit[0](x, *best_fit[1]), label=f"Best fit: {best_fit_name}", color='red')
    ax1.set_xlabel("Input size (n)")
    ax1.set_ylabel("Runtime (seconds)")
    ax1.legend()
    ax1.set_title(f"Runtime of function {f_name} (Regular Scale)")

    # Logarithmic scale plot
    ax2.errorbar(n_values, times, yerr=errors, fmt='o', label="Measured times", capsize=5, color='blue')
    ax2.plot(x, best_fit[0](x, *best_fit[1]), label=f"Best fit: {best_fit_name}", color='red')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel("Input size (n)")
    ax2.set_ylabel("Runtime (seconds)")
    ax2.legend()
    ax2.set_title(f"Runtime of function {f_name} (Logarithmic Scale)")

    plt.show()


def deduce(
    build_input: Callable[[int], Any],
    f: Callable[..., Any],
    time_budget: float = 10.,
    num_samples: int = 5
) -> None:
    n_values, times, errors = collect_data(build_input, f, time_budget, num_samples)
    best_fit, best_fit_name = fit_time_complexity(n_values, times)
    plot_data(n_values, times, errors, best_fit, best_fit_name, f.__name__)
    print(f"Time complexity of the function {f.__name__} is {best_fit_name}")
