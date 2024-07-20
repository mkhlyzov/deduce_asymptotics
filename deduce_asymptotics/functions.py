from collections.abc import Callable
from typing import Iterable
from numbers import Number

import numpy as np


class Function(object):
    def __init__(
        self,
        name: str,
        callable: Callable[..., Number],
        repr: Callable[..., str],
    ):
        self.name = name
        self._callable = callable
        self._repr = repr
        self.last_args = None

    def __call__(self, n, *args) -> float:
        self.last_args = args
        return self._callable(n, *args)

    def repr(self, *args) -> str:
        self.last_args = args
        return self._repr(*args)


REPR_PRECISION = 4
COMPLEXITIES = [
    Function('O(1)',
        lambda n, a: a * np.ones_like(n),
        lambda    a: f'{a:.{REPR_PRECISION}g} * n',
    ),
    Function('O(log n)',
        lambda n, a, b: a * np.log(n) + b,
        lambda    a, b: f'{a:.{REPR_PRECISION}g} * log(n) + {b:.{REPR_PRECISION}g}',
    ),
    Function('O(n)',
        lambda n, a, b: a * n + b,
        lambda    a, b: f'{a:.{REPR_PRECISION}g} * n + {b:.{REPR_PRECISION}g}',
    ),
    Function('O(n log n)',
        lambda n, a, b: a * n * np.log(n) + b,
        lambda    a, b: f'{a:.{REPR_PRECISION}g} * n * log(n) + {b:.{REPR_PRECISION}g}',
    ),
    Function('O(n^2)',
        lambda n, a, b: a * n ** 2 + b,
        lambda    a, b: f'{a:.{REPR_PRECISION}g} * n^2 + {b:.{REPR_PRECISION}g}',
    ),
    Function('O(n^2 log n)',
        lambda n, a, b: a * n ** 2 * np.log(n) + b,
        lambda    a, b: f'{a:.{REPR_PRECISION}g} * n^2 * log(n) + {b:.{REPR_PRECISION}g}',
    ),
    Function('O(n^3)',
        lambda n, a, b: a * n ** 3 + b,
        lambda    a, b: f'{a:.{REPR_PRECISION}g} * n^3 + {b:.{REPR_PRECISION}g}',
    ),
    Function('O(n^3 log n)',
        lambda n, a, b: a * n ** 3 * np.log(n) + b, 
        lambda    a, b: f'{a:.{REPR_PRECISION}g} * n^3 * log(n) + {b:.{REPR_PRECISION}g}',
    ),
]

COMPLEXITIES_EXTRA = [
    Function('O(n^k)',
        lambda n, a, b, k: a * n ** k + b,
        lambda    a, b, k: f'{a:.{REPR_PRECISION}g} * n^{k:.{REPR_PRECISION}g} + {b:.{REPR_PRECISION}g}',
    ),
    Function('O(n^k log^m n)',
        lambda n, a, b, k, m: a * (n ** k) * (np.log(n) ** m) + b,
        lambda    a, b, k, m: f'{a:.{REPR_PRECISION}g} * n^{k:.{REPR_PRECISION}g} * log(n)^{m:.{REPR_PRECISION}g} + {b:.{REPR_PRECISION}g}',
    ),
    Function('O(e^n)',
        lambda n, a, b: a * np.exp(n) + b,
        lambda    a, b: f'{a:.{REPR_PRECISION}g} * exp(n) + {b:.{REPR_PRECISION}g}',
    ),
]
