import warnings
from functools import wraps


def suppress_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # warnings.simplefilter("ignore")
            result = func(*args, **kwargs)
        return result
    return wrapper
