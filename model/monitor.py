from functools import wraps
from time import perf_counter


def monitor_prediction_time():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = perf_counter()

            result = func(*args, **kwargs)

            end_time = perf_counter()

            print(f"Prediction time: {end_time - start_time:.4f} seconds")

            return result

        return wrapper

    return decorator

