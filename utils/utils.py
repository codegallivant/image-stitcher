from tqdm import tqdm as _tqdm
import functools
import time


def tqdm(**kwargs):
    kwargs["bar_format"] = '{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'
    return _tqdm(**kwargs)

def no_tqdm(**kwargs):
    return kwargs["iterable"]

def timer(func, out):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        out("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
        return value
    return wrapper