from tqdm import tqdm as _tqdm
import functools
import time


def tqdm(**kwargs):
    kwargs["bar_format"] = '{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'
    return _tqdm(**kwargs)

def no_tqdm(**kwargs):
    return kwargs["iterable"]

def timer(func):
    def _timer(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        end = time.time()
        self.logger.debug(f"{func.__name__} took {end-start} seconds")
        return result
    return _timer