from tqdm import tqdm as _tqdm

def tqdm(**kwargs):
    kwargs["bar_format"] = '{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}'
    return _tqdm(**kwargs)

def no_tqdm(**kwargs):
    return kwargs["iterable"]
