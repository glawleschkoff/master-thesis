import numpy as np


def safelog(value):
    eps = np.finfo(float).eps
    safe_value = np.clip(value, eps, None)
    safe_log = np.log(safe_value)
    return safe_log


def normalize(value):
    return value / np.sum(value)