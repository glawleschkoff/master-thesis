import numpy as np


def safelog(value):
    return np.log(value + 1e-16)

def normalize(value):
    return value/ np.sum(value)