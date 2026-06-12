import numpy as np


def safelog(value):
    return np.log(value + 1e-16)

def safedivide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=(np.abs(denominator) > 1e-8),
    )

def normalize(value):
    return safedivide(value, np.sum(value))