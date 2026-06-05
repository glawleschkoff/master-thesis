import numpy as np
from scipy.special import logsumexp
from scipy.optimize import root


def safelog(value):
    eps = np.finfo(float).eps
    safe_value = np.clip(value, eps, None)
    safe_log = np.log(safe_value)
    return safe_log


def safedivide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=(np.abs(denominator) > 1e-8),
    )


def normalize(value):
    return value / np.sum(value)


def optimize_s(A, C, q_s_init, q_c, message):

    def error_function(q_s):
        q_o = np.einsum('ijk,j,k->i', A, q_s, q_c)
        f_tilde = np.exp(np.einsum('ijk,k,ijk->j', A, q_c, safelog((A * C[:, np.newaxis, np.newaxis]) / np.clip(q_o[:, np.newaxis, np.newaxis], a_min=np.finfo(float).eps, a_max=None))))

        return q_s - ((f_tilde * message) / np.sum(f_tilde * message))
    
    #q_s_init = np.ones(4) / 4
    solution = root(error_function, x0=q_s_init, method='hybr', tol=1e-3)
    
    if solution.success:
        q_s_star = solution.x
        return q_s_star
    else:
        raise RuntimeError(f"Newton-Solver hat keine Lösung gefunden: {solution.message}")
    

def optimize_c(A, C, q_c_init, q_s, message):

    def error_function(q_c):
        q_o = np.einsum('ijk,j,k->i', A, q_s, q_c)
        f_tilde = np.exp(np.einsum('ijk,j,ijk->k', A, q_s, safelog((A * C[:, np.newaxis, np.newaxis]) / np.clip(q_o[:, np.newaxis, np.newaxis], a_min=np.finfo(float).eps, a_max=None))))

        return q_c - ((f_tilde * message) / np.sum(f_tilde * message))
    
    #q_c_init = np.ones(2) / 2
    solution = root(error_function, x0=q_c_init, method='hybr', tol=1e-3)
    
    if solution.success:
        q_c_star = solution.x
        return q_c_star
    else:
        raise RuntimeError(f"Newton-Solver hat keine Lösung gefunden: {solution.message}")