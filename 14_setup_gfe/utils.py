import numpy as np
from scipy.optimize import root
from scipy.special import softmax



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

def optimize_s(A, C, q_s_init, message):

    nu_init = safelog(q_s_init)

    def error_function(nu):
        q_s = softmax(nu)
        q_o = np.einsum('ij,j->i', A, q_s)
        f_tilde = np.exp(np.einsum('ij,ij->j', A, safelog((A * C[:, np.newaxis]) / np.clip(q_o[:, np.newaxis], a_min=np.finfo(float).eps, a_max=None))))
        q_s_target = (f_tilde * message) / np.sum(f_tilde * message)
        q_s_target = np.clip(q_s_target, 1e-5, None)
        q_s_target = q_s_target / np.sum(q_s_target)
        
        return q_s - q_s_target
    
    solution = root(error_function, x0=nu_init, method='lm', options={'maxiter': 1000})
    
    if solution.success:
        q_s_star = softmax(solution.x)
        return q_s_star
    else:
        if "gtol" in solution.message or "orthogonal" in solution.message:
            return softmax(solution.x)
        
        raise RuntimeError(f"Newton-Solver hat unerwartet abgebrochen: {solution.message}")