import numpy as np
from scipy.optimize import root

def safelog(value):
    eps = np.finfo(float).eps
    safe_value = np.clip(value, eps, None)
    safe_log = np.log(safe_value)
    return safe_log

def optimize_q_s(A, C, *messages):

    def error_function(q_s):
        q_o = np.einsum('ij,j->i', A, q_s)
        f_tilde = np.exp(np.einsum('ij,ij->j', A, safelog((A * C[:, np.newaxis]) / np.clip(q_o[:, np.newaxis], a_min=np.finfo(float).eps, a_max=None))))

        return q_s - ((f_tilde * np.prod(messages, axis=0)) / np.sum(f_tilde * np.prod(messages, axis=0)))
        #return ((f_tilde * np.prod(messages, axis=0)) / np.sum(f_tilde * np.prod(messages, axis=0)))
    
    q_s_init = np.ones(8) / 8

    # q_s = q_s_init
    # for _ in range(100):
    #     q_s = error_function(q_s)
    # return q_s

    
    solution = root(error_function, x0=q_s_init, method='hybr', tol=1e-3)
    
    if solution.success:
        q_s_star = solution.x
        return q_s_star
    else:
        raise RuntimeError(f"Newton-Solver hat keine Lösung gefunden: {solution.message}")