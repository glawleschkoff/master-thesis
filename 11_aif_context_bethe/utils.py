import numpy as np
from scipy.optimize import root


def safelog(value):
    eps = np.finfo(float).eps
    safe_value = np.clip(value, eps, None)
    safe_log = np.log(safe_value)
    return safe_log


def savedivide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=(np.abs(denominator) > 1e-8),
    )


def optimize(
    A, C, q_s_c_init, rightward_context_message, upward_context_messages, *messages
):
    iteration = -1

    def error_function(q_s_c):

        nonlocal iteration
        iteration += 1

        q_s_c = q_s_c.reshape((4, 2))
        q_o = np.einsum("ijk,jk->i", A, q_s_c)

        numerator = A * C[:, np.newaxis, np.newaxis]
        denominator = q_o[:, np.newaxis, np.newaxis]
        term = savedivide(numerator, denominator)
        f_tilde = np.exp(
            np.einsum(
                "ijk,ijk->jk",
                A,
                safelog(term),
            )
        )

        state_messages = np.array(messages)

        if state_messages.shape[0] == 2:
            q_s_c_star = np.einsum(
                "jk,j,j,k,k,k,k->jk",
                f_tilde,
                state_messages[0],
                state_messages[1],
                rightward_context_message,
                upward_context_messages[0],
                upward_context_messages[1],
                upward_context_messages[2],
            )
        else:
            q_s_c_star = np.einsum(
                "jk,j,k,k,k,k->jk",
                f_tilde,
                state_messages[0],
                rightward_context_message,
                upward_context_messages[0],
                upward_context_messages[1],
                upward_context_messages[2],
            )
        q_s_c_star_norm = q_s_c_star / np.sum(q_s_c_star)
        # k = 13
        # if iteration == k:
        #     print('Iteration', k)
        #     print(q_s_c_star_norm)
        #     print(f_tilde)
        #     print()
        # k = 12
        # if iteration == k:
        #     print('Iteration', k)
        #     print(q_s_c_star_norm)
        #     print(f_tilde)
        #     print()
        print('Iteration', iteration)
        print(q_s_c)
        print(q_s_c_star_norm)
        #print(f_tilde)
        print()
        
        return (q_s_c - (q_s_c_star_norm)).flatten()

    q_s_c_init = q_s_c_init

    solution = root(error_function, x0=q_s_c_init, method="hybr", tol=1e-3)

    if solution.success:
        q_s_c_star = solution.x
        return q_s_c_star.reshape((4, 2))
    else:
        raise RuntimeError(
            f"Newton-Solver hat keine Lösung gefunden: {solution.message}"
        )