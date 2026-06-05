import numpy as np
from scipy.optimize import root

def safelog(value):
    eps = np.finfo(float).eps
    safe_value = np.clip(value, eps, None)
    safe_log = np.log(safe_value)
    return safe_log

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def optimize_q_s(A, C, *messages, max_iter=50):
    """
    Berechnet die optimale variationelle Verteilung q(s) über das Newton-Verfahren.
    Bricht nach einer festgelegten Anzahl an Iterationen/Funktionsaufrufen ab.
    
    Inputs:
    - A:         numpy array (16, 8) - Beobachtungsmatrix p(o | s)
    - C:         numpy array (16,)   - Ziel-Prior p~(o)
    - messages:  Variable Anzahl an eingehenden Nachrichten
    - max_iter:  int - Maximale Anzahl der Iterationen/Funktionsauswertungen (Standard: 50)
    
    Returns:
    - q_s_star:  numpy array (8,) - Die konvergierte oder nach max_iter erreichte Verteilung
    """
    
    # --- 1. Verschachtelte Hilfsfunktionen ---
    
    def softmax(x):
        """Numerisch stabiler Softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def error_function(q_s):
        """Berechnet das Residuum F(q_s) für den Solver"""
        q_o = A @ q_s
        h_p_A = -np.sum(A * safelog(A), axis=0)
        rho = A.T @ (safelog(C) - safelog(q_o)) - h_p_A
        log_messages = sum(safelog(msg) for msg in messages)
        return q_s - softmax(rho + log_messages)

    # --- 2. Vorbereitung und Optimierung ---
    
    num_states = A.shape[1]
    q_s_init = np.ones(num_states) / num_states
    
    # Solver aufrufen und Limit für Funktionsauswertungen (maxfev) setzen
    solution = root(error_function, x0=q_s_init, method='hybr', tol=1e-3, 
                    options={'maxfev': max_iter})
    
    # --- 3. Ergebnisverarbeitung ---
    
    # Ergebnis auslesen (egal ob konvergiert oder Limit erreicht)
    q_s_star = solution.x
    
    # Numerische Sicherheitsmaßnahme: Werte zwischen 0 und 1 clippen und normalisieren
    q_s_star = np.clip(q_s_star, 1e-12, 1.0)
    q_s_star /= q_s_star.sum()
        
    return q_s_star