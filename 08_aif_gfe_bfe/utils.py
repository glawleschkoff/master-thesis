import numpy as np
from scipy.optimize import root

def safelog(value):
    eps = np.finfo(float).eps
    safe_value = np.clip(value, eps, None)
    safe_log = np.log(safe_value)
    return safe_log

def optimize_q_s(A, C, *messages):
    """
    Berechnet die optimale variationelle Verteilung q(s) über das Newton-Verfahren.
    
    Inputs:
    - A:      numpy array (16, 8) - Beobachtungsmatrix p(o | s)
    - C:      numpy array (16,)   - Ziel-Prior p~(o)
    
    Returns:
    - q_s_star: numpy array (8,)   - Die konvergierte, optimale Verteilung
    """
    
    # --- 1. Verschachtelte Hilfsfunktionen ---
    
    def softmax(x):
        """Numerisch stabiler Softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def error_function(q_s):
        """Berechnet das Residuum F(q_s) für den Solver"""
        
        # Vorhersage der Beobachtung (q_o)
        q_o = A @ q_s
        
        # Spalten-Entropie von p_A berechnen (h_p_A)
        h_p_A = -np.sum(A * safelog(A), axis=0)
        
        # Fehler-Term rho berechnen
        rho = A.T @ (safelog(C) - safelog(q_o)) - h_p_A
        
        # Eingehende Nachrichten im Log-Raum addieren
        log_messages = sum(safelog(msg) for msg in messages)
        
        # Rückgabe der Differenz (soll 0 werden)
        return q_s - softmax(rho + log_messages)

    # --- 2. Vorbereitung und Optimierung ---
    
    # Bestimme die Anzahl der Zustände dynamisch (hier: 8)
    num_states = A.shape[1]
    
    # Startwert für den Solver (Gleichverteilung)
    q_s_init = np.ones(num_states) / num_states
    
    # Solver aufrufen (die verschachtelte Funktion greift automatisch auf p_A, etc. zu)
    solution = root(error_function, x0=q_s_init, method='hybr', tol=1e-3)
    
    # --- 3. Ergebnisverarbeitung ---
    
    if solution.success:
        # Ergebnis auslesen
        q_s_star = solution.x
        
        # Numerische Sicherheitsmaßnahme: Werte zwischen 0 und 1 clippen und normalisieren
        q_s_star = np.clip(q_s_star, 1e-12, 1.0)
        q_s_star /= q_s_star.sum()
        
        return q_s_star
    else:
        # Falls der Solver mal nicht konvergiert, printen wir die genauen Details
        max_error = np.max(np.abs(solution.fun))
        
        print("\n" + "="*50)
        print("FEHLER: Newton-Solver ist nicht konvergiert!")
        print("="*50)
        print(f"Ursache (Message): {solution.message}")
        print(f"Status-Code:       {solution.status}")
        print(f"Maximaler Fehler:  {max_error:.4e} (sollte nahe 0 sein)")
        print(f"Letzter Stand q_s: {solution.x}")
        print("="*50 + "\n")

        # Falls der Solver mal nicht konvergiert, werfen wir einen sauberen Fehler
        raise RuntimeError(f"Newton-Solver hat keine Lösung gefunden: {solution.message}")