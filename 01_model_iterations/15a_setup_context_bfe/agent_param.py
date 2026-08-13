import numpy as np

# state names
s = {
    'O' : 0, 'C' : 1, 'L' : 2, 'R' : 3
}

# context state names
s_c = {
    'RL': 0, 'RR': 1
}

# observation names
o = {
    'O_CL' : 0, 'O_CR' : 1, 'O_RW' : 2, 'O_NR' : 3, 'C_CL' : 4, 'C_CR' : 5, 'C_RW' : 6, 'C_NR' : 7,
    'L_CL' : 8, 'L_CR' : 9, 'L_RW' : 10, 'L_NR' : 11, 'R_CL' : 12, 'R_CR' : 13, 'R_RW' : 14, 'R_NR' : 15
}

# action names
u = {
    'O' : 0, 'C' : 1, 'L' : 2, 'R' : 3
}

def generate_agent_params(alpha=0.9, c=2, beta=1):
    # A-Matrix "p(o|s, s_c)" (likelihood mapping from states and contexts to observations)
    A = np.zeros((16, 4, 2))
    A[o['O_CL'], s['O'], s_c['RL']] = 0.5
    A[o['O_CL'], s['O'], s_c['RR']] = 0.5
    A[o['L_RW'], s['L'], s_c['RL']] = alpha
    A[o['L_RW'], s['L'], s_c['RR']] = 1 - alpha
    A[o['R_RW'], s['R'], s_c['RL']] = 1 - alpha
    A[o['R_RW'], s['R'], s_c['RR']] = alpha
    A[o['C_CL'], s['C'], s_c['RL']] = beta
    A[o['C_CR'], s['C'], s_c['RL']] = 1 - beta
    A[o['C_CR'], s['C'], s_c['RR']] = beta
    A[o['C_CL'], s['C'], s_c['RR']] = 1 - beta
    A[o['O_CR'], s['O'], s_c['RL']] = 0.5
    A[o['O_CR'], s['O'], s_c['RR']] = 0.5
    A[o['L_NR'], s['L'], s_c['RL']] = 1 - alpha
    A[o['L_NR'], s['L'], s_c['RR']] = alpha
    A[o['R_NR'], s['R'], s_c['RL']] = alpha
    A[o['R_NR'], s['R'], s_c['RR']] = 1 - alpha

    # B-Tensor "p(s_next|s_current, u_next)" (states' transition probability given an action)
    B = np.zeros((4, 4, 4))
    B[s['O'], s['O'], u['O']] = 1
    B[s['O'], s['C'], u['O']] = 1
    B[s['O'], s['R'], u['O']] = 1
    B[s['O'], s['L'], u['O']] = 1
    B[s['C'], s['O'], u['C']] = 1
    B[s['C'], s['C'], u['C']] = 1
    B[s['O'], s['R'], u['C']] = 1
    B[s['O'], s['L'], u['C']] = 1
    B[s['L'], s['O'], u['L']] = 1
    B[s['L'], s['C'], u['L']] = 1
    B[s['O'], s['R'], u['L']] = 1
    B[s['O'], s['L'], u['L']] = 1
    B[s['R'], s['O'], u['R']] = 1
    B[s['R'], s['C'], u['R']] = 1
    B[s['O'], s['R'], u['R']] = 1
    B[s['O'], s['L'], u['R']] = 1

    # B_c-Tensor "p(s_c_next|s_c_current)" (contexts' transition probability)
    B_c = np.zeros((2, 2))
    B_c[s_c['RL'], s_c['RL']] = 0.99
    B_c[s_c['RR'], s_c['RL']] = 0.01
    B_c[s_c['RR'], s_c['RR']] = 0.99
    B_c[s_c['RL'], s_c['RR']] = 0.01

    # C-Vector "p~(o)" (goal distribution at each timestep)
    C = np.ones(16)
    C = C * (1 / (4 * np.exp(c) + 4 * np.exp(-c) + 8))
    C[o['O_RW']] = np.exp(c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)
    C[o['C_RW']] = np.exp(c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)
    C[o['L_RW']] = np.exp(c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)
    C[o['R_RW']] = np.exp(c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)
    C[o['O_NR']] = np.exp(-c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)
    C[o['C_NR']] = np.exp(-c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)
    C[o['L_NR']] = np.exp(-c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)
    C[o['R_NR']] = np.exp(-c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)

    # D-Vector "p(s_0)" (initial state distribution)
    D = np.zeros(4)
    D[s['O']] = 1

    # D_c-Vector "p(s_0)" (initial context distribution)
    D_c = np.zeros(2)
    D_c[s_c['RL']] = 0.5
    D_c[s_c['RR']] = 0.5

    # U-Vector "p(u)" (prior distribution for actions)
    U = np.ones(4) * 0.25
    
    return A, B, B_c, C, D, D_c, U