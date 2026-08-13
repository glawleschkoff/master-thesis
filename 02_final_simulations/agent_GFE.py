import numpy as np
from utils import safelog, safedivide, normalize, optimize_s_without_context, apply_attention
from scipy.special import softmax

np.set_printoptions(
    linewidth=200,
    precision=3,
    suppress=True
)


class Agent:
    def __init__(self, A, B, C, D, U):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.U = U
        self.time_horizon = 3

        self.current_observation_timestep = 0
        self.current_action_timestep = 0

        self.number_states = B.shape[0]
        self.number_observations = A.shape[0]
        self.number_actions = B.shape[2]

        self.acted_u0 = False
        self.acted_u1 = False
        self.observed_o0 = False
        self.observed_o1 = False
        self.observed_o2 = False

        self.q_s0 = np.ones(self.number_states) / self.number_states
        self.q_s0_I = np.ones(self.number_states) / self.number_states
        self.q_s0_II = np.ones(self.number_states) / self.number_states
        self.q_s1 = np.ones(self.number_states) / self.number_states
        self.q_s1_I = np.ones(self.number_states) / self.number_states
        self.q_s1_II = np.ones(self.number_states) / self.number_states
        self.q_s2 = np.ones(self.number_states) / self.number_states
        self.q_u0 = np.ones(self.number_actions) / self.number_actions
        self.q_u1 = np.ones(self.number_actions) / self.number_actions
        self.q_o0 = np.ones(self.number_observations) / self.number_observations
        self.q_o1 = np.ones(self.number_observations) / self.number_observations
        self.q_o2 = np.ones(self.number_observations) / self.number_observations
        # A Node Beliefs:
        self.q_o0_s0_II = np.ones((self.number_observations, self.number_states)) / (self.number_observations * self.number_states)
        self.q_o1_s1_II = np.ones((self.number_observations, self.number_states)) / (self.number_observations * self.number_states)
        self.q_o2_s2 = np.ones((self.number_observations, self.number_states)) / (self.number_observations * self.number_states)
        # B Node Beliefs:
        self.q_s1_s0_I_u0 = np.ones((self.number_states, self.number_states, self.number_actions)) / (self.number_states * self.number_states * self.number_actions)
        self.q_s2_s1_I_u1 = np.ones((self.number_states, self.number_states, self.number_actions)) / (self.number_states * self.number_states * self.number_actions)
        # Equality Node Beliefs:
        self.q_s0_s0_I_s0_II = np.ones((self.number_states, self.number_states, self.number_states)) / (self.number_states * self.number_states * self.number_states)
        self.q_s1_s1_I_s1_II = np.ones((self.number_states, self.number_states, self.number_states)) / (self.number_states * self.number_states * self.number_states)
        # Loop Overarching Messages
        self.mu_down_s0_II = np.ones(8) / 8
        self.mu_down_s1_II = np.ones(8) / 8
        self.mu_down_s2 = np.ones(8) / 8

    def infer(self, iterations=None):
        tolerance = 1e-4
        previous_free_energy = 0
        current_free_energy = float("inf")
        counter = 0
        
        while (
            (iterations is None and abs(current_free_energy - previous_free_energy) > tolerance)
            or (iterations is not None and iterations > 0)
        ):
            previous_free_energy = current_free_energy    
            if iterations is not None:
                iterations -= 1        

            # Bethe Forward Sweep:
            mu_right_s0 = self.D
            if self.observed_o0:
                mu_up_s0_II = softmax(np.einsum('i,ij->j', self.q_o0, safelog(self.A)))
            else:
                q_s = optimize_s_without_context(self.A, self.C, self.q_s0_II, self.mu_down_s0_II)
                mu_up_s0_II = normalize(safedivide(q_s, self.mu_down_s0_II))
            mu_right_s0_I = normalize(mu_right_s0 * mu_up_s0_II)
            if self.acted_u0:
                mu_down_u0 = self.q_u0
            else:
                mu_down_u0 = self.U
            mu_right_s1 = normalize(np.einsum('i,j,kij->k', mu_right_s0_I, mu_down_u0, self.B))
            if self.observed_o1:
                mu_up_s1_II = softmax(np.einsum('i,ij->j', self.q_o1, safelog(self.A)))
            else:
                q_s = optimize_s_without_context(self.A, self.C, self.q_s1_II, self.mu_down_s1_II)
                mu_up_s1_II = normalize(safedivide(q_s, self.mu_down_s1_II))
            mu_right_s1_I = normalize(mu_right_s1 * mu_up_s1_II)
            if self.acted_u1:
                mu_down_u1 = self.q_u1
            else:
                mu_down_u1 = self.U
            self.mu_down_s2 = normalize(np.einsum('i,j,kij->k', mu_right_s1_I, mu_down_u1, self.B))

            # Bethe Backward Sweep:
            if self.observed_o2:
                mu_left_s2 = softmax(np.einsum('i,ij->j', self.q_o2, safelog(self.A)))
            else:
                q_s = optimize_s_without_context(self.A, self.C, self.q_s2, self.mu_down_s2)
                mu_left_s2 = normalize(safedivide(q_s, self.mu_down_s2))
            if self.acted_u1:
                mu_up_u1 = self.q_u1
            else:
                mu_up_u1 = normalize(np.einsum('i,j,jik->k', mu_right_s1_I, mu_left_s2, self.B))
            mu_left_s1_I = normalize(np.einsum('i,j,jki->k', mu_down_u1, mu_left_s2, self.B))
            self.mu_down_s1_II = normalize(mu_right_s1 * mu_left_s1_I)
            mu_left_s1 = normalize(mu_up_s1_II * mu_left_s1_I)
            if self.acted_u0:
                mu_up_u0 = self.q_u0
            else:
                mu_up_u0 = normalize(np.einsum('i,j,jik->k', mu_right_s0_I, mu_left_s1, self.B))
            mu_left_s0_I = normalize(np.einsum('i,j,jki->k', mu_down_u0, mu_left_s1, self.B))
            self.mu_down_s0_II = normalize(mu_right_s0 * mu_left_s0_I)
            mu_left_s0 = normalize(mu_up_s0_II * mu_left_s0_I)

            # Updating Beliefs:
            self.q_s0 = normalize(mu_left_s0 * mu_right_s0)
            self.q_s0_I = normalize(mu_left_s0_I * mu_right_s0_I)
            self.q_s0_II = normalize(mu_up_s0_II * self.mu_down_s0_II)
            self.q_s1 = normalize(mu_left_s1 * mu_right_s1)
            self.q_s1_I = normalize(mu_left_s1_I * mu_right_s1_I)
            self.q_s1_II = normalize(mu_up_s1_II * self.mu_down_s1_II)
            self.q_s2 = normalize(mu_left_s2 * self.mu_down_s2)
            if not self.acted_u0:
                q_u = normalize(mu_up_u0 * mu_down_u0)
                x = np.zeros(self.number_actions)
                x[np.argmax(q_u)] = 1
                #self.q_u0 = x
                self.q_u0 = apply_attention(probs=q_u, attention_param=4.0)
            if not self.acted_u1:
                q_u = normalize(mu_up_u1 * mu_down_u1)
                x = np.zeros(self.number_actions)
                x[np.argmax(q_u)] = 1
                #self.q_u1 = x
                self.q_u1 = apply_attention(probs=q_u, attention_param=4.0)
            if not self.observed_o0:
                self.q_o0 = np.einsum('i,ji->j', self.q_s0_II, self.A)
            if not self.observed_o1:
                self.q_o1 = np.einsum('i,ji->j', self.q_s1_II, self.A)
            if not self.observed_o2:
                self.q_o2 = np.einsum('i,ji->j', self.q_s2, self.A)
            self.q_s1_s0_I_u0 = normalize(np.einsum('ijk,i,j,k->ijk', self.B, mu_left_s1, mu_right_s0_I, mu_down_u0))
            self.q_s2_s1_I_u1 = normalize(np.einsum('ijk,i,j,k->ijk', self.B, mu_left_s2, mu_right_s1_I, mu_down_u1))
            self.q_s0_s0_I_s0_II = normalize(np.einsum('i,i,i->i', mu_right_s0, mu_up_s0_II, mu_left_s0_I))
            self.q_s1_s1_I_s1_II = normalize(np.einsum('i,i,i->i', mu_right_s1, mu_up_s1_II, mu_left_s1_I))

            current_free_energy = self.free_energy()
            #print('Iteration:', str(counter)+",", 'FE:', current_free_energy)
            counter += 1

    def free_energy(self):
        D_free_energy = np.einsum('i,i', self.q_s0, safelog(self.q_s0)) - np.einsum('i,i', self.q_s0, safelog(self.D))
        B0_free_energy = np.einsum('ijk,ijk', self.q_s1_s0_I_u0, safelog(self.q_s1_s0_I_u0)) - np.einsum('ijk,ijk', self.q_s1_s0_I_u0, safelog(self.B))
        B1_free_energy = np.einsum('ijk,ijk', self.q_s2_s1_I_u1, safelog(self.q_s2_s1_I_u1)) - np.einsum('ijk,ijk', self.q_s2_s1_I_u1, safelog(self.B))
        U0_free_energy = np.einsum('i,i', self.q_u0, safelog(self.q_u0)) - np.einsum('i,i', self.q_u0, safelog(self.U))
        U1_free_energy = np.einsum('i,i', self.q_u1, safelog(self.q_u1)) - np.einsum('i,i', self.q_u1, safelog(self.U))
        if self.observed_o0:
            A0_free_energy = np.einsum('i,i', self.q_o0, safelog(self.q_o0)) + np.einsum('i,i', self.q_s0_II, safelog(self.q_s0_II)) - np.einsum('i,j,ij', self.q_o0, self.q_s0_II, safelog(self.A))
        else:
            A0_free_energy = np.einsum('ij,j,i', self.A, self.q_s0_II, safelog(self.q_o0)) + np.einsum('ij,j,j->', self.A, self.q_s0_II, safelog(self.q_s0_II)) - np.einsum('ij,j,ij', self.A, self.q_s0_II, safelog(self.A))
        if self.observed_o1:
            A1_free_energy = np.einsum('i,i', self.q_o1, safelog(self.q_o1)) + np.einsum('i,i', self.q_s1_II, safelog(self.q_s1_II)) - np.einsum('i,j,ij', self.q_o1, self.q_s1_II, safelog(self.A))
        else:
            A1_free_energy = np.einsum('ij,j,i', self.A, self.q_s1_II, safelog(self.q_o1)) + np.einsum('ij,j,j->', self.A, self.q_s1_II, safelog(self.q_s1_II)) - np.einsum('ij,j,ij', self.A, self.q_s1_II, safelog(self.A))
        if self.observed_o2:
            A2_free_energy = np.einsum('i,i', self.q_o2, safelog(self.q_o2)) + np.einsum('i,i', self.q_s2, safelog(self.q_s2)) - np.einsum('i,j,ij', self.q_o2, self.q_s2, safelog(self.A))
        else:
            A2_free_energy = np.einsum('ij,j,i', self.A, self.q_s2, safelog(self.q_o2)) + np.einsum('ij,j,j->', self.A, self.q_s2, safelog(self.q_s2)) - np.einsum('ij,j,ij', self.A, self.q_s2, safelog(self.A))
        C0_free_energy = np.einsum('i,i', self.q_o0, safelog(self.q_o0)) - np.einsum('i,i', self.q_o0, safelog(self.C))
        C1_free_energy = np.einsum('i,i', self.q_o1, safelog(self.q_o1)) - np.einsum('i,i', self.q_o1, safelog(self.C))
        C2_free_energy = np.einsum('i,i', self.q_o2, safelog(self.q_o2)) - np.einsum('i,i', self.q_o2, safelog(self.C))
        EQ0_free_energy = np.einsum('i,i', self.q_s0_s0_I_s0_II, safelog(self.q_s0_s0_I_s0_II))
        EQ1_free_energy = np.einsum('i,i', self.q_s1_s1_I_s1_II, safelog(self.q_s1_s1_I_s1_II))

        s0_entropy = - np.einsum('i,i', self.q_s0, safelog(self.q_s0))
        s0_I_entropy = - np.einsum('i,i', self.q_s0_I, safelog(self.q_s0_I))
        s0_II_entropy = - np.einsum('i,i', self.q_s0_II, safelog(self.q_s0_II))
        s1_entropy = - np.einsum('i,i', self.q_s1, safelog(self.q_s1))
        s1_I_entropy = - np.einsum('i,i', self.q_s1_I, safelog(self.q_s1_I))
        s1_II_entropy = - np.einsum('i,i', self.q_s1_II, safelog(self.q_s1_II))
        s2_entropy = - np.einsum('i,i', self.q_s2, safelog(self.q_s2))
        u0_entropy = - np.einsum('i,i', self.q_u0, safelog(self.q_u0))
        u1_entropy = - np.einsum('i,i', self.q_u1, safelog(self.q_u1))
        o0_entropy = - np.einsum('i,i', self.q_o0, safelog(self.q_o0))
        o1_entropy = - np.einsum('i,i', self.q_o1, safelog(self.q_o1))
        o2_entropy = - np.einsum('i,i', self.q_o2, safelog(self.q_o2))

        summed_free_energy = D_free_energy + B0_free_energy + B1_free_energy + U0_free_energy + U1_free_energy + A0_free_energy + A1_free_energy + A2_free_energy + C0_free_energy + C1_free_energy + C2_free_energy + EQ0_free_energy + EQ1_free_energy
        overcounted_entropy = s0_entropy + s0_I_entropy + s0_II_entropy + s1_entropy + s1_I_entropy + s1_II_entropy + s2_entropy + u0_entropy + u1_entropy + o0_entropy + o1_entropy + o2_entropy
        free_energy = summed_free_energy + overcounted_entropy

        return free_energy

    def observe(self, observation):
        if not self.observed_o0:
            self.q_o0[:] = 0
            self.q_o0[observation] = 1
            self.observed_o0 = True
        elif not self.observed_o1:
            self.q_o1[:] = 0
            self.q_o1[observation] = 1
            self.observed_o1 = True
        elif not self.observed_o2:
            self.q_o2[:] = 0
            self.q_o2[observation] = 1
            self.observed_o2 = True

    def act(self, action=None):
        if not self.acted_u0:
            if action == None:
                action = np.random.choice(self.number_actions, p=self.q_u0)
            self.q_u0[:] = 0
            self.q_u0[action] = 1
            self.acted_u0 = True
        elif not self.acted_u1:
            if action == None:
                action = np.random.choice(self.number_actions, p=self.q_u1)
            self.q_u1[:] = 0
            self.q_u1[action] = 1
            self.acted_u1 = True
        return action

    def print_beliefs(self):
        print("Beliefs:")

        print("\033[1mTimestep 0:\033[0m")
        print("q(s_0) = ", self.q_s0)
        print("q(o_0) = ", self.q_o0)
        print("q(u_0) = ", self.q_u0)

        print("\033[1mTimestep 1:\033[0m")
        print("q(s_1) = ", self.q_s1)
        print("q(o_1) = ", self.q_o1)
        print("q(u_1) = ", self.q_u1)

        print("\033[1mTimestep 2:\033[0m")
        print("q(s_2) = ", self.q_s2)
        print("q(o_2) = ", self.q_o2)
        print()
