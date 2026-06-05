import numpy as np
from utils import safelog, normalize

np.set_printoptions(
    linewidth=200,
    precision=3,
    suppress=True
)


class Agent:
    def __init__(self, A, B, B_c, C, D, D_c, U, time_horizon):
        self.A = A
        self.B = B
        self.B_c = B_c
        self.C = C
        self.D = D
        self.D_c = D_c
        self.U = U
        self.time_horizon = time_horizon

        self.current_observation_timestep = 0
        self.current_action_timestep = 0

        self.number_states = B.shape[0]
        self.number_contexts = B_c.shape[0]
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
        self.q_c0 = np.ones(self.number_contexts) / self.number_contexts
        self.q_c0_I = np.ones(self.number_contexts) / self.number_contexts
        self.q_c0_II = np.ones(self.number_contexts) / self.number_contexts
        self.q_c0_III = np.ones(self.number_contexts) / self.number_contexts
        self.q_c0_IV = np.ones(self.number_contexts) / self.number_contexts
        self.q_c0_V = np.ones(self.number_contexts) / self.number_contexts
        self.q_c0_VI = np.ones(self.number_contexts) / self.number_contexts
        self.q_u0 = np.ones(self.number_actions) / self.number_actions
        self.q_u1 = np.ones(self.number_actions) / self.number_actions
        self.q_o0 = np.ones(self.number_observations) / self.number_observations
        self.q_o1 = np.ones(self.number_observations) / self.number_observations
        self.q_o2 = np.ones(self.number_observations) / self.number_observations
        self.q_o0_s0_II = np.ones((self.number_observations, self.number_states)) / (self.number_observations * self.number_states)
        self.q_o1_s1_II = np.ones((self.number_observations, self.number_states)) / (self.number_observations * self.number_states)
        self.q_o2_s2 = np.ones((self.number_observations, self.number_states)) / (self.number_observations * self.number_states)


    def free_energy(self):
        # energies
        D_energy = np.array([-np.einsum("i,i", self.state_beliefs[0], safelog(self.D))])
        A_energies = np.zeros(self.time_horizon)
        B_energies = np.zeros(self.time_horizon - 1)
        C_energies = np.zeros(self.time_horizon)
        U_energies = np.zeros(self.time_horizon - 1)
        for k in range(self.time_horizon):
            A_energies[k] = -np.einsum("ij,ij", self.A_beliefs[k], safelog(self.A))
            C_energies[k] = -np.einsum(
                "i,i", self.observation_beliefs[k], safelog(self.C)
            )
        for k in range(self.time_horizon - 1):
            B_energies[k] = -np.einsum("ijk,ijk", self.B_beliefs[k], safelog(self.B))
            U_energies[k] = -np.einsum("i,i", self.action_beliefs[k], safelog(self.U))

        # entropies
        state_entropies = np.zeros(self.time_horizon)
        observation_entropies = np.zeros(self.time_horizon)
        action_entropies = np.zeros(self.time_horizon - 1)
        A_entropies = np.zeros(self.time_horizon)
        B_entropies = np.zeros(self.time_horizon - 1)
        for k in range(self.time_horizon):
            state_entropies[k] = -np.einsum(
                "i,i", self.state_beliefs[k], safelog(self.state_beliefs[k])
            )
            observation_entropies[k] = -np.einsum(
                "i,i", self.observation_beliefs[k], safelog(self.observation_beliefs[k])
            )
            A_entropies[k] = -np.einsum(
                "ij,ij", self.A_beliefs[k], safelog(self.A_beliefs[k])
            )
        for k in range(self.time_horizon - 1):
            action_entropies[k] = -np.einsum(
                "i,i", self.action_beliefs[k], safelog(self.action_beliefs[k])
            )
            B_entropies[k] = -np.einsum(
                "ijk,ijk", self.B_beliefs[k], safelog(self.B_beliefs[k])
            )

        # free energies
        D_free_energy = D_energy - state_entropies[0]
        A_free_energies = np.zeros(self.time_horizon)
        B_free_energies = np.zeros(self.time_horizon - 1)
        C_free_energies = np.zeros(self.time_horizon)
        U_free_energies = np.zeros(self.time_horizon - 1)
        for k in range(self.time_horizon):
            A_free_energies[k] = A_energies[k] - A_entropies[k]
            C_free_energies[k] = C_energies[k] - observation_entropies[k]
        for k in range(self.time_horizon - 1):
            B_free_energies[k] = B_energies[k] - B_entropies[k]
            U_free_energies[k] = U_energies[k] - action_entropies[k]

        # total energy
        total_energy = np.zeros(1)
        total_energy += np.sum(D_energy)
        total_energy += np.sum(A_energies)
        total_energy += np.sum(B_energies)
        total_energy += np.sum(U_energies)
        total_energy += np.sum(C_energies)

        # total entropy
        total_entropy = np.zeros(1)
        total_entropy += np.sum(A_entropies)
        total_entropy += np.sum(B_entropies)
        total_entropy -= state_entropies[0]
        total_entropy -= 2 * state_entropies[1]
        total_entropy -= state_entropies[2]

        total_free_energy = total_energy - total_entropy

        return (
            total_free_energy,
            total_energy,
            total_entropy,
            A_free_energies,
            B_free_energies,
            C_free_energies,
            D_free_energy,
            U_free_energies,
            A_energies,
            B_energies,
            C_energies,
            D_energy,
            U_energies,
            A_entropies,
            B_entropies,
            state_entropies,
            observation_entropies,
            action_entropies,
        )


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


    def next_trial(self):
        self.D_c = np.einsum("ij,j->i", self.B_c, self.context_belief)

        self.current_observation_timestep = 0
        self.current_action_timestep = 0
        self.state_beliefs = np.ones((self.time_horizon, self.number_states)) * (
            1 / self.number_states
        )
        self.observation_beliefs = np.ones(
            (self.time_horizon, self.number_observations)
        ) * (1 / self.number_observations)
        self.action_beliefs = np.ones((self.time_horizon - 1, self.number_actions)) * (
            1 / self.number_actions
        )
        self.A_beliefs = np.ones(
            (
                self.time_horizon,
                self.number_observations,
                self.number_states,
                self.number_contexts,
            )
        ) * (1 / self.number_observations)
        self.B_beliefs = np.ones(
            (
                self.time_horizon - 1,
                self.number_states,
                self.number_states,
                self.number_actions,
            )
        ) * (1 / self.number_states)


    def infer(self, iterations):
        tolerance = 1e-4
        previous_free_energy = 0
        current_free_energy = float("inf")

        for iteration in range(iterations):
            mu_right_s0 = self.D
            if self.observed_o0:
                mu_up_o0 = self.q_o0
            else:
                mu_up_o0 = self.C
            f_tilde_o0_s0_II = np.exp(np.einsum('i,kji->kj', self.q_c0_II, safelog(self.A)))
            #mu_up_s0_II = normalize(np.exp(np.einsum('i,j,ikj->k', self.q_o0, self.q_c0_II, safelog(self.A))))
            mu_up_s0_II = normalize(np.einsum('ij,i->j', f_tilde_o0_s0_II, mu_up_o0))
            mu_right_s0_I = normalize(mu_right_s0 * mu_up_s0_II)
            if self.acted_u0:
                mu_down_u0 = self.q_u0
            else:
                mu_down_u0 = self.U
            mu_right_s1 = normalize(np.einsum('i,j,kij->k', mu_right_s0_I, mu_down_u0, self.B))
            if self.observed_o1:
                mu_up_o1 = self.q_o1
            else:
                mu_up_o1 = self.C
            f_tilde_o1_s1_II = np.exp(np.einsum('i,kji->kj', self.q_c0_IV, safelog(self.A)))
            #mu_up_s1_II = normalize(np.exp(np.einsum('i,j,ikj->k', self.q_o1, self.q_c0_IV, safelog(self.A))))
            mu_up_s1_II = normalize(np.einsum('ij,i->j', f_tilde_o1_s1_II, mu_up_o1))
            mu_right_s1_I = normalize(mu_right_s1 * mu_up_s1_II)
            if self.acted_u1:
                mu_down_u1 = self.q_u1
            else:
                mu_down_u1 = self.U
            mu_down_s2 = normalize(np.einsum('i,j,kij->k', mu_right_s1_I, mu_down_u1, self.B))
            f_tilde_o2_s2 = np.exp(np.einsum('i,kji->kj', self.q_c0_VI, safelog(self.A)))
            if self.observed_o2:
                mu_down_o2 = self.q_o2
            else:
                #mu_down_o2 = normalize(np.exp(np.einsum('i,j,kij->k', self.q_s2, self.q_c0_VI, safelog(self.A))))
                mu_down_o2 = normalize(np.einsum('ij,j->i', f_tilde_o2_s2, mu_down_s2))

            if self.observed_o2:
                mu_up_o2 = self.q_o2
            else:
                mu_up_o2 = self.C
            #mu_left_s2 = normalize(np.exp(np.einsum('i,j,ikj->k', self.q_o2, self.q_c0_VI, safelog(self.A))))
            mu_left_s2 = normalize(np.einsum('ij,i->j', f_tilde_o2_s2, mu_up_o2))
            if self.acted_u1:
                mu_up_u1 = self.q_u1
            else:
                mu_up_u1 = normalize(np.einsum('i,j,jik->k', mu_right_s1_I, mu_left_s2, self.B))
            mu_left_s1_I = normalize(np.einsum('i,j,jki->k', mu_down_u1, mu_left_s2, self.B))
            mu_down_s1_II = normalize(mu_right_s1 * mu_left_s1_I)
            if self.observed_o1:
                mu_down_o1 = self.q_o1
            else:
                #mu_down_o1 = normalize(np.exp(np.einsum('i,j,kij->k', self.q_s1_II, self.q_c0_IV, safelog(self.A))))
                mu_down_o1 = normalize(np.einsum('ij,j->i', f_tilde_o1_s1_II, mu_down_s1_II))
            mu_left_s1 = normalize(mu_up_s1_II * mu_left_s1_I)
            if self.acted_u0:
                mu_up_u0 = self.q_u0
            else:
                mu_up_u0 = normalize(np.einsum('i,j,jik->k', mu_right_s0_I, mu_left_s1, self.B))
            mu_left_s0_I = normalize(np.einsum('i,j,jki->k', mu_down_u0, mu_left_s1, self.B))
            mu_down_s0_II = normalize(mu_right_s0 * mu_left_s0_I)
            if self.observed_o0:
                mu_down_o0 = self.q_o0
            else:
                #mu_down_o0 = normalize(np.exp(np.einsum('i,j,kij->k', self.q_s0_II, self.q_c0_II, safelog(self.A))))
                mu_down_o0 = normalize(np.einsum('ij,j->i', f_tilde_o0_s0_II, mu_down_s0_II))
            mu_left_s0 = normalize(mu_up_s0_II * mu_left_s0_I)


            self.q_s0 = normalize(mu_left_s0 * mu_right_s0)
            self.q_s0_I = normalize(mu_left_s0_I * mu_right_s0_I)
            self.q_s0_II = normalize(mu_up_s0_II * mu_down_s0_II)
            self.q_s1 = normalize(mu_left_s1 * mu_right_s1)
            self.q_s1_I = normalize(mu_left_s1_I * mu_right_s1_I)
            self.q_s1_II = normalize(mu_up_s1_II * mu_down_s1_II)
            self.q_s2 = normalize(mu_left_s2 * mu_down_s2)
            if not self.acted_u0:
                q_u = normalize(mu_up_u0 * mu_down_u0)
                x = np.zeros(self.number_actions)
                x[np.argmax(q_u)] = 1
                #self.q_u0 = x
                self.q_u0 = q_u
            if not self.acted_u1:
                q_u = normalize(mu_up_u1 * mu_down_u1)
                x = np.zeros(self.number_actions)
                x[np.argmax(q_u)] = 1
                #self.q_u1 = x
                self.q_u1 = q_u
            if not self.observed_o0:
                self.q_o0 = normalize(mu_up_o0 * mu_down_o0)
            if not self.observed_o1:
                self.q_o1 = normalize(mu_up_o1 * mu_down_o1)
            if not self.observed_o2:
                self.q_o2 = normalize(mu_up_o2 * mu_down_o2)
            self.q_o0_s0_II = normalize(np.einsum('ij,i,j->ij', f_tilde_o0_s0_II, mu_up_o0, mu_down_s0_II))
            self.q_o1_s1_II = normalize(np.einsum('ij,i,j->ij', f_tilde_o1_s1_II, mu_up_o1, mu_down_s1_II))
            self.q_o2_s2 = normalize(np.einsum('ij,i,j->ij', f_tilde_o2_s2, mu_up_o2, mu_down_s2))



            mu_right_c0 = self.D_c
            #mu_up_c0_II = normalize(np.exp(np.einsum('i,j,ijk->k', self.q_o0, self.q_s0_II, safelog(self.A))))
            mu_up_c0_II = normalize(np.exp(np.einsum('ij,ijk->k', self.q_o0_s0_II, safelog(self.A))))
            mu_right_c0_I = normalize(mu_right_c0 * mu_up_c0_II)
            #mu_up_c0_IV = normalize(np.exp(np.einsum('i,j,ijk->k', self.q_o1, self.q_s1_II, safelog(self.A))))
            mu_up_c0_IV = normalize(np.exp(np.einsum('ij,ijk->k', self.q_o1_s1_II, safelog(self.A))))
            mu_right_c0_III = normalize(mu_right_c0_I * mu_up_c0_IV)
            #mu_up_c0_VI = normalize(np.exp(np.einsum('i,j,ijk->k', self.q_o2, self.q_s2, safelog(self.A))))
            mu_up_c0_VI = normalize(np.exp(np.einsum('ij,ijk->k', self.q_o2_s2, safelog(self.A))))
            mu_right_c0_V = normalize(mu_right_c0_III * mu_up_c0_VI)

            mu_left_c0_V = np.array([0.5, 0.5])
            mu_down_c0_VI = normalize(mu_right_c0_III * mu_left_c0_V)
            mu_left_c0_III = normalize(mu_up_c0_VI * mu_left_c0_V)
            mu_down_c0_IV = normalize(mu_right_c0_I * mu_left_c0_III)
            mu_left_c0_I = normalize(mu_up_c0_IV * mu_left_c0_III)
            mu_down_c0_II = normalize(mu_right_c0 * mu_left_c0_I)
            mu_left_c0 = normalize(mu_up_c0_II * mu_left_c0_I)


            self.q_c0 = normalize(mu_left_c0 * mu_right_c0)
            self.q_c0_I = normalize(mu_left_c0_I * mu_right_c0_I)
            self.q_c0_II = normalize(mu_up_c0_II * mu_down_c0_II)
            self.q_c0_III = normalize(mu_left_c0_III * mu_right_c0_III)
            self.q_c0_IV = normalize(mu_up_c0_IV * mu_down_c0_IV)
            self.q_c0_V = normalize(mu_left_c0_V * mu_right_c0_V)
            self.q_c0_VI = normalize(mu_up_c0_VI * mu_down_c0_VI)

            # result = self.free_energy()
            # current_free_energy = result[0]

    

    def print_beliefs(self):

        print("Beliefs:")
        print("q(c_0) = ", self.q_c0)

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
