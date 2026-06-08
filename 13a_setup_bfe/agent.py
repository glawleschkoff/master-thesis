import numpy as np
from utils import safelog, normalize

np.set_printoptions(
    linewidth=200,
    precision=3,
    suppress=True,
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

    def infer(self):
        mu_right_s0 = self.D
        if self.observed_o0:
            mu_up_o0 = self.q_o0
        else:
            mu_up_o0 = self.C
        mu_up_s0_II = normalize(np.einsum('i,ij->j', mu_up_o0, self.A))
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
        mu_up_s1_II = normalize(np.einsum('i,ij->j', mu_up_o1, self.A))
        mu_right_s1_I = normalize(mu_right_s1 * mu_up_s1_II)
        if self.acted_u1:
            mu_down_u1 = self.q_u1
        else:
            mu_down_u1 = self.U
        mu_down_s2 = normalize(np.einsum('i,j,kij->k', mu_right_s1_I, mu_down_u1, self.B))
        if self.observed_o2:
            mu_down_o2 = self.q_o2
        else:
            mu_down_o2 = normalize(np.einsum('i,ji->j', mu_down_s2, self.A))

        if self.observed_o2:
            mu_up_o2 = self.q_o2
        else:
            mu_up_o2 = self.C
        mu_left_s2 = normalize(np.einsum('i,ij->j', mu_up_o2, self.A))
        if self.acted_u1:
            mu_up_u1 = self.q_u1
        else:
            mu_up_u1 = normalize(np.einsum('i,j,jik->k', mu_right_s1_I, mu_left_s2, self.B))
        mu_left_s1_I = normalize(np.einsum('i,j,jki->k', mu_down_u1, mu_left_s2, self.B))
        mu_down_s1_II = normalize(mu_right_s1 * mu_left_s1_I)
        if self.observed_o1:
            mu_down_o1 = self.q_o1
        else:
            mu_down_o1 = normalize(np.einsum('i,ji->j', mu_down_s1_II, self.A))
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
            mu_down_o0 = normalize(np.einsum('i,ji->j', mu_down_s0_II, self.A))
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