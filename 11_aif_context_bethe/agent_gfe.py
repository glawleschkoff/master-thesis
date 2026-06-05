import numpy as np
from utils import optimize, safelog, savedivide

np.set_printoptions(
    linewidth=200,  # Längere Zeilen
    precision=3,  # Nur 3 Nachkommastellen anzeigen
    suppress=True,  # Verhindert die wissenschaftliche Notation (z.B. 1e-05 wird zu 0.000)
)


class GFEAgent:
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
        self.state_beliefs = np.ones((self.time_horizon, self.number_states)) * (
            1 / self.number_states
        )
        self.context_belief = np.ones(self.number_contexts) * (1 / self.number_contexts)
        self.context_belief[:] = self.D_c
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
        self.observation_beliefs[self.current_observation_timestep, :] = 0
        self.observation_beliefs[self.current_observation_timestep, observation] = 1
        self.current_observation_timestep += 1

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

    def infer(self):
        tolerance = 1e-4
        previous_free_energy = 0
        current_free_energy = float("inf")

        # messages for forward sweep
        rightward_state_messages = (
            np.ones((self.time_horizon, self.number_states)) / self.number_states
        )
        rightward_context_message = self.D_c
        upward_state_messages = (
            np.ones((self.time_horizon, self.number_states)) / self.number_states
        )
        downward_action_messages = (
            np.ones((self.time_horizon - 1, self.number_actions)) / self.number_actions
        )

        # messages for backward sweep
        upward_action_messages = (
            np.ones((self.time_horizon - 1, self.number_actions)) / self.number_actions
        )
        leftward_state_messages = (
            np.ones((self.time_horizon - 1, self.number_states)) / self.number_states
        )

        upward_context_messages = (
            np.ones((self.time_horizon, self.number_contexts)) / self.number_contexts
        )

        state_context_beliefs = np.ones((3, 4, 2)) / 8

        # while (abs(current_free_energy - previous_free_energy) > tolerance):
        for iteration in range(15):
            print('Iteration', iteration)
            previous_free_energy = current_free_energy

            # forward sweep (0 to T-1)
            for k in range(self.time_horizon):
                if k == 0:
                    rightward_state_messages[0] = self.D
                else:
                    state_message = np.einsum(
                        "ijk,j,j,k->i",
                        self.B,
                        rightward_state_messages[k - 1],
                        upward_state_messages[k - 1],
                        downward_action_messages[k - 1],
                    )
                    rightward_state_messages[k] = state_message / np.sum(state_message)

                if k < self.time_horizon - 1:
                    if k < self.current_action_timestep:
                        downward_action_messages[k] = self.action_beliefs[k]
                    else:
                        downward_action_messages[k] = self.U

            # backward sweep (T-2 to 0)
            for k in reversed(range(self.time_horizon - 1)):
                if k == self.time_horizon - 2:
                    action_message = np.einsum(
                        "ijk,j,j,i->k",
                        self.B,
                        rightward_state_messages[k],
                        upward_state_messages[k],
                        upward_state_messages[k + 1],
                    )
                    upward_action_messages[k] = action_message / np.sum(action_message)
                    state_message = np.einsum(
                        "ijk,k,i->j",
                        self.B,
                        downward_action_messages[k],
                        upward_state_messages[k + 1],
                    )
                    leftward_state_messages[k] = state_message / np.sum(state_message)
                else:
                    action_message = np.einsum(
                        "ijk,j,j,i,i->k",
                        self.B,
                        rightward_state_messages[k],
                        upward_state_messages[k],
                        upward_state_messages[k + 1],
                        leftward_state_messages[k + 1],
                    )
                    upward_action_messages[k] = action_message / np.sum(action_message)
                    state_message = np.einsum(
                        "ijk,k,i,i->j",
                        self.B,
                        downward_action_messages[k],
                        upward_state_messages[k + 1],
                        leftward_state_messages[k + 1],
                    )
                    leftward_state_messages[k] = state_message / np.sum(state_message)

            for k in range(self.time_horizon):

                ##### OBSERVED #####
                if self.current_observation_timestep > k:

                    ### State Messages ###
                    index_list = [0, 1, 2]
                    index_list.remove(k)
                    state_message = np.einsum(
                        "jk,k,k,k->j",
                        np.exp(
                            np.einsum(
                                "ijk,i->jk",
                                safelog(self.A),
                                self.observation_beliefs[k],
                            )
                        ),
                        rightward_context_message,
                        upward_context_messages[index_list[0]],
                        upward_context_messages[index_list[1]],
                    )
                    upward_state_messages[k] = state_message / np.sum(state_message)

                    ### Context Messages ###
                    # Timestep 2 #
                    if k == self.time_horizon - 1:
                        context_message = np.einsum(
                            "jk,j->k",
                            np.exp(
                                np.einsum(
                                    "ijk,i->jk",
                                    safelog(self.A),
                                    self.observation_beliefs[k],
                                )
                            ),
                            rightward_state_messages[k],
                        )
                        upward_context_messages[k] = context_message / np.sum(
                            context_message
                        )

                    # Timestep 0 and 1 #
                    else:
                        context_message = np.einsum(
                            "jk,j,j->k",
                            np.exp(
                                np.einsum(
                                    "ijk,i->jk",
                                    safelog(self.A),
                                    self.observation_beliefs[k],
                                )
                            ),
                            rightward_state_messages[k],
                            leftward_state_messages[k],
                        )
                        upward_context_messages[k] = context_message / np.sum(
                            context_message
                        )

                ##### UNOBSERVED #####
                else:

                    # TIMESTEP 2 #
                    if k == self.time_horizon - 1:
                        q_s_c = optimize(
                            self.A,
                            self.C,
                            state_context_beliefs[k],
                            rightward_context_message,
                            upward_context_messages,
                            rightward_state_messages[k],
                        )
                        print('k', k)
                        
                        print("state_context_belief_2", state_context_beliefs[2])
                        print("rightward_context_message", rightward_context_message)
                        print("upward_context_messages_0", upward_context_messages[0])
                        print("upward_context_messages_1", upward_context_messages[1])
                        print("upward_context_messages_2", upward_context_messages[2])
                        print("rightward_state_message_2", rightward_state_messages[2])
                        print('=> numerator', np.einsum("ij->j", q_s_c))
                        state_context_beliefs[k] = q_s_c

                        ### State Messages ###
                        numerator = np.einsum("jk->j", q_s_c)
                        denominator = rightward_state_messages[k]
                        # if np.allclose(
                        #     numerator / np.sum(numerator),
                        #     denominator / np.sum(denominator),
                        #     atol=1e-8,
                        # ):
                        #     state_message = np.array([0.5, 0.5, 0.5, 0.5])
                        # else:
                        #     state_message = savedivide(numerator, denominator)
                        state_message = savedivide(numerator, denominator)
                        upward_state_messages[k] = state_message / np.sum(state_message)

                        ### Context Messages ###
                        numerator = np.einsum("jk->k", q_s_c)
                        denominator = (
                            rightward_context_message
                            * upward_context_messages[0]
                            * upward_context_messages[1]
                        )
                        # if np.allclose(numerator / np.sum(numerator),
                        #                denominator / np.sum(denominator),
                        #                atol=1e-8):
                        #     context_message = np.array([0.5, 0.5])
                        # else:
                        #     context_message = savedivide(numerator, denominator)
                        context_message = savedivide(numerator, denominator)
                        upward_context_messages[k] = context_message / np.sum(
                            context_message
                        )
                        print("rightward_context_message", rightward_context_message)
                        print("upward_context_messages_0", upward_context_messages[0])
                        print("upward_context_messages_1", upward_context_messages[1])
                        print(
                            "=> upward_context_message_2", upward_context_messages[k]
                        )
                        print()

                    # TIMESTEP 0 and 1 #
                    else:
                        q_s_c = optimize(
                            self.A,
                            self.C,
                            state_context_beliefs[k],
                            rightward_context_message,
                            upward_context_messages,
                            rightward_state_messages[k],
                            leftward_state_messages[k],
                        )
                        if k == 0:
                            print('k', k)
                            print("state_context_belief_0", state_context_beliefs[0])
                            print("rightward_context_message", rightward_context_message)
                            print("upward_context_messages_0", upward_context_messages[0])
                            print("upward_context_messages_1", upward_context_messages[1])
                            print("upward_context_messages_2", upward_context_messages[2])
                            print("rightward_state_message_0", rightward_state_messages[0])
                            print("leftward_state_message_0", leftward_state_messages[0])
                            print('=> numerator', np.einsum("ij->j", q_s_c))
                        if k == 1:
                            print('k', k)
                            print("state_context_belief_1", state_context_beliefs[1])
                            print("rightward_context_message", rightward_context_message)
                            print("upward_context_messages_0", upward_context_messages[0])
                            print("upward_context_messages_1", upward_context_messages[1])
                            print("upward_context_messages_2", upward_context_messages[2])
                            print("rightward_state_message_1", rightward_state_messages[1])
                            print("leftward_state_message_1", leftward_state_messages[1])
                            print('=> numerator', np.einsum("ij->j", q_s_c))
                            
                        
                        state_context_beliefs[k] = q_s_c

                        ### State Messages ###
                        numerator = np.einsum("jk->j", q_s_c)
                        denominator = (
                            rightward_state_messages[k] * leftward_state_messages[k]
                        )
                        state_message = savedivide(numerator, denominator)
                        upward_state_messages[k] = state_message / np.sum(state_message)

                        ### Context Messages ###

                        index_list = [0, 1, 2]
                        index_list.remove(k)
                        numerator = np.einsum("jk->k", q_s_c)
                        denominator = (
                            rightward_context_message
                            * upward_context_messages[index_list[0]]
                            * upward_context_messages[index_list[1]]
                        )
                        context_message = savedivide(numerator, denominator)
                        upward_context_messages[k] = context_message / np.sum(
                            context_message
                        )
                        if k == 0:
                            print("rightward_context_message", rightward_context_message)
                            print("upward_context_messages_1", upward_context_messages[1])
                            print("upward_context_messages_2", upward_context_messages[2])
                            print(
                                "=> upward_context_message_0", upward_context_messages[k]
                                )
                            print()
                        if k == 1:
                            print("rightward_context_message", rightward_context_message)
                            print("upward_context_messages_0", upward_context_messages[0])
                            print("upward_context_messages_2", upward_context_messages[2])
                            print(
                                "=> upward_context_message_1", upward_context_messages[k]
                            )
                            print()

            # belief updating
            for k in range(self.time_horizon):
                if k == self.time_horizon - 1:
                    self.state_beliefs[k] = (
                        rightward_state_messages[k] * upward_state_messages[k]
                    ) / np.sum(rightward_state_messages[k] * upward_state_messages[k])
                else:
                    self.state_beliefs[k] = (
                        rightward_state_messages[k]
                        * upward_state_messages[k]
                        * leftward_state_messages[k]
                    ) / np.sum(
                        rightward_state_messages[k]
                        * upward_state_messages[k]
                        * leftward_state_messages[k]
                    )
            self.context_belief = (
                rightward_context_message
                * upward_context_messages[0]
                * upward_context_messages[1]
                * upward_context_messages[2]
            ) / np.sum(
                rightward_context_message
                * upward_context_messages[0]
                * upward_context_messages[1]
                * upward_context_messages[2]
            )
            for k in range(self.current_observation_timestep, self.time_horizon):
                self.observation_beliefs[k] = np.einsum(
                    "ijk,j,k->i", self.A, self.state_beliefs[k], self.context_belief
                )
            for k in range(self.current_action_timestep, self.time_horizon - 1):
                action_belief = (
                    downward_action_messages[k] * upward_action_messages[k]
                ) / np.sum(downward_action_messages[k] * upward_action_messages[k])
                self.action_beliefs[k] = action_belief
                # self.action_beliefs[k, :] = 0
                # self.action_beliefs[k, np.argmax(action_belief)] = 1

            for k in range(self.time_horizon):
                self.A_beliefs[k, :, :] = np.einsum(
                    "ijk,j,k->ijk", self.A, self.state_beliefs[k], self.context_belief
                )
            belief_0 = np.einsum(
                "ijk,j,j,k,i,i->ijk",
                self.B,
                rightward_state_messages[0],
                upward_state_messages[0],
                downward_action_messages[0],
                upward_state_messages[1],
                leftward_state_messages[1],
            )
            self.B_beliefs[0, :, :, :] = belief_0 / np.sum(belief_0)
            belief_1 = np.einsum(
                "ijk,j,j,k,i->ijk",
                self.B,
                rightward_state_messages[1],
                upward_state_messages[1],
                downward_action_messages[1],
                upward_state_messages[2],
            )
            self.B_beliefs[1, :, :, :] = belief_1 / np.sum(belief_1)

            # result = self.free_energy()
            # current_free_energy = result[0]

    def act(self, action=None):
        if action == None:
            current_action_belief = self.action_beliefs[self.current_action_timestep]
            action = np.random.choice(self.number_actions, p=current_action_belief)
            self.action_beliefs[self.current_action_timestep, :] = 0
            self.action_beliefs[self.current_action_timestep, action] = 1
        else:
            self.action_beliefs[self.current_action_timestep, :] = 0
            self.action_beliefs[self.current_action_timestep, action] = 1
        self.current_action_timestep += 1
        return action

    def print_beliefs(self):

        print("Beliefs:")
        print("q(c_0) =", self.context_belief)

        print("\033[1mTimestep 0:\033[0m")
        print("q(s_0) = ", self.state_beliefs[0])
        print("q(o_0) = ", self.observation_beliefs[0])
        print("q(u_0) = ", self.action_beliefs[0])

        print("\033[1mTimestep 1:\033[0m")
        print("q(s_1) = ", self.state_beliefs[1])
        print("q(o_1) = ", self.observation_beliefs[1])
        print("q(u_1) = ", self.action_beliefs[1])

        print("\033[1mTimestep 2:\033[0m")
        print("q(s_2) = ", self.state_beliefs[2])
        print("q(o_2) = ", self.observation_beliefs[2])
        print()
