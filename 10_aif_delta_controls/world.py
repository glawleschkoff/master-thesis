import numpy as np
from agent_gfe import GFEAgent
from agent_bfe import BFEAgent
from agent_gfe_param import generate_gfe_params
from agent_bfe_param import generate_bfe_params
from environment import Environment
from environment_param import a, b, d

class World:
    def __init__(self, agent_type, time_horizon=3, c=2, alpha=0.9):
        self.time_horizon = time_horizon
        if agent_type == 'gfe':
            (A, B, C, D, U) = generate_gfe_params(c=c, alpha=alpha)
            self.agent = GFEAgent(A, B, C, D, U, time_horizon)
        elif agent_type == 'bfe':
            (A, B, C, D, U) = generate_bfe_params(c=c, alpha=alpha)
            self.agent = BFEAgent(A, B, C, D, U, time_horizon)
        self.environment = Environment(a, b, d)
        self.observations = np.zeros(self.time_horizon)
        self.time_step = 0

    def run(self):
        for k in range(self.time_horizon - 1):
            observation = self.environment.generate_observation()
            self.observations[k] = observation
            self.agent.observe(observation)
            self.agent.infer()
            action = self.agent.act()
            self.environment.act_upon(action)
        observation = self.environment.generate_observation()
        self.observations[self.time_horizon - 1] = observation

    def dashboard_get_beliefs(self):
        self.agent.infer()
        (total_free_energy, total_energy, total_entropy, A_free_energies, B_free_energies, C_free_energies, D_free_energy, U_free_energies,
                A_energies, B_energies, C_energies, D_energy, U_energies,
                A_entropies, B_entropies, state_entropies, observation_entropies, action_entropies) = self.agent.free_energy()
        return (self.agent.state_beliefs, self.agent.action_beliefs, self.agent.observation_beliefs, total_free_energy, total_energy, total_entropy, A_free_energies, B_free_energies, C_free_energies, D_free_energy, U_free_energies,
                A_energies, B_energies, C_energies, D_energy, U_energies,
                A_entropies, B_entropies, state_entropies, observation_entropies, action_entropies)
    
    def dashboard_infer(self):
        #self.agent.free_energy()
        self.agent.infer()
        self.agent.print_beliefs()
        (total_free_energy, total_energy, total_entropy, A_free_energies, B_free_energies, C_free_energies, D_free_energy, U_free_energies,
                A_energies, B_energies, C_energies, D_energy, U_energies,
                A_entropies, B_entropies, state_entropies, observation_entropies, action_entropies) = self.agent.free_energy()
        print('A Energies: ', A_energies)
        print('B Energies: ', B_energies)
        print('C Energies: ', C_energies)
        print('D Energy: ', D_energy)
        print('U Energies: ', U_energies)
        print()
        print('A Entropies: ', A_entropies)
        print('B Entropies: ', B_entropies)
        print('State Entropies: ', state_entropies)
        print('Observation Entropies: ', observation_entropies)
        print('Action Entropies: ', action_entropies)
        print()
        print('A Free Energies: ', A_free_energies)
        print('B Free Energies: ', B_free_energies)
        print('C Free Energies: ', C_free_energies)
        print('D Free Energy: ', D_free_energy)
        print('U Free Energies: ', U_free_energies)
        print()
        print('Total Energy: ', total_energy)
        print('Total Entropy: ', total_entropy)
        print('Total Free Energy: ', total_free_energy)
        print()
        return (self.agent.state_beliefs, self.agent.action_beliefs, self.agent.observation_beliefs, total_free_energy, total_energy, total_entropy, A_free_energies, B_free_energies, C_free_energies, D_free_energy, U_free_energies,
                A_energies, B_energies, C_energies, D_energy, U_energies,
                A_entropies, B_entropies, state_entropies, observation_entropies, action_entropies)
    
    def dashboard_observe(self):
        observation = self.environment.generate_observation()
        #self.observations[self.time_step] = observation
        self.agent.observe(observation)
        self.agent.print_beliefs()
        (total_free_energy, total_energy, total_entropy, A_free_energies, B_free_energies, C_free_energies, D_free_energy, U_free_energies,
                A_energies, B_energies, C_energies, D_energy, U_energies,
                A_entropies, B_entropies, state_entropies, observation_entropies, action_entropies) = self.agent.free_energy()
        print('A Energies: ', A_energies)
        print('B Energies: ', B_energies)
        print('C Energies: ', C_energies)
        print('D Energy: ', D_energy)
        print('U Energies: ', U_energies)
        print()
        print('A Entropies: ', A_entropies)
        print('B Entropies: ', B_entropies)
        print('State Entropies: ', state_entropies)
        print('Observation Entropies: ', observation_entropies)
        print('Action Entropies: ', action_entropies)
        print()
        print('A Free Energies: ', A_free_energies)
        print('B Free Energies: ', B_free_energies)
        print('C Free Energies: ', C_free_energies)
        print('D Free Energy: ', D_free_energy)
        print('U Free Energies: ', U_free_energies)
        print()
        print('Total Energy: ', total_energy)
        print('Total Entropy: ', total_entropy)
        print('Total Free Energy: ', total_free_energy)
        print()
        return (self.agent.state_beliefs, self.agent.action_beliefs, self.agent.observation_beliefs, total_free_energy, total_energy, total_entropy, A_free_energies, B_free_energies, C_free_energies, D_free_energy, U_free_energies,
                A_energies, B_energies, C_energies, D_energy, U_energies,
                A_entropies, B_entropies, state_entropies, observation_entropies, action_entropies)

    def dashboard_act(self):
        action = self.agent.act()
        self.environment.act_upon(action)
        self.time_step += 1
        self.agent.print_beliefs()
        (total_free_energy, total_energy, total_entropy, A_free_energies, B_free_energies, C_free_energies, D_free_energy, U_free_energies,
                A_energies, B_energies, C_energies, D_energy, U_energies,
                A_entropies, B_entropies, state_entropies, observation_entropies, action_entropies) = self.agent.free_energy()
        return (self.agent.state_beliefs, self.agent.action_beliefs, self.agent.observation_beliefs, total_free_energy, total_energy, total_entropy, A_free_energies, B_free_energies, C_free_energies, D_free_energy, U_free_energies,
                A_energies, B_energies, C_energies, D_energy, U_energies,
                A_entropies, B_entropies, state_entropies, observation_entropies, action_entropies)


    def get_observations(self):
        return self.observations