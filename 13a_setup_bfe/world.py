import numpy as np
from agent import Agent
from agent_param import generate_params
from environment import Environment
from environment_param import a, b, d

class World:
    def __init__(self, c=2, alpha=0.9):
        self.time_horizon = 3
        (A, B, C, D, U) = generate_params(c=c, alpha=alpha)
        self.agent = Agent(A, B, C, D, U)
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

    def get_observations(self):
        return self.observations