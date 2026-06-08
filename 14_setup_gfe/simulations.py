import numpy as np
import xarray as xr
from agent import Agent
from agent_param import generate_params
from environment import Environment
from environment_param import a, b, d

def run_simulation_1():
    c_values = np.arange(11)
    alpha_values = np.arange(0, 1.1, 0.1)
    control_values = np.arange(4)
    data = np.zeros((len(c_values), len(alpha_values), len(control_values)))
    xdata = xr.DataArray(
        data,
        coords={
            'c': c_values,
            'alpha': alpha_values,
            'u0': control_values
        },
        dims=['c', 'alpha', 'u0']
    )

    for c in c_values:
        for alpha in alpha_values:
            A, B, C, D, U = generate_params(c=c, alpha=alpha)
            agent = Agent(A, B, C, D, U)
            agent.infer(iterations=20)
            xdata.loc[c, alpha, :] = agent.q_u0
    xdata.to_netcdf('artifacts/simulation_1.nc')

def run_simulation_2():
    c_values = np.arange(11)
    alpha_values = np.arange(0, 1.1, 0.1)
    control_values = np.arange(4)
    data = np.zeros((len(c_values), len(alpha_values), len(control_values)))
    xdata = xr.DataArray(
        data,
        coords={
            'c': c_values,
            'alpha': alpha_values,
            'u1': control_values
        },
        dims=['c', 'alpha', 'u1']
    )

    for c in c_values:
        for alpha in alpha_values:
            A, B, C, D, U = generate_params(c=c, alpha=alpha)
            agent = Agent(A, B, C, D, U)
            agent.observe(0)
            agent.act(1)
            agent.observe(4)
            agent.infer(iterations=20)
            xdata.loc[c, alpha, :] = agent.q_u1
    xdata.to_netcdf('artifacts/simulation_2.nc')

def run_simulation_3():
    number_trials = 1000
    c_values = np.arange(11)
    alpha_values = np.arange(0, 1.1, 0.1)
    data = np.zeros((len(c_values), len(alpha_values)))
    xdata = xr.DataArray(
        data,
        coords={
            'c': c_values,
            'alpha': alpha_values
        },
        dims=['c', 'alpha']
    )

    for c in c_values:
        for alpha in alpha_values:
            print(c, alpha)
            observations = np.zeros((number_trials, 2))
            for trial in range(number_trials):
                A, B, C, D, U = generate_params(c=c, alpha=alpha)
                agent = Agent(A, B, C, D, U)
                environment = Environment(a, b, d)
                o0 = environment.generate_observation()
                agent.observe(o0)
                agent.infer(iterations=20)
                a0 = agent.act()
                environment.act_upon(a0)
                o1 = environment.generate_observation()
                agent.observe(o1)
                agent.infer(iterations=20)
                a1 = agent.act()
                environment.act_upon(a1)
                o2 = environment.generate_observation()

                observations[trial, 0] = o1
                observations[trial, 1] = o2

            number_rewards = np.sum(observations[:, 0] == 10) + np.sum(observations[:, 0] == 14) + np.sum(observations[:, 1] == 10) + np.sum(observations[:, 1] == 14)
            xdata.loc[c, alpha] = number_rewards / number_trials

    xdata.to_netcdf('artifacts/simulation_3.nc')