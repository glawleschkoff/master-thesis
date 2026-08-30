import numpy as np
import xarray as xr
import concurrent.futures
import itertools
import os

from agent_BFE import Agent as AgentBFE
from agent_Context_BFE import Agent as AgentContextBFE
from agent_Context_GFE import Agent as AgentContextGFE
from agent_GFE import Agent as AgentGFE

from agent_parameters_BFE import generate_agent_params as generate_agent_params_BFE
from agent_parameters_Context_BFE import generate_agent_params as generate_agent_params_Context_BFE
from agent_parameters_Context_GFE import generate_agent_params as generate_agent_params_Context_GFE
from agent_parameters_GFE import generate_agent_params as generate_agent_params_GFE

from environment import Environment
from environment_parameters import generate_environment_params

def run_simulation_1_BFE():
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
            A, B, C, D, U = generate_agent_params_BFE(c=c, alpha=alpha)
            agent = AgentBFE(A, B, C, D, U)
            agent.infer()
            xdata.loc[c, alpha, :] = agent.q_u0

    os.makedirs('results', exist_ok=True)
    xdata.to_netcdf('results/simulation_1_BFE.nc')

def run_simulation_2_BFE():
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
            A, B, C, D, U = generate_agent_params_BFE(c=c, alpha=alpha)
            agent = AgentBFE(A, B, C, D, U)
            agent.observe(0)
            agent.act(1)
            agent.observe(4)
            agent.infer()
            xdata.loc[c, alpha, :] = agent.q_u1

    os.makedirs('results', exist_ok=True)
    xdata.to_netcdf('results/simulation_2_BFE.nc')

def run_simulation_3_BFE():
    number_trials = 100
    c_values = np.arange(11)
    alpha_values = np.arange(0.0, 1.1, 0.1)
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
            print(f"Starte Simulation für c={c}, alpha={alpha:.1f}")
            observations = np.zeros((number_trials, 2))
            for trial in range(number_trials):
                A, B, C, D, U = generate_agent_params_BFE(c=c, alpha=alpha)
                agent = AgentBFE(A, B, C, D, U)
                a, b, d = generate_environment_params(alpha=alpha)
                environment = Environment(a, b, d)
                o0 = environment.generate_observation()
                agent.observe(o0)
                agent.infer()
                a0 = agent.act()
                environment.act_upon(a0)
                o1 = environment.generate_observation()
                agent.observe(o1)
                agent.infer()
                a1 = agent.act()
                environment.act_upon(a1)
                o2 = environment.generate_observation()

                observations[trial, 0] = o1
                observations[trial, 1] = o2

            number_rewards = np.sum(observations[:, 0] == 10) + np.sum(observations[:, 0] == 14) + np.sum(observations[:, 1] == 10) + np.sum(observations[:, 1] == 14)
            xdata.loc[c, alpha] = number_rewards / number_trials

    os.makedirs('results', exist_ok=True)
    xdata.to_netcdf('results/simulation_3_BFE.nc')


def run_simulation_1_Context_BFE():
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
            A, B, B_c, C, D, D_c, U = generate_agent_params_Context_BFE(c=c, alpha=alpha)
            agent = AgentContextBFE(A, B, B_c, C, D, D_c, U)
            agent.infer(500)
            xdata.loc[c, alpha, :] = agent.q_u0

    os.makedirs('results', exist_ok=True)
    xdata.to_netcdf('results/simulation_1_Context_BFE.nc')

def run_simulation_2_Context_BFE():
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
            A, B, B_c, C, D, D_c, U = generate_agent_params_Context_BFE(c=c, alpha=alpha)
            agent = AgentContextBFE(A, B, B_c, C, D, D_c, U)
            agent.observe(0)
            agent.act(1)
            agent.observe(4)
            agent.infer(500)
            xdata.loc[c, alpha, :] = agent.q_u1

    os.makedirs('results', exist_ok=True)
    xdata.to_netcdf('results/simulation_2_Context_BFE.nc')

def simulate_single_combination_Context_BFE(params):
    c, alpha = params
    number_trials = 100
    
    print(f"Starte Simulation für c={c}, alpha={alpha:.1f}")
    observations = np.zeros((number_trials, 2))
    
    for trial in range(number_trials):
        A, B, B_c, C, D, D_c, U = generate_agent_params_Context_BFE(c=c, alpha=alpha)
        agent = AgentContextBFE(A, B, B_c, C, D, D_c, U)
        
        a, b, d = generate_environment_params(alpha=alpha)
        environment = Environment(a, b, d)
        
        o0 = environment.generate_observation()
        agent.observe(o0)
        agent.infer(500)
        a0 = agent.act()
        
        environment.act_upon(a0)
        o1 = environment.generate_observation()
        agent.observe(o1)
        agent.infer(500)
        a1 = agent.act()
        
        environment.act_upon(a1)
        o2 = environment.generate_observation()

        observations[trial, 0] = o1
        observations[trial, 1] = o2

    number_rewards = (np.sum(observations[:, 0] == 10) + 
                      np.sum(observations[:, 0] == 14) + 
                      np.sum(observations[:, 1] == 10) + 
                      np.sum(observations[:, 1] == 14))
    
    result_value = number_rewards / number_trials
    
    return c, alpha, result_value

def run_simulation_3_Context_BFE():
    c_values = np.arange(11)
    alpha_values = np.arange(0, 1.1, 0.1)
    
    task_parameters = list(itertools.product(c_values, alpha_values))
    
    data = np.zeros((len(c_values), len(alpha_values)))
    xdata = xr.DataArray(
        data,
        coords={
            'c': c_values,
            'alpha': alpha_values
        },
        dims=['c', 'alpha']
    )

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(simulate_single_combination_Context_BFE, task_parameters))

    for c, alpha, value in results:
        xdata.loc[c, alpha] = value

    os.makedirs('results', exist_ok=True)
    xdata.to_netcdf('results/simulation_3_Context_BFE.nc')


def run_simulation_1_Context_GFE():
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
            A, B, B_c, C, D, D_c, U = generate_agent_params_Context_GFE(c=c, alpha=alpha)
            agent = AgentContextGFE(A, B, B_c, C, D, D_c, U)
            agent.infer(100)
            xdata.loc[c, alpha, :] = agent.q_u0
    os.makedirs('results', exist_ok=True)
    xdata.to_netcdf('results/simulation_1_Context_GFE.nc')

def run_simulation_2_Context_GFE():
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
            A, B, B_c, C, D, D_c, U = generate_agent_params_Context_GFE(c=c, alpha=alpha)
            agent = AgentContextGFE(A, B, B_c, C, D, D_c, U)
            agent.observe(0)
            agent.act(1)
            agent.observe(4)
            agent.infer(100)
            xdata.loc[c, alpha, :] = agent.q_u1

    os.makedirs('results', exist_ok=True)
    xdata.to_netcdf('results/simulation_2_Context_GFE.nc')

def simulate_single_combination_Context_GFE(params):
    c, alpha = params
    number_trials = 100
    
    print(f"Starte Simulation für c={c}, alpha={alpha:.1f}")
    observations = np.zeros((number_trials, 2))
    
    for trial in range(number_trials):
        A, B, B_c, C, D, D_c, U = generate_agent_params_Context_GFE(c=c, alpha=alpha)
        agent = AgentContextGFE(A, B, B_c, C, D, D_c, U)
        
        a, b, d = generate_environment_params(alpha=alpha)
        environment = Environment(a, b, d)
        
        o0 = environment.generate_observation()
        agent.observe(o0)
        agent.infer(100)
        a0 = agent.act()
        
        environment.act_upon(a0)
        o1 = environment.generate_observation()
        agent.observe(o1)
        agent.infer(100)
        a1 = agent.act()
        
        environment.act_upon(a1)
        o2 = environment.generate_observation()

        observations[trial, 0] = o1
        observations[trial, 1] = o2

    number_rewards = (np.sum(observations[:, 0] == 10) + 
                      np.sum(observations[:, 0] == 14) + 
                      np.sum(observations[:, 1] == 10) + 
                      np.sum(observations[:, 1] == 14))
    
    result_value = number_rewards / number_trials
    
    return c, alpha, result_value

def run_simulation_3_Context_GFE():
    c_values = np.arange(11)
    alpha_values = np.arange(0, 1.1, 0.1)
    
    task_parameters = list(itertools.product(c_values, alpha_values))
    
    data = np.zeros((len(c_values), len(alpha_values)))
    xdata = xr.DataArray(
        data,
        coords={
            'c': c_values,
            'alpha': alpha_values
        },
        dims=['c', 'alpha']
    )

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(simulate_single_combination_Context_GFE, task_parameters))

    for c, alpha, value in results:
        xdata.loc[c, alpha] = value

    os.makedirs('results', exist_ok=True)
    xdata.to_netcdf('results/simulation_3_Context_GFE.nc')


def run_simulation_1_GFE():
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
            A, B, C, D, U = generate_agent_params_GFE(c=c, alpha=alpha)
            agent = AgentGFE(A, B, C, D, U)
            agent.infer(20)
            xdata.loc[c, alpha, :] = agent.q_u0

    os.makedirs('results', exist_ok=True)
    xdata.to_netcdf('results/simulation_1_GFE.nc')

def run_simulation_2_GFE():
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
            A, B, C, D, U = generate_agent_params_GFE(c=c, alpha=alpha)
            agent = AgentGFE(A, B, C, D, U)
            agent.observe(0)
            agent.act(1)
            agent.observe(4)
            agent.infer(20)
            xdata.loc[c, alpha, :] = agent.q_u1

    os.makedirs('results', exist_ok=True)
    xdata.to_netcdf('results/simulation_2_GFE.nc')

def simulate_single_combination_GFE(params):
    c, alpha = params
    number_trials = 100
    
    print(f"Starte Simulation für c={c}, alpha={alpha:.1f}")
    observations = np.zeros((number_trials, 2))
    
    for trial in range(number_trials):
        A, B, C, D, U = generate_agent_params_GFE(c=c, alpha=alpha)
        agent = AgentGFE(A, B, C, D, U)
        
        a, b, d = generate_environment_params(alpha=alpha)
        environment = Environment(a, b, d)
        
        o0 = environment.generate_observation()
        agent.observe(o0)
        agent.infer(20)
        a0 = agent.act()
        
        environment.act_upon(a0)
        o1 = environment.generate_observation()
        agent.observe(o1)
        agent.infer(20)
        a1 = agent.act()
        
        environment.act_upon(a1)
        o2 = environment.generate_observation()

        observations[trial, 0] = o1
        observations[trial, 1] = o2

    number_rewards = (np.sum(observations[:, 0] == 10) + 
                      np.sum(observations[:, 0] == 14) + 
                      np.sum(observations[:, 1] == 10) + 
                      np.sum(observations[:, 1] == 14))
    
    result_value = number_rewards / number_trials
    
    return c, alpha, result_value

def run_simulation_3_GFE():
    c_values = np.arange(11)
    alpha_values = np.arange(0, 1.1, 0.1)
    
    task_parameters = list(itertools.product(c_values, alpha_values))
    
    data = np.zeros((len(c_values), len(alpha_values)))
    xdata = xr.DataArray(
        data,
        coords={
            'c': c_values,
            'alpha': alpha_values
        },
        dims=['c', 'alpha']
    )

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(simulate_single_combination_GFE, task_parameters))

    for c, alpha, value in results:
        xdata.loc[c, alpha] = value

    os.makedirs('results', exist_ok=True)
    xdata.to_netcdf('results/simulation_3_GFE.nc')