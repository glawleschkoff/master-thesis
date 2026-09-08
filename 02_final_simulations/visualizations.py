import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

def experiment_1_figure_cue():
    files = [
    'results/simulation_1_BFE.nc',
    'results/simulation_1_GFE.nc', 
    'results/simulation_1_context_BFE.nc', 
    'results/simulation_1_context_GFE.nc'   
    ]

    titles = [
        "A: BFE Agent",
        "B: GFE Agent",
        "C: Context-BFE Agent",
        "D: Context-GFE Agent"
    ]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 14), sharex=True, sharey=True)
    axes = axes.flatten() 

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    for i, (file, title) in enumerate(zip(files, titles)):
        
        xdata = xr.open_dataarray(file).load()
        xdata = xdata.loc[:, :, 1]
        df_data = xdata.T.to_pandas()
        df_data.index = df_data.index.round(1)
        df_data.columns = df_data.columns.round(1)

        sns.heatmap(
            df_data,
            ax=axes[i],
            annot=True,
            cmap="viridis",
            fmt=".3f",
            annot_kws={'size': 13}, 
            vmin=0.0,
            vmax=0.4,
            cbar=(i == 0),
            cbar_ax=cbar_ax if i == 0 else None
        )

        
        axes[i].tick_params(labelsize=12)
        axes[i].set_title(title, pad=15, fontsize=20)
        
        if i % 2 == 0:
            axes[i].set_ylabel(rf"$\alpha$", fontsize=18)
        else:
            axes[i].set_ylabel("")
            
        if i >= 2:
            axes[i].set_xlabel("r", fontsize=18)
        else:
            axes[i].set_xlabel("")

    fig.subplots_adjust(wspace=0.05, hspace=0.15, right=0.88)

    plt.savefig('figures/simulation_1_cue.pdf', format='pdf', bbox_inches='tight')
    #plt.savefig('../03_thesis/figures/simulation_1_cue.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def experiment_1_figure_arms():
    files = [
    'results/simulation_1_BFE.nc',
    'results/simulation_1_GFE.nc', 
    'results/simulation_1_context_BFE.nc', 
    'results/simulation_1_context_GFE.nc'   
    ]

    titles = [
        "A: BFE Agent",
        "B: GFE Agent",
        "C: Context-BFE Agent",
        "D: Context-GFE Agent"
    ]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 14), sharex=True, sharey=True)
    axes = axes.flatten() 

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    for i, (file, title) in enumerate(zip(files, titles)):
        
        xdata = xr.open_dataarray(file).load()
        xdata = xdata.loc[:, :, 2] + xdata.loc[:, :, 3]
        df_data = xdata.T.to_pandas()
        df_data.index = df_data.index.round(1)
        df_data.columns = df_data.columns.round(1)

        sns.heatmap(
            df_data,
            ax=axes[i],
            annot=True,
            cmap="viridis",
            fmt=".3f",
            annot_kws={'size': 13}, 
            vmin=0.35,
            vmax=0.8,
            cbar=(i == 0),
            cbar_ax=cbar_ax if i == 0 else None
        )
        
        axes[i].tick_params(labelsize=12)
        axes[i].set_title(title, pad=15, fontsize=20)
        
        if i % 2 == 0:
            axes[i].set_ylabel(rf"$\alpha$", fontsize=18)
        else:
            axes[i].set_ylabel("") 
            
        if i >= 2:
            axes[i].set_xlabel("r", fontsize=18)
        else:
            axes[i].set_xlabel("") 

    fig.subplots_adjust(wspace=0.05, hspace=0.15, right=0.88)

    plt.savefig('figures/simulation_1_arms.pdf', format='pdf', bbox_inches='tight')
    #plt.savefig('../03_thesis/figures/simulation_1_arms.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def experiment_2_figure():
    files = [
    'results/simulation_2_BFE.nc',
    'results/simulation_2_GFE.nc', 
    'results/simulation_2_context_BFE.nc', 
    'results/simulation_2_context_GFE.nc'   
    ]

    titles = [
        "A: BFE Agent",
        "B: GFE Agent",
        "C: Context-BFE Agent",
        "D: Context-GFE Agent"
    ]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 14), sharex=True, sharey=True)
    axes = axes.flatten() 

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    for i, (file, title) in enumerate(zip(files, titles)):
        
        xdata = xr.open_dataarray(file).load()
        xdata = xdata.isel(u1=[2, 3]).max(dim='u1')
        #xdata = xdata.isel(u1=1)
        
        df_data = xdata.T.to_pandas()
        df_data.index = df_data.index.round(1)
        df_data.columns = df_data.columns.round(1)

        sns.heatmap(
            df_data,
            ax=axes[i],
            annot=True,
            cmap="viridis",
            fmt=".3f",
            annot_kws={'size': 13}, 
            vmin=0.25,  
            vmax=1.0,   
            cbar=(i == 0),
            cbar_ax=cbar_ax if i == 0 else None
        )
        
        axes[i].tick_params(labelsize=12)
        axes[i].set_title(title, pad=15, fontsize=20)
        
        if i % 2 == 0:
            axes[i].set_ylabel(rf"$\alpha$", fontsize=18)
        else:
            axes[i].set_ylabel("") 
            
        if i >= 2:
            axes[i].set_xlabel("r", fontsize=18)
        else:
            axes[i].set_xlabel("") 

    fig.subplots_adjust(wspace=0.05, hspace=0.15, right=0.88)

    plt.savefig('figures/simulation_2.pdf', format='pdf', bbox_inches='tight')
    #plt.savefig('../03_thesis/figures/simulation_2.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def experiment_3_figure():
    files = [
    'results/simulation_3_BFE.nc',
    'results/simulation_3_GFE.nc', 
    'results/simulation_3_context_BFE.nc', 
    'results/simulation_3_context_GFE.nc'   
    ]

    titles = [
        "A: BFE Agent",
        "B: GFE Agent",
        "C: Context-BFE Agent",
        "D: Context-GFE Agent"
    ]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 14), sharex=True, sharey=True)
    axes = axes.flatten() 

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    for i, (file, title) in enumerate(zip(files, titles)):
        
        xdata = xr.open_dataarray(file).load()
        
        df_rewards = xdata.T.to_pandas()
        df_rewards.index = df_rewards.index.round(1)
        df_rewards.columns = df_rewards.columns.round(1)

        sns.heatmap(
            df_rewards,
            ax=axes[i],
            annot=True,
            cmap="viridis",
            fmt=".3f",
            annot_kws={'size': 13}, 
            vmin=0.25,  
            vmax=0.65,   
            cbar=(i == 0),
            cbar_ax=cbar_ax if i == 0 else None
        )
        
        axes[i].tick_params(labelsize=12)
        axes[i].set_title(title, pad=15, fontsize=20)
        
        if i % 2 == 0:
            axes[i].set_ylabel(rf"$\alpha$", fontsize=18)
        else:
            axes[i].set_ylabel("") 
            
        if i >= 2:
            axes[i].set_xlabel("r", fontsize=18)
        else:
            axes[i].set_xlabel("") 

    fig.subplots_adjust(wspace=0.05, hspace=0.15, right=0.88)

    plt.savefig('figures/simulation_3.pdf', format='pdf', bbox_inches='tight')
    #plt.savefig('../03_thesis/figures/simulation_3.pdf', format='pdf', bbox_inches='tight')
    plt.show()