from svzerodtrees.utils import *
from svzerodtrees.post_processing.stree_visualization import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_changes_subfig(result, branches, qoi, title, ylabel, xlabel=None, ax=None):
    '''
    plot the changes in the LPA and RPA flow, pressure and wss as a grouped bar graph

    :param summary_values: summarized results dict for a given QOI, from postop.summarize_results
    :param branches: list of str containing the branches to plot
    :param qoi: str containing the data name to plot
    :param title: figure title
    :param ylabel: figure ylabel
    :param xlabel: figure xlabel
    :param ax: figure ax object
    :param condition: experimental condition name

    '''

    # intialize plot dict
    timesteps = ['preop', 'postop', 'final']

    bar_width = 1 / (len(branches) + 1)

    x = np.arange(len(timesteps))

    bar_width = 0.25
    shift = 0

    # Plotting the grouped bar chart
    for branch, qois in result.items():
        if branch in branches:
            values = [qois[qoi][timestep] for timestep in timesteps]
            offset = bar_width * shift
            ax.bar(x + offset, values, bar_width, label=branch)
            shift += 1
    

    # Set labels, title, and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([0, 1, 2], timesteps)
    ax.legend()


def plot_LPA_RPA_changes(fig_dir, results, title):
    '''
    plot LPA and RPA changes in q, p, wss as three subfigures

    :param fig_dir: path to directory to save figures
    :param results: summarized results dict
    :param title: figure title
    :param condition: experimental condition name
    
    '''
    if isinstance(results, str):
        with open(results) as ff:
            results = json.load(ff)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.subplots(1, 3)


    # plot the changes in q, p, wss in subfigures
    plot_changes_subfig(results,
                        ['lpa', 'rpa'],
                        'q_out',
                        title='outlet flowrate',
                        ylabel='q (cm3/s)',
                        ax=ax[0])
    
    plot_changes_subfig(results,
                        ['lpa', 'rpa'],
                        'p_out',
                        title='outlet pressure',
                        ylabel='p (mmHg)',
                        ax=ax[1])
    
    plot_changes_subfig(results,
                        ['lpa', 'rpa'],
                        'wss',
                        title='wss',
                        ylabel='wss (dynes/cm2)',
                        ax=ax[2])

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(fig_dir + '/' + title + '.png')


def plot_MPA_changes(fig_dir, result, title, condition='repair'):
    '''
    plot the q, p and wss changes in the MPA

    :param fig_dir: path to directory to save figures
    :param result: summarized results dict
    :param title: figure title
    :param condition: experimental condition name
    '''
    if isinstance(result, str):
        with open(result) as ff:
            result = json.load(ff)
    
    fig = plt.figure()
    ax = fig.subplots(1, 3)

    
    # plot the changes in q, p, wss in subfigures
    plot_changes_subfig(result,
                        ['mpa'],
                        'q_out',
                        title='outlet flowrate',
                        ylabel='q (cm3/s)',
                        ax=ax[0])
    
    plot_changes_subfig(result,
                        ['mpa'],
                        'p_out',
                        title='outlet pressure',
                        ylabel='p (mmHg)',
                        ax=ax[1])
    
    plot_changes_subfig(result,
                        ['mpa'],
                        'wss',
                        title='wss',
                        ylabel='wss (dynes/cm2)',
                        ax=ax[2])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(fig_dir + '/' + title + '.png')
    
