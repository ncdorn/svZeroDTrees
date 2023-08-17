from struct_tree_utils import *
from post_processing.stree_visualization import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_LPA_RPA_changes(summary_values, title, ylabel, xlabel=None, ax=None, condition=None):

    # takes in a set of values from the summary results dict
    names = list(summary_values.keys())
    values = []
    for i in range(len(summary_values['preop'])):
        vals = []
        for name in names:
            vals.append(summary_values[name][i])
        values.append(vals)
    # Create the bar positions
    bar_width = 0.35
    bar_positions = np.arange(len(names))

    # Plotting the grouped bar chart
    ax.bar(bar_positions - bar_width / 2, values[0], bar_width, label='RPA')
    ax.bar(bar_positions + bar_width / 2, values[1], bar_width, label='LPA')

    # Set labels, title, and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(bar_positions, names)
    ax.legend()

    # save the figure

    # plt.bar(range(len(q)), values, tick_label=names)
    # plt.show()


def plot_LPA_RPA_changes_subfigs(fig_dir: Path, condensed_results, title, condition=None):

    fig = plt.figure()
    ax = fig.subplots(1, 3)
    # plot the changes in q, p, wss in subfigures
    plot_LPA_RPA_changes(condensed_results[condition]['q'],
                         'outlet flowrate',
                         'q (cm3/s)',
                         ax=ax[0],
                         condition=condition)
    plot_LPA_RPA_changes(condensed_results[condition]['p'],
                         'outlet pressure',
                         'p (barye)',
                         ax=ax[1],
                         condition=condition)
    plot_LPA_RPA_changes(condensed_results[condition]['wss'],
                         'wall shear stress',
                         'tau (dyne/cm2)',
                         ax=ax[2], condition=condition)

    plt.suptitle(title + ' ' + condition)
    plt.tight_layout()
    plt.savefig(str(fig_dir / condition) + '_' + title + '.png')



if __name__ == '__main__':
    test_dir = Path("tree_tuning_test")
    dirname = 'LPA_RPA_0d_steady'
    summary_results = test_dir / dirname / '{}_summary_results.txt'.format(dirname)

    with open(summary_results) as ff:
        summ = json.load(ff)

    plot_LPA_RPA_changes(test_dir / dirname / 'figures', summ['q'], 'outlet flowrate', 'q (cm3/s)', fig_num=0)
    plot_LPA_RPA_changes(test_dir / dirname / 'figures', summ['p'], 'outlet pressure', 'p (barye)', fig_num=1)
    plot_LPA_RPA_changes(test_dir / dirname / 'figures', summ['wss'], 'wall shear stress', 'tau (dyne/cm2)', fig_num=2)

