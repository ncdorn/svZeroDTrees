from svzerodtrees.post_processing.pa_plotter import PAanalyzer
import numpy as np
import os

if __name__ == '__main__':

    os.chdir('../zerod_models/AS1_SU0308_prestent/')
    plotter = PAanalyzer.from_files('config_w_cwss_trees.in', 'experiments/AS1_cwss_adaptation/full_results.json', 'experiments/AS1_cwss_adaptation/figures/')

    # plot outflow vs. distance
    plotter.scatter_qoi_vs_distance('all', 'resistance', filename='resistance_vs_distance.png')

    # plot q_out mpostop vs distance
    plotter.scatter_qoi_vs_distance('all', 'q_out postop', filename='q_out_postop_vs_distance.png')

    # plot q_out final vs distance
    plotter.scatter_qoi_vs_distance('all', 'q_out final', filename='q_out_final_vs_distance.png')
    # plot length vs distance
    plotter.scatter_qoi_vs_distance('all', 'length', filename='length_vs_distance.png')

    # plot tree R_eq vs distance
    plotter.scatter_qoi_vs_distance('outlets', 'tree_resistance', filename='tree_R_vs_distance.png')

