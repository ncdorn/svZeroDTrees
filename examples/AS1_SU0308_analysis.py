from svzerodtrees.post_processing.pa_plotter import PAanalyzer
import numpy as np
import os

if __name__ == '__main__':

    os.chdir('../zerod_models/AS1_SU0308_prestent/')
    plotter = PAanalyzer.from_files('preop_config.json', 'experiments/AS1_cwss_adaptation_extensive/full_results.json', 'experiments/AS1_cwss_adaptation_extensive/figures/')


    # plot outflow vs. distance
    plotter.scatter_qoi_adaptation_distance('all', 'q_out')
    plotter.scatter_qoi_adaptation_distance('outlets', 'q_out', filename='adaptation_scatter_outlets.png')

    # plot pressure vs. distance
    plotter.scatter_qoi_adaptation_distance('all', 'p_out')
    plotter.scatter_qoi_adaptation_distance('outlets', 'p_out', filename='adaptation_scatter_outlets.png')

    # plot wss vs. distance
    plotter.scatter_qoi_adaptation_distance('all', 'wss')
    plotter.scatter_qoi_adaptation_distance('outlets', 'wss', filename='adaptation_scatter_outlets.png')

    # plot flow adaptation in the lpa and rpa
    plotter.plot_flow_adaptation([plotter.lpa.branch, plotter.rpa.branch], filename='lpa_rpa_flow_adaptation.png', threshold=0.0)

    # plot lpa and rpa changes
    plotter.plot_lpa_rpa_diff()





