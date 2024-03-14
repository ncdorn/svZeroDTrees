from svzerodtrees.post_processing.pa_plotter import PAanalyzer
import numpy as np
import os

if __name__ == '__main__':

    os.chdir('/Users/ndorn/ndorn@stanford.edu - Google Drive/My Drive/Stanford/PhD/Simvascular/zerod_models/AS2_prestent')
    experiment = 'AS2_5mm_stent'
    plotter = PAanalyzer.from_files('preop_config.json', 'experiments/' + experiment + '/full_results.json', 'experiments/' + experiment + '/figures/')


    # plot outflow vs. distance
    # plotter.scatter_qoi_adaptation_distance('all', 'q_out')
    # plotter.scatter_qoi_adaptation_distance('outlets', 'q_out', filename='adaptation_scatter_outlets.png')

    # # plot pressure vs. distance
    # plotter.scatter_qoi_adaptation_distance('all', 'p_out')
    # plotter.scatter_qoi_adaptation_distance('outlets', 'p_out', filename='adaptation_scatter_outlets.png')

    # # plot wss vs. distance
    # plotter.scatter_qoi_adaptation_distance('all', 'wss')
    # plotter.scatter_qoi_adaptation_distance('outlets', 'wss', filename='adaptation_scatter_outlets.png')


    # plot lpa and rpa flow adaptation
    plotter.plot_lpa_rpa_adaptation()

    # plot lpa and rpa changes
    plotter.plot_lpa_rpa_diff()





