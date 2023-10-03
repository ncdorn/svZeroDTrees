from svzerodtrees.post_processing.pa_plotter import PAPlotter
import numpy as np
import os

if __name__ == '__main__':

    os.chdir('../zerod_models/AS1_SU0308_prestent/')
    plotter = PAPlotter.from_file('preop_config.json', 'experiments/AS1_cwss_adaptation/full_results.json', 'experiments/AS1_cwss_adaptation/figures/')

    # plotter.plot_flow_adaptation(['lpa', 'rpa'])

    # plotter.plot_flow_adaptation(vessels='all', filename='flow_adaptation_all.png')

    # plotter.plot_flow_adaptation(vessels='outlets', filename='flow_adaptation_outlets.png')

    # plotter.plot_outlet_flow_histogram()

    plotter.build_tree_map()

    print(len(plotter.vessels))
    print([child.path for child in plotter.root.children])


