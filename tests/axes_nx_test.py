import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from stree_visualization import *
from structured_tree_simulation import *

def test_axes_nx(input_file):
    G = nx.path_graph(10)

    fig, ax = plt.subplots(2, 2)

    nx.draw_networkx(
        G,
        ax=ax[0, 0]
    )

    # make the test tree
    with open(input_file) as ff:
        zeroD_config = json.load(ff)
    
    roots = construct_trees(zeroD_config)
    label_dict = {'nodes': {'outlet': 1},
                  'edges': {}}

    visualize_binary_tree(roots[0],
                          labels = label_dict,
                          vessel_lengths=[],
                          ax=ax[1, 0])

    

    plt.show()


if __name__ == '__main__':
    input_file = 'structured_tree_tuning/models/struct_tree_test/struct_tree_test.in'
    test_axes_nx(input_file)