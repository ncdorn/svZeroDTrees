import matplotlib.pyplot as plt
from svzerodtrees.utils import *



def plot_distal_wss(config, result):
    '''
    plot the wss at the outlets of the model

    :param config: config dict
    '''
    

    plt.hist(get_outlet_data(config, result, "wss"), bins=20)

    plt.show()


def plot_micro_wss(trees, max_d=.015):
    '''
    plot the wss in the distal vessels of the tree, below a certain diameter threshold

    :param trees: list of StructuredTreeOutlet instances
    :param max_d: maximum diameter threshold for plotting distal wss
    '''
    
    wss_list = []

    for tree in trees:
        
        def get_distal_wss(vessel):
            if vessel:
                if vessel.d < max_d:
                    wss_list.append(vessel.wss)
                get_distal_wss(vessel.left)
                get_distal_wss(vessel.right)

        get_distal_wss(tree.root)
    
    plt.hist(wss_list, bins=100)
    plt.title('Histogram of distal WSS')


def plot_outlet_flow_histogram():

    pass

