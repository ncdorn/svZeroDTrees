import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from struct_tree_utils import *
import json
from math import trunc
import seaborn as sns
import random

def visualize_binary_tree(root,
                          node_label,
                          ax=None,
                          last_vessel=None,
                          edge_labeling=False):
    G = nx.Graph()
    edges = []  # need to figure out how to add variable edge length and labels
    vessel_ds = []
    def traverse(node, parent='outlet'):
        if node is None:
            return

        G.add_node(node.id)
        if parent is not None:
            # set up edges dict to add in edge lengths later
            edges.append((parent, node.id))
            vessel_ds.append(node.d) # create the list of vessel diameters for later labeling
            # could also do this for resistances, etc.
            G.add_edge(parent, node.id)

        traverse(node.left, node.id)
        traverse(node.right, node.id)

    traverse(root)
    edges = edges[:last_vessel]
    # edge_lengths = {edges[i]: vessel_lengths[i] for i in range(len(edges))}
    # edge_labels = {edges[i]: labels['edges'][i] for i in range(len(edges))}
    edge_labels = {edges[i]: round(vessel_ds[i], 3) for i in range(len(edges))}
    # nx.set_edge_attributes(G, edge_lengths, name='weight')

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    # shift the first two nodes to the center
    # pos[0] = (pos[0][0] + 125, pos[0][1])
    # move the outlet right above the first node
    # pos['outlet'] = (pos[0][0], pos[0][1] + 50)

    # need to add in total resistance
    nx.draw_networkx(G,
                     pos,
                     with_labels=True,
                     labels = node_label,
                     ax=ax,
                     node_color="red",
                     node_shape='s',
                     node_size=0,
                     font_size=7,
                     font_weight="bold",
                     font_color="k",
                     width=[vessel_d * 10 for vessel_d in vessel_ds],
                     edge_color='red'
                     )

    if edge_labeling:
        nx.draw_networkx_edge_labels(G,
                                     pos,
                                     edge_labels=edge_labels,
                                     ax=ax,
                                     font_size=6
                                     )


def build_tree_figure(tree_config, root, ax, last_vessel=None, edge_labeling=False, fig_dir=None, fig_name=None):
    vessel_ids = []
    node_label = {'outlet': 'outlet D = ' + str(round(tree_config["origin_d"], 3)) + '\n' +
                                      'tree D = ' + str(round(root.d, 3))}

    visualize_binary_tree(root, # visualize the tree
                          node_label,
                          ax=ax,
                          last_vessel=last_vessel,
                          edge_labeling=edge_labeling)

    ax.set_title(tree_config["name"] + '_'+ fig_name) # title the tree figure


def visualize_trees(preop_config, adapted_config, preop_roots, postop_roots, fig_dir=None, fig_name=None):
    # method to visualze all trees

    for i, vessel_config in enumerate(adapted_config["vessels"]):
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                for j, root in enumerate(preop_roots):
                    if root.name in vessel_config["tree"]["name"]:
                        print('building tree vis for ' + root.name)
                        # print(root.d, postop_roots[i].d)
                        fig, axs = plt.subplots(2)
                        # plot the preop tree visualization
                        build_tree_figure(preop_config["vessels"][i]["tree"], root, axs[0], fig_name='preop') # need to get tree from preop config
                        # # plot the postop tree visualization
                        build_tree_figure(vessel_config["tree"], postop_roots[j], ax=axs[1], fig_name='postop')
                        plt.suptitle(fig_name)
                        if fig_dir is not None: # save the figure if a directory is specified
                            fig.savefig(str(fig_dir) + '/' + vessel_config["tree"]["name"] + '_' + str(fig_name) + '_visualized.png')
                        else:
                            fig.show()


def plot_vessels_per_generation(tree_config: dict, ax=None, name=None):
    '''
    plot a bar chart for the number of vessels in each tree generation level.
    Args:
        tree_config: config dict of tree
        name: extra naming convention to add on>to the tree["name"]

    Returns:
        A bar chart of number of vessels plotted against tree generation

    '''
    gen_list = []
    for vessel in tree_config["vessels"]:
        gen_list.append(vessel["generation"])

    gen_count = []
    for i in range(max(gen_list) + 1):
        gen_count.append(gen_list.count(i))

    # bar chart comparing number of vessels per generation to generation

    bars = ax.bar(range(max(gen_list) + 1),
            gen_count,
            tick_label=range(max(gen_list) + 1),
            log=True)
    # create the bar labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',  # Text label
                    xy=(bar.get_x() + bar.get_width() / 2, height),  # Position of the label
                    xytext=(0, 3),  # Offset (to move the text above the bar)
                    textcoords="offset points",
                    ha='center', va='bottom')  # Alignment of the label
    # add in an annotation for d_out and d_min
    ax.annotate('D_out = ' + str(round(tree_config["origin_d"], 3)) + 'cm \n' + 'D_min = ' + str(tree_config["D_min"]) + 'cm',
                xy = (0, 512))
    # axis labels and title
    ax.set_xlabel('generation number')
    ax.set_ylabel('number of vessels')
    ax.set_title('Number of vessels per tree generation for ' + f'{tree_config["name"]}')


def plot_terminal_vessel_diameter(tree_config: dict, fig: int=1):
    terminal_dias = []
    terminal_gens = []
    for vessel in tree_config["vessels"]:
        if vessel["vessel_D"] < tree_config["r_min"]:
            terminal_dias.append(vessel["vessel_D"])
            terminal_gens.append(vessel["generation"])

    # scatter plot of terminal diameters
    plt.figure(fig)
    sns.swarmplot(terminal_dias)
    # axis labels and title
    plt.ylabel('terminal diameter (mm)')
    plt.title('Diameter of ' + str(len(terminal_dias)) + ' terminal vessels in a structured tree with r_min ' + str(tree_config["r_min"]))

    # plot terminal vessel diameter vs generation


