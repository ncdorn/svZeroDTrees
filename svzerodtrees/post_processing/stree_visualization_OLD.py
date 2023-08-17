import networx as nx

# Function to visualize the binary tree
def visualize_binary_tree_v1(root,
                             labels,
                             vessel_ds,
                             vessel_lengths,
                             figure_number=0,
                             edge_labeling=False):

    G = nx.Graph()
    edges = [] # need to figure out how to add variable edge length and labels

    def traverse(node, parent='outlet'):
        if node is None:
            return

        G.add_node(node.value)
        if parent is not None:
            # set up edges dict to add in edge lengths later
            edges.append((parent, node.value))
            G.add_edge(parent, node.value)

        traverse(node.left, node.value)
        traverse(node.right, node.value)
    traverse(root)
    edge_lengths = {edges[i]: vessel_lengths[i] for i in range(len(edges))}
    # edge_labels = {edges[i]: labels['edges'][i] for i in range(len(edges))}
    # edge_labels = {edges[i]: round(vessel_ds[i], 3) for i in range(len(edges))}
    nx.set_edge_attributes(G, edge_lengths, name='weight')

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    # shift the first two nodes to the center
    pos[0] = (pos[0][0] + 125, pos[0][1])
    # move the outlet right above the first node
    pos['outlet'] = (pos[0][0], pos[0][1] + 50)

    plt.figure(figure_number)

    # need to add in total resistance
    nx.draw_networkx(G,
                     pos,
                     with_labels=edge_labeling,
                     labels=labels['nodes'],
                     node_color="red",
                     node_shape='s',
                     node_size=0,
                     font_size=7,
                     font_weight="bold",
                     font_color="k",
                     width=[vessel_d * 100 for vessel_d in vessel_ds],
                     edge_color='red'
                     )

    if edge_labeling:
        nx.draw_networkx_edge_labels(G,
                                     pos,
                                     edge_labels=edge_labels,
                                     font_size=6
                                     )
