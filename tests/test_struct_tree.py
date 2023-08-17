import json
from svzerodsolver.model.structuredtreebc import StructuredTreeOutlet
from pathlib import Path
from stree_visualization import *
import matplotlib.pyplot as plt

def test_tree_build(config):
    simparams = config["simulation_parameters"]
    test_dict = {}
    vessel_trees = []
    for vessel_config in config["vessels"]:
        if "boundary_conditions" in vessel_config:
            if "outlet" in vessel_config["boundary_conditions"]:
                outlet_stree = StructuredTreeOutlet.from_outlet_vessel(vessel_config, simparams)
                # outlet_stree.build_tree_olufsen()
                #R_mat = outlet_stree.calculate_resistance()
                vessel_tree = outlet_stree.build_tree()
                outlet_stree.optimize_tree_radius(Resistance=10000)
                # R_1 = outlet_stree.calculate_resistance()
                # outlet_stree.adapt_constant_wss(1, 2, disp=True)
                # vessel_trees.append(vessel_tree)
                print("number of vessels: " + str(len(outlet_stree.block_dict["vessels"])))
                print("third D: " + str(outlet_stree.block_dict["vessels"][2]["vessel_D"]))
                build_tree_figure(outlet_stree.block_dict, outlet_stree.root, edge_labeling=False)

    # for i, vessel_tree in enumerate(vessel_trees):
    #     build_tree_figures(outlet_stree.block_dict, vessel_tree, last_vessel=20, fig=i)

    return outlet_stree.block_dict


def run_from_file(input_file, output_file):
    """Run the svZeroDSolver from file.

    Args:
        input_file: Input file with configuration.
        output_file: Output file with configuration.
    """
    with open(input_file) as ff:
        config = json.load(ff)
        result = test_tree_build(config)
    with open(output_file, "w") as ff:
        json.dump(result, ff)

    # plot_vessels_per_generation(result, 1)
    #plot_terminal_vessel_diameter(result, 2)

    # build_tree_figures(result, last_vessel=31)
    plt.show()


if __name__ == '__main__':
    model_dir = Path("../models/struct_tree_test")
    run_from_file(model_dir / "struct_tree_test.in", model_dir / "struct_tree_test.out")