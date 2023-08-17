from stree_visualization import *
from pathlib import Path

if __name__ == '__main__':
    test_dir = Path("../models")
    dirname = 'LPA_RPA_0d_steady'
    filepath = test_dir / dirname / "adapted_config.in"

    visualize_trees(filepath)
