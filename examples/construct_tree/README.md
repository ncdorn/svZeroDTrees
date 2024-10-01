This example will construct a simple tree from a simple svZeroDSolver config with one outlet `simple_config.json`.
The config file used to run this experiment is `construct_trees_example.json`. This is run by the command `run_from_file` in `construct_trees.py`
The outputs from this experiment are:
1. `config_w_cwss_trees.in`, a pickled file which contains the StructuredTreeBC class linked to the original config handler for the svZeroDSolver input file. 
2. `construct_trees_example.log` a log file which has information about the number of vessels in the tree. In a more complex experiment, such as a tree optimization, this would include information about the optimization and the various optimized tree resistances 

`config_w_cwss_trees.in` can be loaded into a `ConfigHandler` instance to inspect the tree and access the various `StructuredTree` class attributes. see the `ConfigHandler` and `StructuredTree` code for more information.