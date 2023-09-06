# svZeroDTrees
This is a package to generate structured and morphometric trees to simulate microvascular adaptation in svZeroDPlus.

## Installing

1. Run `git clone https://github.com/ncdorn/svZeroDTrees.git`
2. In desired python environment run `pip install -e .`

To use the svZeroDPlus solver (required), it must be installed separately.

3. Install solver following directions at https://stanfordcbcl.github.io/svZeroDPlus/cpp/

## Overview
Generate structured trees at the outlet of 0d models created in SimVascular based on outflow boundary conditions to simulate microvasculature. Adapt these trees to changes in flow at the outlet based on upstream changes, such as stenosis relief.
There are currently two microvascular adaptation schemes:
* constant wall shear stress assumption (Yang et al. 2016)
* Pries and Secomb model (Pries et al. 1998)

## todo
[] document all code
[] robustify documentation
[] robustify tree and data visualization methods

## More documentation to come...

