import json
import svzerodplus
from svzerodtrees.utils import *


def test_pulmonary():
    with open('models/AS1_SU0308_r_steady/AS1_SU0308_r_steady.in') as ff:
        config = json.load(ff)
    
    result = run_svzerodplus(config)
    arr = get_result(result, 'pressure_in', 50)
    print(config["simulation_parameters"])


def test_example_tree():
    with open('tests/cases/ps_tree_example.json') as ff:
        config = json.load(ff)

    result = run_svzerodplus(config)
    print(result)


if __name__ == "__main__":
    test_example_tree()