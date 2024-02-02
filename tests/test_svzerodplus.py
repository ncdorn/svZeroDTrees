import json
import svzerodplus
from svzerodtrees.utils import *
from svzerodsolver import runner


def test_pulmonary():
    with open('models/AS1_SU0308_r_steady/AS1_SU0308_r_steady.in') as ff:
        config = json.load(ff)
    
    result = run_svzerodplus(config)
    arr = get_branch_result(result, 'pressure_in', 50)
    print(arr)


def test_one_vessel(python=False):
    with open('tests/cases/steadyFlow_R_R.json') as ff:
    # with open('models/LPA_RPA_0d_steady/LPA_RPA_0d_steady.json') as ff:
        config = json.load(ff)

    if python:
        python_result = runner.run_from_config(config)
        print(python_result)

    solver = svzerodplus.Solver(config)
    print('made the solver')
    solver.run()
    print('ran the solver')
    result = run_svzerodplus(config)
    print(result)

def test_one_bifurc(python=False):
    with open('tests/cases/steadyFlow_bifurcationR_R1.json') as ff:
    # with open('models/LPA_RPA_0d_steady/LPA_RPA_0d_steady.json') as ff:
        config = json.load(ff)

    if python:
        python_result = runner.run_from_config(config)
        print(python_result)
    solver = svzerodplus.Solver(config)
    print('made the solver')
    solver.run()
    print('ran the solver')
    result = run_svzerodplus(config)
    print(result)

def test_two_bifurcs(python=False):
    with open('tests/cases/steadyFlow_2bifurcR_R1.json') as ff:
    # with open('models/LPA_RPA_0d_steady/LPA_RPA_0d_steady.json') as ff:
        config = json.load(ff)

    if python:
        python_result = runner.run_from_config(config)
        print(python_result)
    solver = svzerodplus.Solver(config)
    print('made the solver')
    solver.run()
    print('ran the solver')
    result = run_svzerodplus(config)
    print(result)

def test_mine(python=False):
    with open('tests/cases/simple_tree.json') as ff:
    # with open('models/LPA_RPA_0d_steady/LPA_RPA_0d_steady.json') as ff:
        config = json.load(ff)

    if python:
        python_result = runner.run_from_config(config)
        print(python_result)
    solver = svzerodplus.Solver(config)
    print('made the solver')
    solver.run()
    print('ran the solver')
    result = run_svzerodplus(config)
    print(result)


if __name__ == "__main__":
    test_one_vessel()
    print('tested one vessel')

    test_one_bifurc()
    print('tested one bifurc')

    test_two_bifurcs()
    print('tested two bifurcs')

    test_mine()
    print('tested mine')