import os
import numpy as np
import pandas as pd
import scipy.signal
from svzerodtrees.structuredtree import StructuredTree
from svzerodtrees.inflow import Inflow
import matplotlib.pyplot as plt
from svzerodtrees.utils import *
from scipy.optimize import minimize
from svzerodtrees.adaptation import *
from svzerodtrees._config_handler import ConfigHandler, SimParams
import scipy
import json
import pysvzerod

def compare_to_rcr():
    '''
    compare the structured tree result to a simple rcr
    '''

    # take the inflow and get the period
    with open('tests/cases/pa_unsteady/inflow.flow') as ff:
        inflow_raw = pd.read_csv(ff, delimiter=' ', header=None, names=['t', 'q'])
    
    inflow_raw['q'] = inflow_raw['q'] * -1

    # inflow_raw = pd.DataFrame({'q': [50.0, 50.0], 't': [0.0, 1.0]})

    # make some simulation parameters
    simparams = SimParams({
        'number_of_time_pts_per_cardiac_cycle': 1000
    })

    # initialize the structured tree
    tree = StructuredTree(name='test_impedance', Q_in=inflow_raw['q'], time=inflow_raw['t'], simparams=simparams)

    # build the tree
    tree.build_tree(initial_d=0.5, d_min=.01)

    # compute the impedance in frequency domain
    Z_om = tree.compute_olufsen_impedance()

    # convert flow to frequency domain
    Q_t = scipy.signal.resample(inflow_raw['q'], len(Z_om))
    t = np.linspace(0, inflow_raw['t'].iloc[-1], len(Q_t))
    Q_om = np.fft.fft(Q_t)

    # convert pressure to frequency domain
    P_om = Z_om * Q_om

    # P_om[0] = 50 * 1333.2

    P_t = np.fft.ifft(P_om)

    Z_t = np.fft.ifft(Z_om)

    p_conv = np.convolve(Z_t, Q_t)


    ### compute the rcr result
    with open('tests/cases/olufsen_impedance/rcr_comparison.json') as ff:
        config = json.load(ff)

    result_df = pysvzerod.simulate(config)

    fig, axs = plt.subplots(2)
    axs[0].plot(t, P_t / 1333.2, label='structured tree')
    axs[0].plot(t, p_conv[:len(t)] / 1333.2, '--', label='structured tree conv')
    axs[0].plot(result_df['time'], result_df['pressure_in'] / 1333.2, label='rcr')
    axs[0].set_ylabel('Pressure (mmHg)')
    axs[0].legend(loc='upper right')
    fig.suptitle(f'structured tree with {tree.count_vessels()} vessels, d_min = {tree.d_min} cm')

    axs[1].plot(t, Q_t)
    axs[1].set_ylabel('Flow (cm3/s)')
    axs[1].set_title('Inflow')

    plt.savefig('tests/cases/olufsen_impedance/test_tree_const_visc.png')

def get_tree_pressure(initial_d=1.0, d_min=0.05, n_tsteps=8192, inflow_path='tests/cases/pa_unsteady/inflow.flow'):
    '''
    check the pressure of a structrued tree in response to an arbitrary inflow
    '''

    # take the inflow and get the period
    inflow = Inflow.periodic(path='tests/cases/olufsen_impedance/flow_in2.dat', t_per=0.750732422, n_periods=3)

    inflow.plot()

    # inflow_raw = pd.DataFrame({'q': [50.0, 50.0], 't': [0.0, 1.0]})

    # make some simulation parameters
    simparams = SimParams({
        'number_of_time_pts_per_cardiac_cycle': n_tsteps
    })

    q_period, t_period = inflow.period()
    # initialize the structured tree
    tree = StructuredTree(name='test_impedance', Q_in=q_period, time=t_period, simparams=simparams)

    # build the tree
    tree.build_tree(initial_d=initial_d, d_min=d_min)

    # compute the impedance in frequency domain
    Z_om, omega = tree.compute_olufsen_impedance()

    # convert flow to frequency domain
    Q_t = scipy.signal.resample(inflow.q, len(Z_om))
    t = np.linspace(0, inflow.t[-1], len(Q_t))
    Q_om = np.fft.fft(Q_t)

    # convert pressure to frequency domain
    P_om = Z_om * Q_om

    # P_om[0] = 50 * 1333.2

    P_t = np.fft.ifft(P_om)

    Z_t = np.fft.ifft(Z_om)

    P_conv = np.convolve(Z_t, Q_t)

    n_vessels = tree.count_vessels()

    return t, P_t, Z_t, P_conv[:len(t)], n_vessels


def tree_pressure_subfig(ax, t, P_t, P_conv, title):
    '''
    plot p_t and p_conv for a structured tree
    '''

    ax.plot(t, P_t / 1333.2, label='pressure')
    ax.plot(t, P_conv, '--', label='pressure conv')
    ax.set_ylabel('Pressure (mmHg)')
    ax.set_xlabel('time (s)')
    ax.legend(loc='upper right')
    ax.set_title(title)

def tree_pres_flow_fig(t, inflow, P_t, P_conv, title, figpath=None, show=False):
    '''
    plot P_t and Q_t for a structured tree
    '''

    fig, axs = plt.subplots(2,2)
    
    tree_pressure_subfig(axs[0, 0], t, P_t, P_conv, 'pressure')

    inflow.plot(axs[0, 1])

    axs[1, 0].plot(inflow.q, P_t / 1333.2, label='pressure')
    axs[1, 0].plot(inflow.q, P_conv, '--', label='pressure conv')
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].set_xlabel('flow rate (ml/s)')
    axs[1, 0].set_ylabel('pressure (mmHg)')

    if title is not None:
        fig.suptitle(title)

    if figpath is not None:
        plt.savefig(figpath)
    else:
        show = True
    
    if show:
        plt.show()

    return fig

def tree_impedance_subfig(ax, t, Z_t, title):
    '''
    plot Z_t for a structured tree
    '''

    ax.plot(t, Z_t, label='impedance')
    ax.set_ylabel('Impedance')
    ax.set_xlabel('time (s)')
    ax.set_title(title)

def plot_at_diff_params():

    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    fig2, axs2 = plt.subplots(3, 3, figsize=(20, 20))

    for i, d_min in enumerate([0.1, 0.05, 0.01]):
        for j, initial_d in enumerate([1.0, 0.5, 0.1]):

            t, P_t, Z_t, P_conv, n_vessels = get_tree_pressure(initial_d=initial_d, d_min=d_min)

            tree_pressure_subfig(axs[i, j], t, P_t, P_conv, f'{n_vessels} vessels, d_min = {d_min}, initial_d = {initial_d}')
            tree_impedance_subfig(axs2[i, j], t, Z_t, f'{n_vessels} vessels, d_min = {d_min}, initial_d = {initial_d}')

    
    fig.savefig('tests/cases/olufsen_impedance/tree_pressure_at_diff_params.png')
    fig2.savefig('tests/cases/olufsen_impedance/tree_impedance_at_diff_params.png')


def optimize_rcr_against_tree():

    pass

def test_visc_models():

    pass


def plot_pressure_post(initial_d=1.0, d_min=0.1):

    res_dir=f'tests/cases/olufsen_impedance/results_{initial_d}_{d_min}'

    result = pd.read_csv(f'{res_dir}/tree_results_1.0_0.1.csv')

    print(result.P_t)

    plt.figure()
    plt.plot(result.t, result.P_t)
    plt.show()

def plot_tree_imp():
    '''
    plot tree impedance as in Olufsen et al. 1999'''

    inflow = Inflow.periodic(path='~/Documents/Stanford/PhD/Marsden_Lab/PPAS/olufsen_impedance/CDG_NCSU/1D_CFD_struct_tree/Qin_8192.dat',t_per=1.1, n_periods=5)
    
    # inflow = Inflow.periodic(path='tests/cases/olufsen_impedance/inflow.flow')

    inflow.rescale(tsteps=2**9 * 5)

    # inflow.plot()

    q_period, t_period = inflow.period()

    # make some simulation parameters
    simparams = SimParams({
        'number_of_time_pts_per_cardiac_cycle': len(t_period)
    })

    # initialize the structured tree
    tree = StructuredTree(name='test_impedance', time=t_period, simparams=simparams)

    # build the tree
    initial_d = 1.0
    d_min = 0.1
    tree.build_tree(initial_d, d_min)

    # compute the impedance in frequency domain
    Z_om, omega = tree.compute_olufsen_impedance()

    # get positive values only
    # omega_pos = [w for w in omega if w >= 0]
    # Z_om_pos = [Z_om[omega.index(w)] for w in omega_pos]


    # compute RCR impedance
    R_1 = 1000.0
    C = 1e-5
    R_2 = 900.0
    Z_rcr = [(R_1 + R_2  + 1j * w * C * R_1 * R_2)/ (1 + 1j * w * R_2 * C) for w in omega]

    ## plot tree impedance ##

    fig1, ax = plt.subplots(3, 1)

    # get index of zero
    zero_idx = np.where(np.array(omega) == 0.0)[0][0]
    print(f'Z(0) = {Z_om[zero_idx]}')
    
    ax[0].plot(omega, np.abs(Z_om))
    # ax[0].plot(omega[zero_idx:zero_idx+20], np.abs(Z_rcr)[zero_idx:zero_idx+20])
    ax[0].set_ylabel('|Z|')
    ax[0].set_ylim([0, 3500])
    ax[0].set_xlim([0, 125])

    ax[1].plot(omega, np.abs(Z_om))
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_ylabel('|Z|')
    ax[1].set_ylim([1e2, 1e4])
    ax[1].set_xlim([1, 1e5])

    ax[2].plot(omega, np.degrees(np.angle(Z_om)))
    ax[2].set_xscale('log')
    ax[2].set_ylabel('phase(Z)')
    ax[2].set_ylim([-90, 90])
    ax[2].set_xlim([1, 1e5])

    plt.xlabel('Frequency (Hz)')

    fig1.suptitle(f'structured tree with {tree.count_vessels()} vessels, initial_d = {initial_d}cm, d_min = {d_min}cm')

    ## plot tree pressure ##
    # shift Z_om to be able to get inverse fft and match fft output
    Z_om = np.fft.ifftshift(Z_om)

    Q_om = np.fft.fft(inflow.q)

   #  Z_inv = [1 / z for z in Z_om]

    # convert pressure to frequency domain

    P_om = Z_om * Q_om[:len(Z_om)]

    # P_om[0] = 50 * 1333.2

    # P_t = np.array(np.fft.ifft(P_om).tolist() * 3)

    P_t = np.zeros(len(inflow.t))

    Z_t = np.fft.ifft(Z_om)


    # P_conv = np.convolve(Z_t, inflow.q)

    # P_conv1 = discrete_convolution_integral(Z_t, inflow.q, inflow.t)

    dt = np.mean(np.gradient(inflow.t))

    ### Q_T NEEDS TO BE SLIDING ACROSS THE PERIOD

    P_conv = discrete_conv_int2(Z_t, inflow.q, dt, N=2**9)

    n_vessels = tree.count_vessels()

    res_dir = f'tests/cases/olufsen_impedance/results_{initial_d}_{d_min}_cint2_qin8192'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    tree_pres_flow_fig(inflow.t, inflow, P_t, P_conv[:len(inflow.t)], title=f'{n_vessels} vessels, initial_d = 1.0cm, d_min = 0.05cm', figpath=f'{res_dir}/tree_pres_flow_{initial_d}_{d_min}.png')

    fig1.savefig(f'{res_dir}/tree_impedance_{initial_d}_{d_min}.png')

    result_df = pd.DataFrame({'omega': omega, 'Z_t': Z_t,
                              't': t_period, 'Q_t': q_period
                              # 'P_t': P_t, 'P_conv': P_conv[:len(inflow.t)]
                              })

    result_df.to_csv(f'{res_dir}/tree_results_{initial_d}_{d_min}.csv')


def discrete_convolution_integral(Z_t: list, Q_t: list, t: list):
    '''
    compute the convolution integral of Z_t and Q_t
    '''

    def conv_int_disc(q: list, z: list, dt: float, n: int, N: int):
        
        t1 = q[n] * z[0] * dt

        t2 = 0.0
        for k in range(1, N):
            t2 += z[k] * q[(n-k)%N] * dt
        
        p = t1 + t2

        return p
    
    N = len(t)
    dt = np.mean(np.gradient(t))

    P_conv = np.zeros(N)

    for n in range(N):
        P_conv[n] = conv_int_disc(Q_t, Z_t, dt, n, N)


    return P_conv


def discrete_conv_int2(Z_t: list, Q_t: list, dt: float, N: int):
    '''
    compute the convolution integral of Z_t and Q_t by multiplying Z_t by Q_t in reverse over the last interval

    N is the number of timesteps in a period
    '''

    p_conv = np.zeros(len(Q_t))

    for n, q in enumerate(Q_t):
        if n < N:
            p_conv[n] = Z_t[0] * q / 1333.2
        else:
            Q_per = Q_t[n-N:n]
            Q_rev = Q_per[::-1]
            Z_rev = Z_t[::-1]
            for k in range(N):
                p_conv[n] += Z_t[k] * Q_rev[k] / 1333.2
            
            print(f'p_conv[{n}] = {p_conv[n]}')
    
    return p_conv


def test_Zfunc(omega):
    '''
    test the Zfunc function
    '''

    rho = 1.055
    g = 981.0
    Lr = 1.0
    q = 10.0

    inflow = Inflow.periodic(path='~/Documents/Stanford/PhD/Marsden_Lab/PPAS/olufsen_impedance/CDG_NCSU/1D_CFD_struct_tree/Qin_8192.dat',t_per=1.1, n_periods=1)

    inflow.rescale(100.0)

    omega = omega * Lr**3 / q


    # make some simulation parameters
    simparams = SimParams({
        'number_of_time_pts_per_cardiac_cycle': len(inflow.t)
    })

    # initialize the structured tree
    tree = StructuredTree(name='test_impedance', simparams=simparams)

    initial_d = 1.0
    d_min = 0.05
    tree.build_tree(initial_d, d_min)

    Z_om = tree.root.z0_olufsen(omega)

    print(f'Z(w={omega/Lr**3*q}) = {Z_om}')
    

def test_fft_shift():

    inflow = Inflow.periodic(path='~/Documents/Stanford/PhD/Marsden_Lab/PPAS/olufsen_impedance/CDG_NCSU/1D_CFD_struct_tree/Qin_8192.dat',t_per=1.1, n_periods=1)

    inflow.rescale(100.0, tsteps=2**10)

    # inflow.plot()

    q_period, t_period = inflow.period()

    Q_om = np.fft.fft(inflow.q, norm='backward')

    print(f'Q_om before fftshift = {Q_om}')

    Q_om_shift = np.fft.fftshift(Q_om)

    print(f'Q_om after fftshift = {Q_om_shift}')


def create_impedance_config(initial_d=1.0, d_min=0.1, filepath='tests/cases/olufsen_impedance/pulsatileFlow_IMPEDANCE.json'):
    '''
    create a config file for impedance test case
    '''

    # get the inflow
    # inflow = Inflow.periodic(path='~/Documents/Stanford/PhD/Marsden_Lab/PPAS/olufsen_impedance/CDG_NCSU/1D_CFD_struct_tree/Qin_8192.dat',t_per=1.1)

    inflow = Inflow.periodic(path='tests/cases/olufsen_impedance/inflow.flow')

    inflow.rescale(tsteps=2**9)

    # simulation parameters
    simparams = SimParams({
        "number_of_cardiac_cycles": 10,
        "number_of_time_pts_per_cardiac_cycle": len(inflow.t),
        "output_all_cycles": True
    })

    # compute impedance
    # initialize the structured tree
    tree = StructuredTree(name='test_impedance', time=inflow.t, simparams=simparams)

    # build the tree
    tree.build_tree(initial_d, d_min)

    # compute the impedance in frequency domain
    Z_om, omega = tree.compute_olufsen_impedance()

    Z_om = np.fft.ifftshift(Z_om)

    Z_t = np.fft.ifft(Z_om)

    print(Z_t)

    Z_t = np.real(Z_t) 

    config = {
        "description": {
            "description of test case" : "pulsatile flow -> R -> impedance bc"
        },
        "boundary_conditions": [
            inflow.to_dict(),
            {
                "bc_name": "OUTLET",
                "bc_type": "IMPEDANCE",
                "bc_values": {
                    "Z": Z_t.tolist(),
                    "t": inflow.t.tolist()
                }
            }
        ],
        "simulation_parameters": simparams.to_dict(),
        "vessels": [
            {
                "boundary_conditions": {
                    "inlet": "INFLOW",
                    "outlet": "OUTLET"
                },
                "vessel_id": 0,
                "vessel_length": 10.0,
                "vessel_name": "branch0_seg0",
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {
                    "R_poiseuille": 100.0
                }
            }
        ]
    }

    with open(filepath, 'w') as ff:
        json.dump(config, ff, indent=4)










if __name__ == '__main__':


    # create_impedance_config(filepath='tests/cases/olufsen_impedance/paFlow_IMPEDANCE.json')

    plot_tree_imp()

    # plot_pressure_post()

    # test_fft_shift()

    # FOR OLUFSEN Z(5.71) = 2353.02-1127.52j
    # test_Zfunc(5.7119866428905333)

