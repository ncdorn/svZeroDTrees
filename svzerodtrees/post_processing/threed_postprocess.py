from svzerodtrees.threedutils import *
from matplotlib import pyplot as plt
import numpy as np


def plot_preop_postop_change(svpre_file, filepath='prepost_outlet_change.png', n_steps=1000):
    '''
    plot the adaptation at each outlet for the 3d simulation
    
    :svpre_file: path to the preop svpre file
    :filepath: filepath to save the plot'''

    lpa_idxs, rpa_idxs = get_lpa_rpa_idxs(svpre_file)
    
    preop_q = pd.read_csv('preop/Q_svZeroD', sep='\s+')
    postop_q = pd.read_csv('postop/Q_svZeroD', sep='\s+')

    
    preop_q_lpa = preop_q.loc[:, lpa_idxs]
    preop_q_rpa = preop_q.loc[:, rpa_idxs]

    postop_q_lpa = postop_q.loc[:, lpa_idxs]
    postop_q_rpa = postop_q.loc[:, rpa_idxs]

    lpa_adapt = (preop_q_lpa.mean() - postop_q_lpa.mean()) / preop_q_lpa.mean()
    rpa_adapt = (preop_q_rpa.mean() - postop_q_rpa.mean()) / preop_q_rpa.mean()

    plt.figure()
    plt.bar(lpa_adapt.index, lpa_adapt * 100, label='LPA', color='tomato')
    plt.bar(rpa_adapt.index, rpa_adapt * 100, label='RPA', color='cornflowerblue')
    plt.ylabel(f'% change in flow')
    plt.xticks([])
    plt.legend()
    plt.title('change in flow at outlets after repair')
    plt.savefig(filepath)


def plot_bc_adaptation(svpre_file, filepath='outlet_adaptation.png', n_steps=1000):
    '''
    plot the adaptation at each outlet for the 3d simulation'''

    lpa_idxs, rpa_idxs = get_lpa_rpa_idxs(svpre_file)
    
    postop_q = pd.read_csv('postop/Q_svZeroD', sep='\s+')
    adapted_q = pd.read_csv('adapted/Q_svZeroD', sep='\s+')

    
    adapted_q_lpa = adapted_q.loc[:, lpa_idxs]
    adapted_q_rpa = adapted_q.loc[:, rpa_idxs]

    postop_q_lpa = postop_q.loc[:, lpa_idxs]
    postop_q_rpa = postop_q.loc[:, rpa_idxs]

    lpa_adapt = (adapted_q_lpa.mean() - postop_q_lpa.mean()) / adapted_q_lpa.mean()
    rpa_adapt = (adapted_q_rpa.mean() - postop_q_rpa.mean()) / adapted_q_rpa.mean()

    plt.figure()
    plt.bar(lpa_adapt.index, lpa_adapt * 100, label='LPA', color='tomato')
    plt.bar(rpa_adapt.index, rpa_adapt * 100, label='RPA', color='cornflowerblue')
    plt.ylabel(f'% change in flow')
    plt.xticks([])
    plt.legend()
    plt.title('Outlet adaptation')
    plt.savefig(filepath)
    

def plot_data(sim_dir, coupling_block, block_name):
    '''
    plot the data from svZerod_data file
    
    :field: field to plot, flow or pressure
    :block_name: name of the sv0d block to plot'''

    # name of pd column is of the form field:coupling:block_name

    pres_col = f'pressure:{coupling_block}:{block_name}'
    flow_col = f'flow:{coupling_block}:{block_name}'

    # load the data
    data = pd.read_csv(os.path.join(sim_dir, 'svZeroD_data'), sep='\s+')
    data.rename({data.columns[0]: 'time'}, axis=1, inplace=True)

    data[pres_col] = data[pres_col] / 13.332 # convert pressure to mmHg

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(data['time'], data[pres_col])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('pressure (mmHg)')

    axs[1].plot(data['time'], data[flow_col])
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('flow (cm3/s)')
    
    plt.suptitle(f'{block_name}')

    plt.tight_layout()
    plt.show()


def plot_mpa_and_flowsplit(sim_dir):
    '''
    plot mpa flow + pressure and the flow split between the lpa and rpa
    
    param sim_dir: path to the simulation directory containing mesh-complete, svZeroD_data, svZeroD_interface.dat, svzerod_3Dcoupling.json
    '''

    data = pd.read_csv(os.path.join(sim_dir, 'svZeroD_data'), sep='\s+')

    data.rename({data.columns[0]: 'time'}, axis=1, inplace=True)

    # get the mesh complete names to form lpa/rpa map
    filelist_raw = glob.glob(os.path.join(sim_dir, 'mesh-complete/mesh-surfaces/*.vtp'))

    filelist = [file for file in filelist_raw if 'wall' not in file]

    filelist.sort()

    # remove inflow
    filelist.remove(filelist[-1])

    zerod_coupler = os.path.join(sim_dir, 'svzerod_3Dcoupling.json')

    threed_coupler = ConfigHandler.from_json(zerod_coupler, is_pulmonary=False, is_threed_interface=True)

    # get a map of bc names to outlet idxs
    outlet_blocks = [block.name for block in list(threed_coupler.coupling_blocks.values())]

    outlet_blocks.remove('mpa')

    block_to_outlet = {block: file for block, file in zip(outlet_blocks, filelist)}

    lpa_rpa_block_map = {'lpa': [], 'rpa': []}
    for block in outlet_blocks:
        if 'lpa' in block_to_outlet[block].lower():
            lpa_rpa_block_map['lpa'].append(block)
        elif 'rpa' in block_to_outlet[block].lower():
            lpa_rpa_block_map['rpa'].append(block)

    # get the mpa pressure and flow
    plot_data(sim_dir, 'branch0_seg0', 'mpa')

    # get the flow split
    lpa_flow = 0.0
    rpa_flow = 0.0
    for lpa_block in lpa_rpa_block_map['lpa']:
        res = lpa_block[:10]
        idx = int(lpa_block[10:])
        bc_name = f'{res}_{idx}'
        lpa_flow += integrate_flow(data, lpa_block, bc_name)

    for rpa_block in lpa_rpa_block_map['rpa']:
        res = rpa_block[:10]
        idx = int(rpa_block[10:])
        bc_name = f'{res}_{idx}'
        rpa_flow += integrate_flow(data, rpa_block, bc_name)

    plt.figure()

    percent = {'LPA': lpa_flow / (lpa_flow + rpa_flow) * 100,
                'RPA': rpa_flow / (lpa_flow + rpa_flow) * 100}
        
    q = {'LPA': lpa_flow,
            'RPA': rpa_flow}

    # plot the stacked bar graph
    bottom = 0.0
    for vessel, value in q.items():
        plt.bar(' ', value, label=vessel, bottom=bottom, 
                     # color=colors[vessel]
                     )
        plt.text(' ', value / 2 + bottom, f'{vessel}: {str(int(percent[vessel]))}%', ha='center', va='center')
        bottom += value
    plt.ylabel('flow [cm^3/s]')

    plt.show()


        

def integrate_flow(svzerod_data, coupling_block, block_name):
    '''
    integrate the flow at the outlet
    
    :coupling_block: name of the coupling block
    :block_name: name of the block to integrate the flow over'''

    flow_col = f'flow:{coupling_block}:{block_name}'

    flow = svzerod_data[flow_col]

    return np.trapz(flow, svzerod_data['time'])






if __name__ == '__main__':

    sim_dir = '../threed_models/SU0243/preop'

    plot_mpa_and_flowsplit(sim_dir)

    




