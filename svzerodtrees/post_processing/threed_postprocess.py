from svzerodtrees.threedutils import *
from matplotlib import pyplot as plt


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
    data.rename({'188': 'time'}, axis=1, inplace=True)

    data[pres_col] = data[pres_col] / 1333.2 # convert pressure to mmHg

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






if __name__ == '__main__':

    sim_dir = '../threed_models/impedance_3D/pipe_imp_t002'

    plot_data(sim_dir, 'branch0_seg0', 'inflow')




