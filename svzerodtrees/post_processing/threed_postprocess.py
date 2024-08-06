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



if __name__ == '__main__':

    os.chdir('../threed_models/AS2')
    svpre_file = 'preop/preop.svpre'

    plot_bc_adaptation(svpre_file)






