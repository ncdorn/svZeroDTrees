from svzerodtrees.threedutils import *
from matplotlib import pyplot as plt


def plot_bc_adaptation(svpre_file, n_steps=1000):
    '''
    plot the adaptation at each outlet for the 3d simulation'''

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
    plt.title('Outlet adaptation')
    plt.savefig('outlet_adaptation.png')





if __name__ == '__main__':

    os.chdir('../threed_models/AS2_ext_stent')
    svpre_file = 'preop/preop.svpre'

    plot_bc_adaptation(svpre_file)






