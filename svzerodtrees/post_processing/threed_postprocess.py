from svzerodtrees.threedutils import *


if __name__ == '__main__':

    os.chdir('../threed_models/AS2_stent')

    preop_split = compute_flow_split('preop/Q_svZeroD', 'preop/preop.svpre')
    postop_split = compute_flow_split('postop/Q_svZeroD', 'postop/postop.svpre')
    adapted_split = compute_flow_split('adapted/Q_svZeroD', 'adapted/adapted.svpre')

    with open('flow_split_results.txt', 'w') as f:
        f.write('flow splits as LPA/RPA\n\n')
        f.write('preop flow split: ' + str(preop_split[0]) + '/' + str(preop_split[1]) + '\n')
        f.write('postop flow split: ' + str(postop_split[0]) + '/' + str(postop_split[1]) + '\n')
        f.write('adapted flow split: ' + str(adapted_split[0]) + '/' + str(adapted_split[1]) + '\n')


