from svzerodtrees.interface import run_threed_adaptation
import os


if __name__ == '__main__':

    # preop_dir = '~/Documents/Stanford/PhD/Marsden_Lab/SimVascular/threed_models/LPA_RPA/steady/preop'
    # postop_dir = '~/Documents/Stanford/PhD/Marsden_Lab/SimVascular/threed_models/LPA_RPA/steady/postop'
    # adapted_dir = '~/Documents/Stanford/PhD/Marsden_Lab/SimVascular/threed_models/LPA_RPA/steady/adapted'

    preop_dir = '../threed_models/LPA_RPA/steady/preop'
    postop_dir = '../threed_models/LPA_RPA/steady/postop2'
    adapted_dir = '../threed_models/LPA_RPA/steady/adapted2'

    run_threed_adaptation(preop_dir, postop_dir, adapted_dir)