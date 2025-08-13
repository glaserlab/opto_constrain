from src.MPOpto import *
from src.LinModel import *
from src.NLModel import *
from src.NLModel_single import *
from src.CURBD.CurbdModel import *
from src.analysis import *
from src.regression import * 
from src.utils.gen_utils import *
from tqdm.auto import tqdm
from itertools import permutations, compress, product

import sys
import re

temp = sys.argv[0]
script_name = temp.split('.')[0]
binsize=10
nlags=5
pre=binsize*nlags
post=45
sw_weight=50
num_epochs=500

session_path_list = []
session_name_list = []
co_path = '../data/co'

for root, dirs, files in os.walk(co_path, topdown=True):
    for name in dirs:
        if re.search('co._', name) or re.search('co.._', name):
            session_name_list.append(name)
            session_path_list.append(os.path.join(root, name))



random_state=0
rng = np.random.default_rng(random_state)
output=[]

b_predics=[]
c_predics=[]
cc_predics=[]

for session_path in tqdm(session_path_list):
    session_name = session_path.split('/')[-1]
    print(f'session: {session_name}')
    session = NLModel(session_path)
    num_region1 = session.num_region1
    reg_sweep_c = (-1.5,-2)
    reg_sweep_o = (-1,-1.5)

    laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre=pre, post=post)
    train_lasers, test_lasers = train_test_split(laser_bounds, train_size=.8,
            random_state=random_state)
    train_lasers = np.sort(train_lasers, axis=0)
    test_lasers = np.sort(test_lasers, axis=0)
    omitLaserBounds= omitBoundInBounds(session.climbing_bounds, test_lasers)
    omitLaserBounds = omitBoundInBounds(omitLaserBounds, ctrl_bounds)
    omitLaserBounds = minimumBoundSize(omitLaserBounds, min_size = post+pre) 



    sws = session.getSWs(bigBound=omitLaserBounds, smallBound=train_lasers,
            sw_weight=sw_weight, binsize=binsize, nlags=nlags)
    new_ctrls = session.generate_new_ctrls(len(train_lasers), pre=pre,
            post=post)
    cws = session.getSWs(bigBound=omitLaserBounds, smallBound=new_ctrls,
            sw_weight=sw_weight, binsize=binsize, nlags=nlags)


    Xs, Y = session.format_nlags(omitLaserBounds, binsize=binsize,
            nlags=nlags)

    Xs_ct, Y_ct = session.format_nlags(bounds=ctrl_bounds, binsize=binsize,
        nlags=nlags)

    Xs_lt, Y_lt = session.format_nlags(bounds=test_lasers, binsize=binsize,
        nlags=nlags)

    models_c, _ = session.train_sweep(Xs, Y, num_epochs, reg_sweep=reg_sweep_c,
            sw=None, num_sweep=5, display=True)
    _, (_, c_r2_c) = session.test(models_c, Xs_ct, Y_ct)
    _, (_, c_r2_l) = session.test(models_c, Xs_lt, Y_lt)

    c_predics.append((c_r2_c, c_r2_l))

    models_b, _ = session.train_sweep(Xs, Y, num_epochs, reg_sweep=reg_sweep_o,
            sw=sws, num_sweep=5, display=True)
    _, (_, b_r2_c) = session.test(models_b, Xs_ct, Y_ct)
    _, (_, b_r2_l) = session.test(models_b, Xs_lt, Y_lt)

    b_predics.append((b_r2_c, b_r2_l))

    print(f'ctrl predics: {c_r2_c}, {b_r2_c}')
    print(f'laser predics: {c_r2_l}, {b_r2_l}')

    models_cc,_ = session.train_sweep(Xs, Y, num_epochs, reg_sweep=reg_sweep_c,
            sw=cws, num_sweep=5, display=False)
    _, (_, cc_r2_c) = session.test(models_cc, Xs_ct, Y_ct)
    _, (_, cc_r2_l) = session.test(models_cc, Xs_lt, Y_lt)

    cc_predics.append((cc_r2_c, cc_r2_l))


output = (b_predics, c_predics, cc_predics)
pdump(output, f'../picklejar/sweepNLPredics_test.pickle')
