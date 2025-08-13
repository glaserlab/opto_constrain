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
num_PCs=20

session_path_list = []
session_name_list = []
co_path = '../data/co'

for root, dirs, files in os.walk(co_path, topdown=True):
    for name in dirs:
        if re.search('co._', name) or re.search('co.._', name):
            session_name_list.append(name)
            session_path_list.append(os.path.join(root, name))

random_state=1
rng = np.random.default_rng(random_state)

b_predics=[]
c_predics=[]
cc_predics=[]

c=(2,6)

for session_path in tqdm(session_path_list):
    session_name = session_path.split('/')[-1]
    session = LinModel(session_path)

    if num_PCs is None:
        num_region1 = session.num_region1
    else:
        num_region1 = num_PCs


    laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre=pre, post=post)
    train_lasers, test_lasers = train_test_split(laser_bounds, train_size=.8,
            random_state=random_state)
    temp_mask = rng.choice(len(ctrl_bounds), len(test_lasers), replace=False)
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
    
    (X,Y), PCs = session.generate_trainset(omitLaserBounds, binsize=binsize,
        nlags=nlags, num_PCs=num_PCs)

    X_ct, Y_ct = session.generate_testset(bounds=ctrl_bounds, binsize=binsize,
        nlags=nlags, PCAObjs = PCs)
    X_lt, Y_lt = session.generate_testset(bounds=test_lasers, binsize=binsize,
        nlags=nlags, PCAObjs = PCs)
    
    h_b=train_wiener_filter(X, Y, c=c,sw=sws)
    b_yct = test_wiener_filter(X_ct, h_b)
    b_ct_r2 = weighted_r2(Y_ct, b_yct)
    b_ylt = test_wiener_filter(X_lt, h_b)
    b_lt_r2 = weighted_r2(Y_lt, b_ylt)
    b_predics.append((b_ct_r2, b_lt_r2))
    

    h_c = train_wiener_filter(X, Y, c=c,sw=None)
    c_yct = test_wiener_filter(X_ct, h_c)
    c_ct_r2 = weighted_r2(Y_ct, c_yct)
    c_ylt = test_wiener_filter(X_lt, h_c)
    c_lt_r2 = weighted_r2(Y_lt, c_ylt)
    c_predics.append((c_ct_r2, c_lt_r2))

    h_cc=train_wiener_filter(X, Y, c=c,sw=cws)
    cc_yct = test_wiener_filter(X_ct, h_cc)
    cc_ct_r2 = weighted_r2(Y_ct, cc_yct)
    cc_ylt = test_wiener_filter(X_lt, h_cc)
    cc_lt_r2 = weighted_r2(Y_lt, cc_ylt)
    cc_predics.append((cc_ct_r2, cc_lt_r2))
    
output=(b_predics, c_predics, cc_predics)

pdump(output, f'../picklejar/sweepLinPredics.pickle')
