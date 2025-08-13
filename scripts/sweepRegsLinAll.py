from src.MPOpto import *
from src.LinModel import *
from src.NLModel import *
from src.CURBD.CurbdModel import *
from src.analysis import *
from src.regression import *
from src.utils.gen_utils import *
from tqdm.auto import tqdm
from itertools import permutations, compress, product
import re
import sys

temp = sys.argv[0]
script_name = temp.split('.')[0]
num_workers = 20
post=45
binsize=10
nlags=5
pre=binsize*nlags
num_PCs=20
sw_weight=50

random_state=1
rng = np.random.default_rng(random_state)


session_path_list = []
session_name_list = []
co_path = '../../data/co'

for root, dirs, files in os.walk(co_path, topdown=True):
    for name in dirs:
        if re.search('co._', name) or re.search('co.._', name):
            session_name_list.append(name)
            session_path_list.append(os.path.join(root, name))

#cu_path = '../../data/cu2'
#if os.path.isdir(cu_path):
#    for item in os.listdir(cu_path):
#        session_path_list.append(os.path.join(cu_path, item))
#        session_name_list.append(item)

def analyze(reg):

        print('starting analysis')
        c = generate_sepreg(reg, X_b, nlags=nlags, num_region1=num_region1)
        h_b=train_wiener_filter(X_b, Y_b, c=c,sw=sws)

        h_c = train_wiener_filter(X_c, Y_c, c=c,sw=None)

        h_cc=train_wiener_filter(X_c, Y_c, c=c,sw=cws)
        
        drive_b = session.get_relative_drive(h_b, X_bt, nlags=nlags,
            num_region1=num_region1)
        drive_c = session.get_relative_drive(h_c, X_ct, nlags=nlags,
            num_region1=num_region1)
        drive_cc = session.get_relative_drive(h_cc, X_ct, nlags=nlags,
            num_region1=num_region1)


        yhat_c = test_wiener_filter(X_ct, h_c)
        r2_c = weighted_r2(Y_ct, yhat_c)

        yhat_b = test_wiener_filter(X_bt, h_b)
        r2_b = weighted_r2(Y_bt, yhat_b)

        yhat_cc = test_wiener_filter(X_ct, h_cc)
        r2_cc = weighted_r2(Y_ct, yhat_cc)

        return (drive_b, r2_b), (drive_c, r2_c), (drive_cc,r2_cc)


output=[]
asymps=[]

for session_path in session_path_list:
    session_name = session_path.split('/')[-1]
    session = LinModel(session_path)

    if num_PCs is None:
        num_region1 = session.num_region1
    else:
        num_region1 = num_PCs


    laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre=pre, post=post)
    omit_lasers, _ = session.adjustLaserBounds(pre=0, post=300)
    train_lasers, test_lasers = train_test_split(laser_bounds, train_size=.8,
            random_state=random_state)
    temp_mask = rng.choice(len(ctrl_bounds), len(test_lasers), replace=False)
    ctrl_bounds_subsamp = ctrl_bounds[temp_mask, :]
    both_bounds = np.vstack((test_lasers, ctrl_bounds_subsamp))
    train_lasers = np.sort(train_lasers, axis=0)
    test_lasers = np.sort(test_lasers, axis=0)
    justClimbs = omitBoundInBounds(session.climbing_bounds, omit_lasers)
    justClimbs = omitBoundInBounds(justClimbs, ctrl_bounds)
    justClimbs = minimumBoundSize(justClimbs, min_size=post+pre)
    both_bounds = np.sort(both_bounds, axis=0)
    omitLaserBounds= omitBoundInBounds(session.climbing_bounds, test_lasers)
    omitLaserBounds = omitBoundInBounds(omitLaserBounds, ctrl_bounds)
    omitLaserBounds = minimumBoundSize(omitLaserBounds, min_size = post+pre) 

    sws = session.getSWs(bigBound=omitLaserBounds, smallBound=train_lasers,
            sw_weight=sw_weight, binsize=binsize, nlags=nlags)
    new_ctrls = session.generate_new_ctrls(len(train_lasers), pre=pre,
            post=post)
    cws = session.getSWs(bigBound=justClimbs, smallBound=new_ctrls,
            sw_weight=sw_weight, binsize=binsize, nlags=nlags)

    (X_c, Y_c), c_PCs = session.generate_trainset(justClimbs, binsize=binsize,
            nlags=nlags, num_PCs=num_PCs)
    (X_b, Y_b), b_PCs = session.generate_trainset(justClimbs, binsize=binsize,
            nlags=nlags, num_PCs=num_PCs)



    X_ct, Y_ct = session.generate_testset(bounds=ctrl_bounds, binsize=binsize,
        nlags=nlags, PCAObjs = c_PCs)
    X_bt, Y_bt = session.generate_testset(bounds=both_bounds, binsize=binsize,
        nlags=nlags, PCAObjs = b_PCs)

    r1_mask, r2_mask = mask_input_region(X_c, nlags=nlags, num_region1=num_region1)

    h_c_r1 = train_wiener_filter(X_c[:, r1_mask], Y_c, c=(4,8))
    yhat_c_r1 = test_wiener_filter(X_ct[:, r1_mask], h_c_r1)
    r2_c_r1 = weighted_r2(Y_ct, yhat_c_r1)
    h_c_r2 = train_wiener_filter(X_c[:, r2_mask], Y_c, c=(4,8))
    yhat_c_r2 = test_wiener_filter(X_ct[:, r2_mask], h_c_r2)
    r2_c_r2 = weighted_r2(Y_ct, yhat_c_r2)

    h_b_r1 = train_wiener_filter(X_b[:, r1_mask], Y_b, sw=sws, c=(4,8))
    yhat_b_r1 = test_wiener_filter(X_bt[:, r1_mask], h_b_r1)
    r2_b_r1 = weighted_r2(Y_bt, yhat_b_r1)
    h_b_r2 = train_wiener_filter(X_b[:, r2_mask], Y_b, sw=sws, c=(4,8))
    yhat_b_r2 = test_wiener_filter(X_bt[:, r2_mask], h_b_r2)
    r2_b_r2 = weighted_r2(Y_bt, yhat_b_r2)

    h_cc_r1 = train_wiener_filter(X_c[:, r1_mask], Y_c, sw=cws, c=(4,8))
    yhat_cc_r1 = test_wiener_filter(X_ct[:, r1_mask], h_cc_r1)
    r2_cc_r1 = weighted_r2(Y_ct, yhat_cc_r1)
    h_cc_r2 = train_wiener_filter(X_c[:, r2_mask], Y_c, c=(4,8), sw=cws)
    yhat_cc_r2 = test_wiener_filter(X_ct[:, r2_mask], h_cc_r2)
    r2_cc_r2 = weighted_r2(Y_ct, yhat_cc_r2)



    asymps.append(((r2_b_r2, r2_b_r1), (r2_c_r2, r2_c_r1), (r2_cc_r1,
        r2_cc_r2)))

    
    ustream = np.logspace(4,8,18)
    dstream = np.logspace(4,8,18)
    reg_combos = list(product(dstream, ustream))

    temp = multipool(analyze, iterable=reg_combos, num_workers=num_workers)
    output.append(temp)

pdump(output, 
        f'../../picklejar/{script_name}_co.pickle')
pdump(asymps, 
        f'../../picklejar/{script_name}_co_asymps.pickle')

