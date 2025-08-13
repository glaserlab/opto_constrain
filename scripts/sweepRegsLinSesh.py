from src.MPOpto import *
from src.LinModel import *
from src.NLModel import *
from src.CURBD.CurbdModel import *
from src.analysis import *
from src.regression import *
from src.utils.gen_utils import *
from tqdm.auto import tqdm
from itertools import permutations, compress, product

import sys

temp = sys.argv[0]
script_name = temp.split('.')[0]
num_workers = 20
session_path = sys.argv[1]
#post = int(sys.argv[2])
print(session_path)
#print(post)
post=45
session_name = session_path.split('/')[-1]
binsize=10
nlags=5
pre=binsize*nlags
num_PCs=20
sw_weight=50

random_state=1
rng = np.random.default_rng(random_state)

session = LinModel(session_path)

if num_PCs is None:
    num_region1 = session.num_region1
else:
    num_region1 = num_PCs


laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre=pre, post=post)
omit_lasers, _  = session.adjustLaserBounds(pre=pre, post=300)
train_lasers, test_lasers = train_test_split(laser_bounds, train_size=.8,
        random_state=random_state)
temp_mask = rng.choice(len(ctrl_bounds), len(test_lasers), replace=False)
ctrl_bounds_subsamp = ctrl_bounds[temp_mask, :]
both_bounds = np.vstack((test_lasers, ctrl_bounds_subsamp))
train_lasers = np.sort(train_lasers, axis=0)
test_lasers = np.sort(test_lasers, axis=0)
both_bounds = np.sort(both_bounds, axis=0)
omitLaserBounds= omitBoundInBounds(session.climbing_bounds, test_lasers)
omitLaserBounds = omitBoundInBounds(omitLaserBounds, ctrl_bounds)
omitLaserBounds = minimumBoundSize(omitLaserBounds, min_size = post+pre) 
justClimbs = omitBoundInBounds(session.climbing_bounds, omit_lasers)
justClimbs = omitBoundInBounds(justClimbs, ctrl_bounds)
justClimbs = minimumBoundSize(justClimbs, min_size=post+pre)

sws = session.getSWs(bigBound=omitLaserBounds, smallBound=train_lasers,
        sw_weight=sw_weight, binsize=binsize, nlags=nlags)

(X,Y), b_PCs = session.generate_trainset(omitLaserBounds, binsize=binsize,
        nlags=nlags, num_PCs=num_PCs)
(X_climb,Y_climb), c_PCs = session.generate_trainset(justClimbs, binsize=binsize,
        nlags=nlags, num_PCs=num_PCs)

X_ctrl, Y_ctrl = session.generate_testset(bounds=ctrl_bounds, binsize=binsize,
    nlags=nlags, PCAObjs = c_PCs)
X_both, Y_both = session.generate_testset(bounds=both_bounds, binsize=binsize,
    nlags=nlags, PCAObjs = b_PCs)

r1_mask, r2_mask = mask_input_region(X, nlags=nlags, num_region1=num_region1)
h_ctrl_r1 = train_wiener_filter(X_climb[:, r1_mask], Y_climb, c=(4,8))
yhat_ctrl_r1 = test_wiener_filter(X_ctrl[:, r1_mask], h_ctrl_r1)
r2_ctrl_r1 = weighted_r2(Y_ctrl, yhat_ctrl_r1)
h_ctrl_r2 = train_wiener_filter(X_climb[:, r2_mask], Y_climb, c=(4,8))
yhat_ctrl_r2 = test_wiener_filter(X_ctrl[:, r2_mask], h_ctrl_r2)
r2_ctrl_r2 = weighted_r2(Y_ctrl, yhat_ctrl_r2)

h_both_r1 = train_wiener_filter(X[:, r1_mask], Y, sw=sws, c=(4,8))
yhat_both_r1 = test_wiener_filter(X_both[:, r1_mask], h_both_r1)
r2_both_r1 = weighted_r2(Y_both, yhat_both_r1)
h_both_r2 = train_wiener_filter(X[:, r2_mask], Y, sw=sws, c=(4,8))
yhat_both_r2 = test_wiener_filter(X_both[:, r2_mask], h_both_r2)
r2_both_r2 = weighted_r2(Y_both, yhat_both_r2)

asymps = ((r2_both_r2, r2_both_r1), (r2_ctrl_r2, r2_ctrl_r1))

def analyze(reg):

    print('starting analysis')
    c = generate_sepreg(reg, X, nlags=nlags, num_region1=num_region1)
    h_both=train_wiener_filter(X, Y, c=c,sw=sws)
    print('finished h_both')

    h_climb = train_wiener_filter(X_climb, Y_climb, c=c,sw=None)
    
    drive_both = session.get_relative_drive(h_both, X_both, nlags=nlags,
        num_region1=num_region1)
    drive_ctrl = session.get_relative_drive(h_climb, X_ctrl, nlags=nlags,
        num_region1=num_region1)


    yhat_ctrl = test_wiener_filter(X_ctrl, h_climb)
    r2_ctrl = weighted_r2(Y_ctrl, yhat_ctrl)
    print(r2_ctrl)
    yhat_both = test_wiener_filter(X_both, h_both)
    r2_both = weighted_r2(Y_both, yhat_both)
    print(r2_both)

    return (drive_both, r2_both), (drive_ctrl, r2_ctrl)


ustream = np.logspace(4,8,20)
dstream = np.logspace(4,8,20)
reg_combos = list(product(dstream, ustream))

output = multipool(analyze, iterable=reg_combos, num_workers=num_workers)
print('finished!')

pdump(output, 
        f'../../picklejar/{script_name}_{session_name}.pickle')
pdump(asymps, 
        f'../../picklejar/{script_name}_{session_name}_asymps.pickle')

