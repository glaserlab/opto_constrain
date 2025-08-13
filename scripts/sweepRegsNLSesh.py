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
import time

import sys

temp = sys.argv[0]
script_name = temp.split('.')[0]
session_path = sys.argv[1]
session_name = session_path.split('/')[-1]
binsize=10
nlags=5
pre=binsize*nlags
post=45
sw_weight=30
num_epochs=1000
random_state=1
rng = np.random.default_rng(random_state)

def find_lowest(drive_list, curr_target, threshold):
    while True:
        drive_array = np.array(drive_list)
        drive_array = drive_array[drive_array>curr_target]
        drive_array = drive_array[drive_array < (curr_target + threshold)]

        if drive_array.size>0:
            curr_target = np.max(drive_array)
            print(f'new target is {curr_target}')
        else:
            return curr_target

def analyze(reg, Xs_test, Y_test, sws):
    models, _ = session.train(Xs,Y, num_epochs, reg, sw=sws, valid_split=True,
            early_stop=True, display=False)
    _, (_, r2) = session.test(models, Xs_test, Y_test)
    drive = session.get_relative_drive(models, Xs_test)

    return (drive, r2)

def asymps(Xs_test, Y_test, sws, region=0):
    single_session = NLModel_single(session_path)
    X = Xs[region]
    X_test = Xs_test[region]

    models = single_session.train_sweep(X, Y, num_epochs, reg_sweep=(-2, -1),
            sw=sws, display=True)
    _, (_, r2) = single_session.test(models, X_test, Y_test)

    return r2

#set-up
session = NLModel(session_path)
num_region1 = session.num_region1

laser_bounds, ctrl_bounds = session.adjustLaserBounds(pre=pre, post=post)
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



sws = session.getSWs(bigBound=omitLaserBounds, smallBound=train_lasers,
        sw_weight=sw_weight, binsize=binsize, nlags=nlags)
new_ctrls = session.generate_new_ctrls(len(train_lasers), pre=pre,
        post=post)
cws = session.getSWs(bigBound=omitLaserBounds, smallBound=new_ctrls,
        sw_weight=sw_weight, binsize=binsize, nlags=nlags)


Xs, Y = session.format_nlags(omitLaserBounds, binsize=binsize,
        nlags=nlags)

Xs_ctrl, Y_ctrl = session.format_nlags(bounds=ctrl_bounds, binsize=binsize,
    nlags=nlags)

Xs_both, Y_both = session.format_nlags(bounds=both_bounds, binsize=binsize,
    nlags=nlags)

opto_output=[]
ctrl_output=[]
cctrl_output=[]

opto_asymps=[]
ctrl_asymps=[]
cctrl_asymps=[]

#run opto first

reg1_start = 1#3 3 for CU, .5 for CO
reg2_start=.1

target=-2
threshold=.15
reg1_increment=.05 #usually .05, but trying .03 for co9_1215
reg2_increment=.05
continue_counter=0
reg1=reg1_start
reg2=reg2_start
opto_output=[]
curr_time = time.time()
change_reg1=True
find_start=True

drive_list=[]

opto_r2_reg2 = asymps(Xs_both, Y_both, sws, region=1)
opto_r2_reg1 = asymps(Xs_both, Y_both, sws, region=0)
opto_asymps.append((opto_r2_reg2, opto_r2_reg1))

while target < 2:
    if find_start:
        continue_counter=0

    if reg2 > reg1_start+1:
        break
    if change_reg1:
       if reg1 <= reg1_increment:
        print('turn!!!!')
        drive_list=[]
        reg2 = 0
        reg1 = reg2_start
        change_reg1=False

    regs=(reg1, reg2)
    print(regs)
    opto_output.append(analyze(regs, Xs_both, Y_both, sws))
    
    drive_temp = opto_output[-1][0]
    print(f'{session_name}:, score: {opto_output[-1][1]}, drive:{drive_temp}')
    drive_list.append(drive_temp)
    diff = 1/(np.abs(drive_temp - (target+threshold)))
    diff = min(diff, 10)
    print(diff)

    if drive_temp>target:
        if abs(drive_temp - target) < threshold:
            continue_counter=0
            if find_start:
                find_start = False
            print('success')
            target = find_lowest(drive_list, drive_temp, threshold) 
            print(target)
            print(drive_temp)
            diff = 1/(np.abs(drive_temp - (target+threshold)))
            diff = min(diff, 10)
            if change_reg1:
                reg1 -= reg1_increment / diff
                reg1 = np.abs(reg1)
            else:
                reg2 += reg2_increment / diff
        else:
            continue_counter+=1
            print('too big!!')
            if change_reg1:
                reg1 += reg1_increment/diff
            else:
                reg2 -= reg2_increment/diff
                reg2 = np.abs(reg2)
    else:
        continue_counter+=1
        print('too small')
        if change_reg1:
            reg1 -= reg1_increment/diff
            reg1 = np.abs(reg1)
        else:
            reg2 += reg2_increment/diff
    if continue_counter > 7:
        print('FORCE continue!')
        target = target + threshold
        continue_counter=0
    loop_time = time.time()
    elapsed = loop_time - curr_time
    print(f'time: {elapsed:.2f}') 

target=-1
continue_counter=0
reg1=reg1_start
reg2=reg2_start

curr_time = time.time()
change_reg1=True
find_start=True

drive_list=[]
ctrl_r2_reg2 = asymps(Xs_ctrl, Y_ctrl, sws=None, region=1)
ctrl_r2_reg1 = asymps(Xs_ctrl, Y_ctrl, sws=None, region=0)
ctrl_asymps.append((ctrl_r2_reg2, ctrl_r2_reg1))
print('ctrl!')
while target < 2:
    if find_start:
        continue_counter=0

    if reg2 > reg1_start+1:
        break
    if change_reg1:
        if reg1 <= reg1_increment:
            drive_list=[]
            reg2 = 0
            reg1 = reg2_start
            change_reg1=False

    regs=(reg1, reg2)
    print(regs)
    ctrl_output.append(analyze(regs, Xs_ctrl, Y_ctrl, sws=None))
    
    drive_temp = ctrl_output[-1][0]
    drive_list.append(drive_temp)
    print(f'{session_name}:, score: {ctrl_output[-1][1]}, drive:{drive_temp}')
    diff = 1/(np.abs(drive_temp - (target+threshold)))
    diff = min(diff, 10)
    print(diff)

    if drive_temp>target:
        if abs(drive_temp - target) < threshold:
            continue_counter=0
            print('success')
            if find_start:
                find_start=False
            target = find_lowest(drive_list, drive_temp, threshold) 
            diff = 1/(np.abs(drive_temp - (target+threshold)))
            diff = min(diff, 10)
            if change_reg1:
                reg1 -= reg1_increment/diff
                reg1 = np.abs(reg1)
            else:
                reg2 += reg2_increment/diff
        else:
            continue_counter+=1
            print('too big!!')
            if change_reg1:
                reg1 += reg1_increment/diff
            else:
                reg2 -= reg2_increment/diff
                reg2 = np.abs(reg2)
    else:
        continue_counter+=1
        print('too small')
        if change_reg1:
            reg1 -= reg1_increment/diff
            reg1 = np.abs(reg1)
        else:
            reg2 += reg2_increment/diff
    if continue_counter > 7:
        print('FORCE continue!')
        continue_counter=0
        target = target+threshold
    loop_time = time.time()
    elapsed = loop_time - curr_time
    print(f'time: {elapsed:.2f}') 

target=-1
continue_counter=0
reg1=reg1_start
reg2=reg2_start

curr_time = time.time()
change_reg1=True
find_start=True

drive_list=[]

cctrl_r2_reg2 = asymps(Xs_ctrl, Y_ctrl, sws=cws, region=1)
cctrl_r2_reg1 = asymps(Xs_ctrl, Y_ctrl, sws=cws, region=0)
cctrl_asymps.append((cctrl_r2_reg2, cctrl_r2_reg1))

print('cctrl!')
while target < 2:
    if find_start:
        continue_counter=0

    if reg2 > reg1_start+1:
        break
    if change_reg1:
        if reg1 <= reg1_increment:
            drive_list=[]
            reg2 = 0
            reg1 = reg2_start
            change_reg1=False

    regs=(reg1, reg2)
    print(regs)
    cctrl_output.append(analyze(regs, Xs_ctrl, Y_ctrl, sws=cws))
    
    drive_temp = cctrl_output[-1][0]
    drive_list.append(drive_temp)
    print(f'{session_name}:, score: {cctrl_output[-1][1]}, drive:{drive_temp}')
    diff = 1/(np.abs(drive_temp - (target+threshold)))
    diff = min(diff, 10)
    print(diff)

    if drive_temp>target:
        if abs(drive_temp - target) < threshold:
            if find_start:
                find_start=False
            continue_counter=0
            print('success')
            target = find_lowest(drive_list, drive_temp, threshold) 
            diff = 1/(np.abs(drive_temp - (target+threshold)))
            diff = min(diff, 10)
            if change_reg1:
                reg1 -= reg1_increment/diff
                reg1 = np.abs(reg1)
            else:
                reg2 += reg2_increment/diff
        else:
            continue_counter+=1
            print('too big!!')
            if change_reg1:
                reg1 += reg1_increment/diff
            else:
                reg2 -= reg2_increment/diff
                reg2 = np.abs(reg2)
    else:
        continue_counter+=1
        print('too small')
        if change_reg1:
            reg1 -= reg1_increment/diff
            reg1 = np.abs(reg1)
        else:
            reg2 += reg2_increment/diff
    if continue_counter > 7:
        print('FORCE continue!')
        continue_counter=0
        target = target+threshold
    loop_time = time.time()
    elapsed = loop_time - curr_time
    print(f'time: {elapsed:.2f}') 

temp_output = (opto_output, ctrl_output, cctrl_output)
asymps = (opto_asymps, ctrl_asymps, cctrl_asymps)
output = (temp_output, asymps)
print('finished!')
pdump(output, 
        f'../../picklejar/NLSweeps/{script_name}_{session_name}.pickle')
