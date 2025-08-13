import numpy as np
from src.utils.gen_utils import *
from src.utils.filters import *
from src.load import *
from src.experiment import *
from scipy.ndimage import gaussian_filter1d
from src.neural import *
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

class MPOpto:
    ''' MPOpto designed to handle data with two region recordings and
    stimming one of the regions

    session_path = MOUSE_SESSIONDATE, like co6_10122023, or the CU path

    updating  
    '''
    def __init__(self, session_path):
        yaml_data = load_yaml(f'{session_path}/session_info.yaml')
        self.name = yaml_data['session_name']
        if 'rates' in yaml_data:
            self.binner = self.binner_rates
            self.rate_model = True

            print('this is a rate model')
        else:
            self.binner = self.binner_spikes
            self.rate_model=False
            print('this is a spike train model')

        loader = yaml_data['loader']
        if loader == 'CO':

            regions = yaml_data['record']
            stim_regions = yaml_data['stim']

            assert len(regions)==2, 'this class for two region only'
            assert len(stim_regions)==1, 'this class for one stim region only'
            assert stim_regions[0] == regions[1], 'second region should be stim region'

            laser_channel = yaml_data['analogin']['stim_pulse']
            ctrl_channel = yaml_data['analogin']['ctrl_pulse']

            load_neural = load_co_neural
            laser_thres=.1

            analogin = load_analogin(session_path)
            isclimbing = load_climb(session_path)

            self.region1 = load_neural(session_path, regions[0])
            self.region2 = load_neural(session_path, regions[1])

            self.num_region1 = self.region1['num_neurons']
            self.num_region2 = self.region2['num_neurons']

            self.inactivate = yaml_data['record'][-1]
            self.duration = analogin.shape[1] 

            self.climbing_bounds = climbing_bounds_from_logical(isclimbing)
            self.climbing_logical = np.squeeze(isclimbing.astype(int))

            self.laser = {}
            self.laser['laser_ts'] = analogin[laser_channel,:]
            self.laser['ctrl_ts'] = analogin[ctrl_channel,:]
            self.laser['laser_bounds'], self.laser['laser_powers'] =\
                    getLaserBounds(self.laser['laser_ts'], laser_thres)

            self.laser['ctrl_bounds'], nada =\
                    getLaserBounds(self.laser['ctrl_ts'], laser_thres)

            self.num_trials = self.laser['laser_bounds'].shape[0]
            self.num_ctrls = self.laser['ctrl_bounds'].shape[0]

            print('love, finished loading a co mouse')
        else:
            picklepath=f'{session_path}/{self.name}.pickle'
            d = pload(picklepath)
            self.model = d['model']
            m = d['model'] #quick access
            self.J = m['J']
            regions = m['regions']
            regions = m['regions']

            self.num_region1 = len(regions['region1'])
            self.idx_region1 = regions['region1']
            self.num_region2 = len(regions['region2'])
            self.idx_region2 = regions['region2']

            self.region1={}
            self.region2={}
            self.laser={}

            self.true_rates = d['true']

            self.region1['train'] = d['region1']
            self.region2['train'] = d['region2']
            self.duration = d['duration'] 
            self.climbing_bounds = np.array([[0, self.duration-300]])

            self.laser['laser_bounds'] = d['opto_bounds']
            ctrl_bounds = deepcopy(d['opto_bounds'])
            jitter = np.random.choice(np.arange(400, 1000),
                    size=len(ctrl_bounds), replace=True)
            ctrl_bounds[:,0] = ctrl_bounds[:,0] + jitter
            ctrl_bounds[:,1] = ctrl_bounds[:,1] + jitter
            self.laser['ctrl_bounds'] = ctrl_bounds

            print('finished loading curbd sim data')


    def threshold_FRs(self, threshold=.5, bounds=None, overwrite=False):
        ''' keep only neurons over threshold in hz. if overwrite, "deletes" the
        removed neurons from raw data, if not overwrite, just returns mask
        '''
        if bounds is None:
            bounds = self.climbing_bounds


        region1, region2 = self.binner(binsize=1, bounds=bounds, concat=True)

        region1_hz = (np.sum(region1, axis=0) / region1.shape[0]) * 1000
        region2_hz = (np.sum(region2, axis=0) / region2.shape[0]) * 1000

        if overwrite:
            mask = region1_hz > threshold
            keeps = np.arange(self.num_region1)[mask]
            new_reg1 = [self.region1['train'][idx] for idx in keeps]
            new_reg1_width = np.array(self.region1['width'])[keeps]
            try:
                new_reg1_depth = np.array(self.region1['depth'])[keeps]

                self.region1['depth'] = new_reg1_depth.tolist()
            except:
                print('no dpeth')

            mask = region2_hz > threshold
            keeps = np.arange(self.num_region2)[mask]
            new_reg2 = [self.region2['train'][idx] for idx in keeps]
            new_reg2_width = np.array(self.region2['width'])[keeps]
            try:
                new_reg2_depth = np.array(self.region2['depth'])[keeps]
                self.region2['depth'] = new_reg2_depth.tolist()
            except:
                print('no dpeth')



            print(f'new num region1: {len(new_reg1)}')
            print(f'new num region2: {len(new_reg2)}')

            self.region1['train'] = new_reg1
            self.region1['num_neurons'] = len(new_reg1)
            self.region1['width'] = new_reg1_width.tolist()
            self.num_region1 = len(new_reg1)
            self.region2['train'] = new_reg2
            self.region2['num_neurons'] = len(new_reg2)
            self.region2['width'] = new_reg2_width.tolist()
            self.num_region2 = len(new_reg2)

            return

        return region1_hz > threshold, region2_hz > threshold

    def subsampleNeurons(self, percent_region1, percent_region2=None,
            random_state=None):
        '''randomly only keeps PERCENT neurons. always overwrites atm         
        '''

        rng = np.random.default_rng(seed=random_state)
        if percent_region2 is None:
            percent_region2 = percent_region1
        subsamp_region1 = rng.choice(self.num_region1, 
                    size=int(percent_region1*self.num_region1), replace=False)
        subsamp_region2 = rng.choice(self.num_region2, 
                int(percent_region2*self.num_region2), replace=False)

        self.region1['orig_train'] = self.region1['train']
        self.region2['orig_train'] = self.region2['train']

        if percent_region1 < 1:
            region1_newtrain = [self.region1['train'][idx] for idx in
                    subsamp_region1]
        else:
            region1_newtrain = self.region1['train']
        if percent_region2<1:
            region2_newtrain = [self.region2['train'][idx] for idx in
                    subsamp_region2]
        else:
            region2_newtrain = self.region2['train']

        self.region1['train'] = region1_newtrain
        self.region2['train'] = region2_newtrain
        self.num_region1 = len(region1_newtrain)
        self.num_region2 = len(region2_newtrain)

        print(f'new num region1: {self.num_region1}')
        print(f'new num region2: {self.num_region2}')

        return

    def adjustLaserBounds(self, pre, post, only_climb=False):
        ''' new lasers are start-pre, end+post
        '''
        #need to fix good lasers at some point
        laser_bounds = adjustBounds(self.laser['laser_bounds'], pre, post)
        ctrl_bounds = adjustBounds(self.laser['ctrl_bounds'], pre, post)

        if only_climb:
            laser_mask = goodLasers(laser_bounds, self.climbing_logical)
            ctrl_mask = goodLasers(ctrl_bounds, self.climbing_logical)
            laser_bounds = laser_bounds[laser_mask, :]
            ctrl_bounds = ctrl_bounds[ctrl_mask, :]

        return laser_bounds, ctrl_bounds

    def getInactNeurons(self, pre=0, post=0, threshold=.1):
        baseline_bounds, _ = self.adjustLaserBounds(pre=-15, post=0)
        _, bsl_ups = self.binner(binsize=3, bounds=baseline_bounds, concat=True)
        scaler = StandardScaler()
        scaler.fit(bsl_ups)
        laser_bounds, _ = self.adjustLaserBounds(pre, post)
        _, ups = self.binner(binsize=3, bounds=laser_bounds, concat=False)
        ups_avg = np.average(ups, axis=0)
        ups_scaled = scaler.transform(ups_avg)
        ups_excit = np.any(ups_scaled > threshold, axis=0)

        return ups_excit, ~ups_excit

    def binner_rates(self, bounds=None, binsize=1, concat=False):
        if bounds is None:
            bounds = np.array([[0, self.duration]])
            concat=True
        else:
            assert bounds.shape[1] == 2, 'bounds looks weird'

        region1_binned = bin_neural_bounds(self.region1['train'], binsize, bounds)
        region2_binned = bin_neural_bounds(self.region2['train'], binsize, bounds)
        if concat:
            region1_binned = np.vstack(region1_binned)
            region2_binned = np.vstack(region2_binned)
        return region1_binned, region2_binned

    def binner_spikes(self, bounds=None, binsize=1, concat=False):

        if bounds is None:
            bounds = np.array([[0, self.duration]])
            concat=True
        else:
            assert bounds.shape[1] == 2, 'bounds looks weird'

        region1_binned = bin_spiketimes_bounds(self.region1['train'], binsize, bounds)
        region2_binned = bin_spiketimes_bounds(self.region2['train'], binsize, bounds)
        if concat:
            region1_binned = np.vstack(region1_binned)
            region2_binned = np.vstack(region2_binned)
        return region1_binned, region2_binned

    def smoother(self, bounds=None, sigma=10, binsize=1, concat=False,
            smooth_type='causal, future, or anything for twosided'):
        # some assumptions, but lets bin/smooth all data, and then truncate
        #concat only works if bounds are even, will fix later
        #edgesmooth = 100//binsize #we smooth a little past bounds, and then remove
        assert sigma%binsize==0, 'sigma must be multiple of binsize'

        if bounds is None:
            print('smoothing entire thing')
            region1_binned, region2_binned = self.binner(binsize=binsize, concat=True)

            if smooth_type == 'causal':
                gausmooth = gaussian_filter1d_oneside
            elif smooth_type == 'future':
                gausmooth = gaussian_filter1d_future
            else:
                gausmooth = gaussian_filter1d

            region1_smooth = gausmooth(region1_binned, sigma=sigma//binsize, axis=0,
                    mode='constant')

            region2_smooth = gausmooth(region2_binned, sigma=sigma//binsize, axis=0, 
                    mode='constant')

            return region1_smooth, region2_smooth

        mod_bounds, edgesmooth = self._mod_bounds(bounds, sigma)
        region1_binned, region2_binned = self.binner(bounds=mod_bounds, 
                binsize=binsize, concat=False)

        region1_smooth = self._smooth_in_modbounds(region1_binned, sigma, binsize,
                edgesmooth, smooth_type)
        region2_smooth = self._smooth_in_modbounds(region2_binned, sigma, binsize,
                edgesmooth, smooth_type)

        if concat:
            region1_smooth = np.vstack(region1_smooth)
            region2_smooth = np.vstack(region2_smooth)

        return region1_smooth, region2_smooth

    def _mod_bounds(self, bounds, sigma):
        duration = self.duration
        edgesmooth = sigma*2

        mod_bounds = copy.deepcopy(bounds)
        if mod_bounds[0,0] - edgesmooth < 0:
            #if too near the end, none of this extra smoothing
            edgesmooth=0
        if mod_bounds[-1,1] + edgesmooth > duration:
            #same with too near the beginning!
            edgesmooth=0
        mod_bounds[:,0]  = mod_bounds[:,0] - edgesmooth
        mod_bounds[:,1] = mod_bounds[:,1] + edgesmooth

        return mod_bounds, edgesmooth

    def _smooth_in_modbounds(self, neural_data, sigma, binsize,edgesmooth, smooth_type):

        if smooth_type == 'causal':
            gausmooth = gaussian_filter1d_oneside
        elif smooth_type == 'future':
            gausmooth = gaussian_filter1d_future
        else:
            gausmooth = gaussian_filter1d

        edgesmooth_bin = edgesmooth//binsize
        output = []

        for trial_num in range(len(neural_data)):
            trial = neural_data[trial_num]
            temp = gausmooth(trial, sigma=sigma//binsize,
                    axis=0, mode='constant')

            if edgesmooth > 0:
                temp = temp[edgesmooth_bin:-edgesmooth_bin]
            output.append(temp)

        return output

    def generate_new_ctrls(self, num_trials, pre, post=50, omit_post=250):
        _, ctrl_bounds = self.adjustLaserBounds(pre, post)
        laser_bounds, _ = self.adjustLaserBounds(pre, omit_post)
        bounds = omitBoundInBounds(self.climbing_bounds, laser_bounds)
        bounds = omitBoundInBounds(bounds, ctrl_bounds)

        trial_length = ctrl_bounds[0][1] - ctrl_bounds[0][0]

        bounds = minimumBoundSize(bounds, min_size=trial_length)


        num_bouts = len(bounds)
        rng = np.random.default_rng(1)
        temp =rng.choice(num_bouts, size=num_trials, replace=True)
        random_bouts_idxs, counts = np.unique(temp, return_counts=True)
        random_bouts = bounds[random_bouts_idxs, :]

        start_idx_list = []

        for idx, bout in enumerate(random_bouts):
            start = bout[0]
            end = bout[1]-trial_length

            for c in np.arange(counts[idx]):
                start_idx = rng.choice(np.arange(start, end))
                start_idx_list.append(start_idx)

        output = np.zeros((num_trials, 2))
        output[:,0] = start_idx_list
        output[:,1] = [x + trial_length for x in start_idx_list] 

        return np.sort(output, axis=0).astype(int)
