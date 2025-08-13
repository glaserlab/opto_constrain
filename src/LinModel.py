import numpy as np
from src.MPOpto import *
from src.experiment import *
from src.regression import *
from src.analysis import *
from sklearn.decomposition import PCA

class LinModel(MPOpto):
    def __init__(self, session_path):
        super().__init__(session_path)


    def format_nlags(self, bounds=None, binsize=10, nlags=10, num_PCs=None,
            smooth=None, zscore=False):
        #nasty,nasty function
        #PCs can be None (no PCA), a number (fit to this number), or a tuple
        #of PCAObjects
        if bounds is None:
            bounds = self.climbing_bounds

        if smooth is not None:
            assert isinstance(smooth, int), 'pass in smooth kernel if smooth'
            mod_bounds, edgesmooth = self._mod_bounds(bounds, smooth)
            bounds = mod_bounds


        region1_binned, region2_binned = self.binner(bounds=bounds, binsize=binsize,
                concat=True)

        if num_PCs is None:
            Xs = np.hstack((region1_binned, region2_binned))
            PCA_output = None
            print('no PCA')
        else:
            if isinstance(num_PCs, tuple):
                reg1_PCAObj = num_PCs[0]
                reg2_PCAObj = num_PCs[1]
                X1 = reg1_PCAObj.transform(region1_binned)
                X2 = reg2_PCAObj.transform(region2_binned)
                PCA_output = (reg1_PCAObj, reg2_PCAObj)
                print('applying PCA')
            else:
                reg1_PCAObj = PCA(n_components = num_PCs)
                reg2_PCAObj = PCA(n_components = num_PCs)
                X1 = reg1_PCAObj.fit_transform(region1_binned)
                X2 = reg2_PCAObj.fit_transform(region2_binned)
                PCA_output = (reg1_PCAObj, reg2_PCAObj)
                print('fitting and applyingPCA')
            Xs = np.hstack((X1, X2))

        if zscore:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xs)
            self.scaler = scaler
        
        seams = getSeamsFromBounds(bounds, binsize=binsize)
        Xs = unstitchSeams(Xs, seams)

        if smooth is not None:
            Xs = self._smooth_in_modbounds(Xs, sigma=smooth, binsize=binsize,
                    edgesmooth=edgesmooth, smooth_type='future')

        all_nlags, all_cut  = format_and_stitch_ar(Xs, nlags=nlags)
        
        return (all_nlags, all_cut), PCA_output


    def generate_trainset(self, bounds, binsize, nlags, num_PCs, sigma_mult=1,
            zscore=False):
        #diya modifying stupidly for one thing

        (X, Y), PCAObjs = self.format_nlags(bounds=bounds, 
                binsize=binsize, nlags=nlags, num_PCs=num_PCs, zscore=zscore)
        if zscore:
            zscore= self.scaler
        #(_, Y),_ = self.format_nlags(bounds=bounds, binsize=binsize, nlags=nlags, 
        #        num_PCs=PCAObjs, smooth=binsize*sigma_mult, zscore=zscore)
        if num_PCs is None:
            Y=Y[:, :self.num_region1]
        else:
            Y=Y[:,:num_PCs]

        return (X, Y), PCAObjs

    def generate_testset(self, bounds, binsize, nlags, PCAObjs=None,
            sigma_mult=1, zscore=False):
        #diya modifying stupidly for one thing
        (X, Y), _= self.format_nlags(bounds=bounds, 
                binsize=binsize, nlags=nlags, num_PCs=PCAObjs, zscore=zscore)
        #(_, Y), _ = self.format_nlags(bounds=bounds, binsize=binsize, 
        #        nlags=nlags, num_PCs=PCAObjs, smooth=binsize*sigma_mult,
        #        zscore=zscore)
        if PCAObjs is None:
            Y=Y[:, :self.num_region1]
        else:
            Y=Y[:,:PCAObjs[0].n_components]

        return X, Y

    def getSWs(self, bigBound, smallBound, sw_weight=50, binsize=10, nlags=10):
        # confusing, but if we're grabbing data in bigBound
        # and we want to weight extra stuff in smallBound
        # which i guess is a subset of all in bigBOund
        # then run this, it'll give a logical that should be
        # the same duration as whatever operation in BigBOund

        sw_list = []

        mod_smallBound, duration  = reBoundInBounds(bigBound, smallBound)
        mod_smallBound_logical = bounds2Logical(mod_smallBound, duration=duration)
        mod_smallBound_trials = unstitchSeams(mod_smallBound_logical,
                getSeamsFromBounds(bigBound, binsize=1))

        for trial in mod_smallBound_trials:
            temp = bin_timeseries(trial, binsize=binsize)
            sw_list.append(temp[nlags:])
        sw = np.hstack(sw_list)
        sw[sw >= 1] = sw_weight - 1
        sw = sw+1

        return sw

    def get_relative_drive(self, h, X_format, nlags, num_region1=None):
        if num_region1==None:
            num_region1 = self.num_region1
        region1_mask, region2_mask = mask_input_region(X_format, nlags, num_region1)

        region1 = X_format[:, region1_mask]
        region2 = X_format[:, region2_mask]

        region1_h_mask, region2_h_mask = mask_h_region(h, nlags,
                num_region1, keep_bias=False)
        
        region1_h = h[region1_h_mask, :]
        region2_h = h[region2_h_mask, :]

        region1_drive = np.sum(np.abs(region1@region1_h))
        region2_drive = np.sum(np.abs(region2@region2_h))

        return np.log10(region1_drive/region2_drive)



    def get_region_predics(self, h, X_format, nlags, num_region1):
        region1_mask, region2_mask = mask_input_region(X_format, nlags, num_region1)

        region1 = X_format[:, region1_mask]
        region2 = X_format[:, region2_mask]

        region1_h_mask, region2_h_mask = mask_h_region(h, nlags,
                num_region1, keep_bias=False)
        
        region1_h = h[region1_h_mask, :]
        region2_h = h[region2_h_mask, :]

        region1_drive = region1@region1_h
        region2_drive = region2@region2_h

        return region1_drive, region2_drive

    def get_relative_drive_noself(self, h, X_format, nlags, num_region1=None):
        if num_region1==None:
            num_region1 = self.num_region1
        h_noself = copy.deepcopy(h)
        features = X_format.shape[1]
        num_neurons = int(features / nlags)
        for idx in np.arange(num_region1):
            which_neuron = np.arange(features) % num_neurons
            which_neuron_mask = which_neuron == idx
            h_noself[1:, idx][which_neuron_mask] = 0

        region1_mask, region2_mask = mask_input_region(X_format, nlags, num_region1)

        region1 = X_format[:, region1_mask]
        region2 = X_format[:, region2_mask]

        region1_h_mask, region2_h_mask = mask_h_region(h_noself, nlags,
                num_region1, keep_bias=False)
        
        region1_h = h_noself[region1_h_mask, :]
        region2_h = h_noself[region2_h_mask, :]

        region1_drive = np.sum(np.abs(region1@region1_h))
        region2_drive = np.sum(np.abs(region2@region2_h))

        return np.log10(region1_drive/region2_drive)



    def get_region_predics_noself(self, h, X_format, nlags, num_region1):
        h_noself = copy.deepcopy(h)
        features = X_format.shape[1]
        num_neurons = int(features / nlags)
        for idx in np.arange(num_region1):
            which_neuron = np.arange(features) % num_neurons
            which_neuron_mask = which_neuron == idx
            h_noself[1:, idx][which_neuron_mask] = 0

        region1_mask, region2_mask = mask_input_region(X_format, nlags, num_region1)

        region1 = X_format[:, region1_mask]
        region2 = X_format[:, region2_mask]

        region1_h_mask, region2_h_mask = mask_h_region(h_noself, nlags,
                num_region1, keep_bias=False)
        
        region1_h = h_noself[region1_h_mask, :]
        region2_h = h_noself[region2_h_mask, :]

        region1_drive = region1@region1_h
        region2_drive = region2@region2_h

        return region1_drive, region2_drive




