import numpy as np
from src.utils.gen_utils import *
from src.regression import *
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
    
def weights_per_region(h, nlags, num_region1=None, var=None):

        #move to analysis.md, soon to be deprecated
        if num_region1 is None:
            num_region1 = self.num_region1
        region1_mask, region2_mask = mask_h_region(h, nlags, num_region1,
                keep_bias=False)
        region1 = np.sum(np.abs(h[region1_mask,:]), axis=0)
        region2 = np.sum(np.abs(h[region2_mask,:]), axis=0)

        r1 = np.average(region1, weights=var)
        r2 = np.average(region2, weights=var)

        return r1, r2


def relative_input(h, X_format, nlags, num_region1,var=None):
        #probably should be redone, and moved to analysis.md
        region1_mask, region2_mask = mask_input_region(X_format, nlags, num_region1)

        region1 = X_format[:, region1_mask]
        region2 = X_format[:, region2_mask]

        region1_h_mask, region2_h_mask = mask_h_region(h, nlags,
                num_region1, keep_bias=False)
        
        region1_h = h[region1_h_mask, :]
        region2_h = h[region2_h_mask, :]

        region1_drive = np.sum(np.abs(region1@region1_h),axis=0)
        region2_drive = np.sum(np.abs(region2@region2_h),axis=0)

        return np.average(region1_drive, weights=var), np.average(region2_drive, weights=var)

def get_top_models(ratio_vector, r2_vector, edges=None, num_bins=15,
        normalize=False, evenspace=False):
    if edges is None:
        _, edges = np.histogram(ratio_vector, bins=num_bins)
    if type(ratio_vector) is list:
        ratio_vector = np.array(ratio_vector)
    if type(r2_vector) is list:
        r2_vector = np.array(r2_vector)

    idxs = np.arange(len(ratio_vector))

    ratio_keep = []
    r2_keep = []
    idxs_keep = [] 

    
    for i in range(edges.size-1):
        condition1 = ratio_vector >= edges[i]
        condition2 = ratio_vector < edges[i+1]
        indices = np.argwhere(np.logical_and(condition1,condition2))
        if indices.size < 1:
            #r2_keep.append([np.nan])
            #ratio_keep.append([np.nan])
            continue
        
        keep = np.argmax(r2_vector[indices])
        temp1 = r2_vector[indices]
        temp2 = temp1[keep]
        r2_keep.append(temp2)

        if not evenspace:
            temp1 = ratio_vector[indices]
            temp2 = temp1[keep]
            ratio_keep.append(temp2)
        else:
            ratio_keep.append(np.average([edges[i], edges[i+1]]))

        temp1 = idxs[indices]
        temp2 = temp1[keep]
        idxs_keep.append(temp2)

    ratio_keep, r2_keep, idxs_keep = squeezer(ratio_keep, r2_keep,
            idxs_keep)

    sort_idxs = ratio_keep.argsort()
    sort_ratio = ratio_keep[sort_idxs]
    sort_r2 = r2_keep[sort_idxs]
    sort_idxs = idxs_keep[sort_idxs]
    if normalize:
        sort_r2 = normalizeData(sort_r2)
    return sort_ratio, sort_r2, sort_idxs, edges 

def get_median_models(ratio_vector, r2_vector, edges=None, num_bins=20,
        evenspace=False):
    if edges is None:
        _, edges = np.histogram(ratio_vector, bins=num_bins)

    bin_width=4/num_bins
    overlap = 0.25
    stepsize = bin_width * (1-overlap)
    bin_starts = np.arange(-2, 2, stepsize)
    bins = [[b, b + bin_width] for b in bin_starts]
    if type(ratio_vector) is list:
        ratio_vector = np.array(ratio_vector)
    if type(r2_vector) is list:
        r2_vector = np.array(r2_vector)

    idxs = np.arange(len(ratio_vector))

    ratio_med = []
    r2_med = []
    #idxs_keep = [] 

    
    for edges in bins:
        condition1 = ratio_vector >= edges[0]
        condition2 = ratio_vector < edges[1]

        indices = np.argwhere(np.logical_and(condition1,condition2))
        if indices.size < 1:
            #r2_keep.append([np.nan])
            #ratio_keep.append([np.nan])
            continue
        

        r2_med.append(np.median(r2_vector[indices]))
        if not evenspace:
            ratio_med.append(np.median(ratio_vector[indices]))
        else:
            ratio_med.append(np.average([edges[i], edges[i+1]]))

    ratio_med = np.array(ratio_med)
    r2_med = np.array(r2_med)

    sort_idxs = ratio_med.argsort()
    sort_med = ratio_med[sort_idxs]
    sort_r2 = r2_med[sort_idxs]



    return sort_med, sort_r2, edges 


def trial_sem(neural, fs=1, pre_baseline=None, sem = 'neurons'):

    #trials x time x neurons
    if sem == 'neurons':
        num_sem = neural.shape[2]
        temp = np.average(neural, axis=0)
        neural_avgsem = np.std(temp, axis=1) / np.sqrt(num_sem) / (fs/1000)
    else:
        num_sem = neural.shape[0]
        temp = np.average(neural, axis=2)
        neural_avgsem = np.std(temp, axis=0) / np.sqrt(num_sem) / (fs/1000)
    neural_avgavg = np.average(neural, axis=(0,2)) / (fs/1000)
    #temp = np.average(neural, axis=2)

    if pre_baseline is not None:
        neural_bs = np.average(neural[:,:pre_baseline,:]) / (fs/1000)
        neural_avgavg = neural_avgavg-neural_bs


    return neural_avgavg, neural_avgsem

def trial_sem_fractional(neural, fs=1, pre_baseline=None, sem = 'neurons'):

    #trials x time x neurons
    if sem == 'neurons':
        num_sem = neural.shape[2]
        temp = np.average(neural, axis=0)
        neural_avgsem = np.std(temp, axis=1) / np.sqrt(num_sem) / (fs/1000)
    else:
        num_sem = neural.shape[0]
        temp = np.average(neural, axis=2)
        neural_avgsem = np.std(temp, axis=0) / np.sqrt(num_sem) / (fs/1000)
    neural_avgavg = np.average(neural, axis=(0,2)) / (fs/1000)
    #temp = np.average(neural, axis=2)

    if pre_baseline is not None:
        neural_bs = np.average(neural[:,:pre_baseline,:]) / (fs/1000)
        neural_avgavg = (neural_avgavg-neural_bs) / neural_bs


    return neural_avgavg, neural_avgsem



def trial_sem_multisession(neural_list, fs=1, pre_baseline=None):
    new_list = []
    num_trials = len(neural_list)
    curr_neurons = 0
    num_neurons=0
    for trial in neural_list:
        if trial.shape[1] != curr_neurons:
            curr_neurons = trial.shape[1]
            num_neurons+= curr_neurons
        new_list.append(np.average(trial, axis=1))
    avg = np.average(new_list, axis=0) / (fs/1000)
    sem = np.std(new_list, axis=0) / np.sqrt(num_trials) / (fs/1000)

    if pre_baseline is not None:
        temp = np.array(new_list)
        bs = np.average(temp[:, :pre_baseline]) / (fs/1000)
        avg = avg - bs


    return avg, sem

def trial_sem_multisession_fractional(neural_list, fs=1, pre_baseline=None):
    new_list = []
    num_trials = len(neural_list)
    curr_neurons = 0
    num_neurons=0
    for trial in neural_list:
        if trial.shape[1] != curr_neurons:
            curr_neurons = trial.shape[1]
            num_neurons+= curr_neurons
        new_list.append(np.average(trial, axis=1))
    #return new_list
    temp = np.array(new_list)
    bs = np.average(temp[:, :pre_baseline])
    avg = np.average(new_list, axis=0)
    avg = (avg - bs) / bs
    sem = np.std(avg, axis=0) / np.sqrt(num_neurons)

    #if pre_baseline is not None:
    #    temp = np.array(new_list)
    #    bs = np.average(temp[:, :pre_baseline]) / (fs/1000)
    #    avg = (avg - bs)/ bs

    return avg, sem





def collapse_h(h, nlags, var=None, normalize=False):
        features = h.shape[0]
        mod_test = features % nlags
        if mod_test == 1:
            print('this h has bias')
            features=features-1
            has_bias=True
        elif mod_test ==0:
            print('this h has no bias term already')
            has_bias=False
        else:
            print('i think an error has happened')
            return 0
        repeats = features / nlags
        assert features % repeats ==0, 'something wrong'
        if has_bias:
            new_h = np.abs(h[1:,:])
        else:
            new_h = np.abs(h)
        temp = np.arange(features) % repeats
        collapse_list = []

        for neuron in np.arange(repeats):
            collapse_list.append(np.average(np.sum(new_h[temp==neuron,:],
                axis=0), weights=var))

        if normalize:
            return normalizeData(np.array(collapse_list))
        else:
            return np.array(collapse_list)

def collapse_h_lags(h, nlags, var=None, normalize=False):
        features = h.shape[0]
        mod_test = features % nlags
        if mod_test == 1:
            print('this h has bias')
            features=features-1
            has_bias=True
        elif mod_test ==0:
            print('this h has no bias term already')
            has_bias=False
        else:
            print('i think an error has happened')
            return 0


        num_neurons = features / nlags
        assert features % nlags ==0, 'something wrong'
        assert features % num_neurons == 0, 'something wrong'
        if has_bias:
            new_h = np.abs(h[1:,:])
        else:
            new_h = np.abs(h)
        collapse_list = []

        iterator = np.arange(nlags+1) * num_neurons
        iterator = iterator.astype(int)

        for idx in range(len(iterator)-1):
            start = iterator[idx]
            end = iterator[idx+1]
            collapse_list.append(np.average(np.sum(new_h[start:end,:],
                axis=0), weights=var))


        if normalize:
            return normalizeData(np.array(collapse_list))
        else:
            return np.array(collapse_list)

def split_collapse_h(h, nlags, num_region1, var=None, normalize=False):
    #normalize together by passing in tuple 
    reg1_mask, reg2_mask = mask_h_region(h, nlags, num_region1,
            keep_bias=False)
    h1 = h[reg1_mask,:]
    h2 = h[reg2_mask,:]

    if type(normalize) is not tuple:

        reg1_col = collapse_h_lags(h1, nlags, var, normalize=normalize)
        reg2_col = collapse_h_lags(h2, nlags, var, normalize=normalize)
    else:
        reg1_col = collapse_h_lags(h1, nlags, var, normalize=False)
        reg2_col = collapse_h_lags(h2, nlags, var, normalize=False)
        split_point = len(reg1_col)
        temp = normalizeData(np.concatenate((reg1_col, reg2_col)))
        reg1_col = temp[:split_point]
        reg2_col = temp[split_point:]

    return reg1_col, reg2_col



#def collapse_eigen(eigenvectors, num_region1):
    #for idx in np.arange(len(eigenvectors)):
     #   ev = eigenvectors[:,idx]


def mask_input_region(X_format, nlags, num_region1):
        features = X_format.shape[1]
        repeats = features/nlags
        assert features % repeats == 0, 'something wrong'

        temp =  np.arange(features) % repeats
        region1_mask = temp < num_region1
        region2_mask = temp >= num_region1 

        return region1_mask, region2_mask

def mask_h_region(h, nlags, num_region1, keep_bias=False):
        features = h.shape[0]
        mod_test = features % nlags
        if mod_test == 1:
            #print('this h has bias')
            features=features-1
            has_bias=True
        elif mod_test ==0:
            #print('this h has no bias term already')
            has_bias=False
        else:
            print('i think an error has happened')
            return 0

        repeats = features / nlags
        assert features % repeats ==0, 'something wrong'

        temp = np.arange(features) % repeats
        region1_mask = temp < num_region1
        region2_mask = temp >= num_region1

        if has_bias:

            region1_mask = np.hstack([keep_bias, temp < num_region1])
            region2_mask = np.hstack([keep_bias, temp >= num_region1])

        return region1_mask, region2_mask



def split_h_predics(X, h, nlags, num_region1):
    h1_mask, h2_mask = mask_h_region(h, nlags, num_region1, keep_bias=False)
    r1_mask, r2_mask = mask_input_region(X, nlags, num_region1)
    bias = h[0,:] 

    X1 = X[:,r1_mask]
    X2 = X[:,r2_mask]
    h1 = h[h1_mask,:]
    h2 = h[h2_mask,:]

    yhat1 = X1@h1
    yhat2 = X2@h2

    return yhat1, yhat2, bias

def divergence(obs1, obs2, PCAObject=None):
    assert obs1.shape[1] == obs2.shape[1], 'issue with size'
    if PCAObject==None:
        PCAObject1 = PCA(n_components=obs1.shape[1]).fit(obs1)
        PCAObject2 = PCA(n_components=obs2.shape[1]).fit(obs2)
    PCs1 = PCAObject1.transform(obs1)
    PCs2 = PCAObject2.transform(obs1)

    temp1 = _calc_divergence(PCs1, PCs2)


    PCs1 = PCAObject2.transform(obs2)
    PCs2 = PCAObject1.transform(obs2)

    temp2 = _calc_divergence(PCs1, PCs2)

    return (temp1+temp2)/2

def _calc_divergence(PCs1, PCs2):

        #helper function when calculating have two subspaces


    difference = np.abs(np.var(PCs2, axis=0) - np.var(PCs1, axis=0))
    summy = np.var(PCs2, axis=0) + np.var(PCs1, axis=0)
    weights = np.var(PCs1, axis=0)

    return np.average(difference / summy, weights=weights)

def invert_PCs(X, PCAObj):
    #Invert PCs after already discarding certain PCs.
    num_PCs = X.shape[1]
    X_og = X_og @ PCAObj.components_[:num_PCs,:]
    return X_og

def invert_H(H, PCAObj):
    num_PCs = H.shape[0]
    components = PCAObj.components_[:num_PCs,:]
    H_transform = components.T @ H
    return H_transform

def invert_H_tworegions(H, PCAObjs):
    reg1_PCAObj = PCAObjs[0]
    reg2_PCAObj = PCAObjs[1]

    num_PCs = H.shape[0] // 2
    #print(f'num_PCs:{num_PCs}')

    H_reg1 = H[:num_PCs, :]
    H_reg2 = H[num_PCs:, :]

    H_reg1_invert = invert_H(H_reg1, reg1_PCAObj)
    H_reg2_invert = invert_H(H_reg2, reg2_PCAObj)

    H_concat = np.concatenate((H_reg1_invert, H_reg2_invert))

    #print(H_concat.shape)

    return H_concat

def invert_collapse_h(h, PCAObjs, nlags):
    features = h.shape[0]
    mod_test = features % nlags
    if mod_test == 1:
        print('this h has bias')
        features=features-1
        has_bias=True
    elif mod_test ==0:
        print('this h has no bias term already')
        has_bias=False
    else:
        print('i think an error has happened')
        return 0
    repeats = features // nlags
    assert features % repeats ==0, 'something wrong'
    if has_bias:
        h = h[1:,:]

    inverted_h = []
    for i in np.arange(nlags):
        start = i*repeats
        end = (i+1)*repeats

        inverted_h.append(invert_H_tworegions(h[start:end, :], PCAObjs))
    return inverted_h
    #abs_h = np.abs(inverted_h)
	
    #return np.sum(abs_h, axis=(0,2))
