import numpy as np
import time
from src.analysis import *
from src.utils.evaluate import *
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import least_squares
from tqdm.auto import tqdm

def flatten_array(X):
    """
    Converting list containing multiple ndarrays into a large ndarray
    X: a list
    return: a numpy ndarray

    old ass code, i think np has a functionn for this but still using
    """
    n_col = np.size(X[0],1)
    Y = np.empty((0, n_col))
    for each in X:
        Y = np.vstack((Y, each))
    return Y

def format_data_ar(x, nlags=10):
    #not efficient but should work
    x_nlags = []
    for i in range(np.size(x, 0) - nlags):
        temp = x[i:i+nlags, :]
        temp = temp.reshape((np.size(temp)))
        x_nlags.append(temp)
    x_nlags = np.array(x_nlags)
    x_cut = x[nlags:, :]

    return x_nlags, x_cut

    
def format_and_stitch_ar(x, nlags=10):
    #format_data, working only on trials
    #if array, data must be trial_length x neurons x trials
    #if list, must be a list of trial_length x neurons np arrays
    assert type(x) is list, 'use format_data if just time series'
    num_trials = len(x)
    
    x_nlags = []
    x_cut = []
    for i in range(num_trials):
        temp_x = x[i]

        temp1, temp2 = format_data_ar(temp_x, nlags=nlags)
        if temp1.shape[0] > 0:
            x_nlags.append(temp1)
            x_cut.append(temp2)
        else:
            print('bounds too small for history')
    return flatten_array(x_nlags), flatten_array(x_cut)

def parameter_fit(x, y, c, sw=None, zscore=False):
    """
    c : L2 regularization coefficient
    I : Identity Matrix
    Linear Least Squares (code defaults to this if c is not passed)
    H = ( X^T * X )^-1 * X^T * Y
    Ridge Regression
    R = c * I
    ridge regression doesn't penalize x
    R[0,0] = 0
    H = ( (X^T * SW*X) + R )^-1 * X^T * SW*Y
    sw = saple weights
    """
    temp = np.c_[np.ones((np.size(x, 0), 1)), x]
    x_t = temp.T
    if sw is not None:
        assert sw.size == x.shape[0], f'something weird with sample\
        weight, sw size={sw.size}, x shape = {x.shape[0]}'
        
        #breakpoint()
        sw = sw / np.average(sw)
        x = sw[:, None] * x
        if y.ndim==1:
            #stupid...
            y = sw * y
        else:
            y = sw[:, None] * y
        #x = x / np.average(x, axis=0)
        #y = y / np.average(y, axis=0)
        
    if zscore:
        x = StandardScaler().fit_transform(x)
    x_plus_bias = np.c_[np.ones((np.size(x, 0), 1)), x]
    R = c * np.eye( x_plus_bias.shape[1] )
    R[0,0] = 0;
    temp = np.linalg.inv(np.dot(x_t, x_plus_bias) + R)
    temp2 = np.dot(temp,x_t)
    H = np.dot(temp2,y)
    return H #code is a little awkward, i tacked sw on top of stuff. 


def parameter_fit_with_sweep( x, y, c, sw=None, zscore=False):
    #now weighted by variance, might not be correct for all thigns
    reg_r2 = []
    if sw is not None:
        train_x, test_x, train_y, test_y, sw, nada = train_test_split(x, y, sw, test_size=.2)
    else:
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.2)
    for c_ in c:
        #print( 'Testing c= ' + str(c) )
        cv_r2 = []
        # fit decoder
        H = parameter_fit(train_x, train_y, c_, sw, zscore)
        #print( H.shape )
        # predict
        test_y_pred = test_wiener_filter(test_x, H)
        # evaluate performance
        reg_r2.append(weighted_r2(test_y, test_y_pred))
        # append mean of CV decoding for output

    reg_r2 = np.asarray(reg_r2)        
    best_c = c[ np.argmax( reg_r2 ) ] 
    print(f'best test:{np.max(reg_r2)}')
    return best_c

def train_wiener_filter(x, y, c = 0, n_l2=10, sw = None, sweep='log',
        zscore=False, display=True):
    """
    deceptive name, just regression with sample weight + ez sweeps
    x: input data, e.g. neural firing rates
    y: expected results, e.g. true EMG values
    C: if tuple, then sweep betweenlower log bound, upper low boudn
    else, if an int, then use as reg term
    """
    #print(sw)
    if type(c) is tuple:
        #n_l2 = 10 #edited
        if sweep == 'log':
            c = np.logspace(c[0], c[1], n_l2 )#edited to be larger numbers
        else:
            c = np.linspace(c[0], c[1], n_l2)
        best_c = parameter_fit_with_sweep( x, y, c, sw, zscore)
    else:
        best_c = c
    H_reg = parameter_fit( x, y, best_c, sw, zscore)

    yhat = test_wiener_filter(x, H_reg)
    train_score = weighted_r2(y, yhat)
    if display:
        print(f'best_c: {best_c}, train_r2: {train_score}')
    return H_reg

def generate_sepreg(regs, X, nlags, num_region1):
    reg1 = regs[0]
    reg2 = regs[1]
    C = np.zeros(X.shape[1] + 1)
    c_r1_mask, c_r2_mask = mask_h_region(C, nlags=nlags,
            num_region1=num_region1, keep_bias=False)
    C[c_r1_mask] = reg1
    C[c_r2_mask] = reg2
    return C


def _generate_clist(c, X,  nlags, num_region1, self_reg=0):
    if isinstance(c, tuple):
        reg1 = c[0]
        reg2 = c[1]
    else:
        reg1 = c
        reg2= c
     
    c_list = []
    features = X.shape[1]
    num_neurons = int(features / nlags)
    for idx in np.arange(num_region1): 
        C = np.zeros(X.shape[1] + 1)
        c_r1_mask, c_r2_mask = mask_h_region(C, nlags=nlags,
                num_region1=num_region1, keep_bias=False)
        C[c_r1_mask] = reg1
        C[c_r2_mask] = reg2

        which_neuron = np.arange(features) % num_neurons
        which_neuron_mask = which_neuron == idx

        C[1:][which_neuron_mask] = self_reg
        c_list.append(C)

    return c_list

def train_regression_sepreg(x, y, c, nlags, num_region1, sw=None, self_reg=0):
    c_list = _generate_clist(c, x, nlags, num_region1, self_reg=self_reg) 
    hs = []
    for idx, c in enumerate(c_list):
        hs.append(parameter_fit(x, y[:, idx], c, sw))
    return np.array(hs).T

def train_regression_selfonly(x, y, c, nlags, num_region1, sw=None):
    features = x.shape[1]
    num_neurons = int(features / nlags)
    hs=[]
    for idx in tqdm(np.arange(num_region1)): 
        which_neuron = np.arange(features) % num_neurons
        #keep_r2 = which_neuron >= num_region1
        keep_self = which_neuron == idx
        #keeps = np.logical_or(keep_r2, keep_self)
        keeps = keep_self

        hs.append(parameter_fit(x[:, keeps], y[:, idx], c, sw))
    return np.array(hs).T

def test_regression_selfonly(x, h, nlags, num_region1):
    features = x.shape[1]
    num_neurons = int(features / nlags)
    hs=[]
    yhat_list= []
    for idx in tqdm(np.arange(num_region1)): 
        which_neuron = np.arange(features) % num_neurons
        #keep_r2 = which_neuron >= num_region1
        keep_self = which_neuron == idx
        keeps = keep_self
        #keeps = np.logical_or(keep_r2, keep_self)
        yhat = test_wiener_filter(x[:, keeps], h[:, idx])
        yhat_list.append(yhat)
    return np.array(yhat_list).T

def test_wiener_filter(x, H, zscore_scaler=None):
    """
    To get predictions from input data x with linear decoder
    x: input data
    H: parameter vector obtained by training
    """

    x_plus_bias = np.c_[np.ones((np.size(x, 0), 1)), x]
    y_pred = np.dot(x_plus_bias, H)
    return y_pred    


def format_single_array(x, N=10):
    #deprecated
    data_N_lag = []
    for i in range(np.size(x)-N):
        data_N_lag.append(x[i+N-1])

    return np.asarray(data_N_lag)


