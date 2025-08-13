from sklearn.metrics import r2_score
import numpy as np

def vaf(x,xhat, round_values=True):
    """
    Calculating vaf value
    x: actual values, a numpy array
    xhat: predicted values, a numpy array
    """
    x = x - x.mean(axis=0)
    xhat = xhat - xhat.mean(axis=0)
    if round_values is True:
        return np.round((1-(np.sum(np.square(x -
            xhat))/np.sum(np.square(x)))),2)
    else:
        return (1-(np.sum(np.square(x - xhat))/np.sum(np.square(x))))

def r2_score_mse(x, xhat, remove_lows=True):
    score = r2_score(x, xhat, multioutput='raw_values')
    if remove_lows:
        score[score<-1]=-1
    return score
    
    

def weighted_r2(x, xhat, remove_lows=True):
    score = r2_score(x, xhat, multioutput='raw_values')
    if remove_lows:
        score[score<-1]=-1
    variance = np.var(xhat, axis=0)
    return np.average(score, weights=variance)

# Maybe adding remove_lows like weighted_r2
def pseudo_R2(y, yhat, ynull = False, eps=1e-8, remove_lows = True,
        weighted=False):
    """Pseudo-R2 metric. Using for poission loss (Raw Neuron Activity, not PCs)
    Parameters
    ----------
    y : array
        Target labels of shape (timepoints, number of neurons)
    yhat : array
        Predicted labels of shape (timepoints, number of neurons)
    ynull_ : float
        Mean of the target labels (number of neurons,)
    Returns
    -------
    score : float
        Pseudo-R2 score.
    """
    ynull = np.mean(y, axis=0)
    LS = logL(y, y)
    L0 = logL(y, ynull)
    L1 = logL(y, yhat)

    score = (1 - (LS - L1) / (LS - L0))
    if remove_lows:
        score[score<-1]=-1
    return score 

def logL(y,y_hat):
    """Log likelihood, using for pseudo R2"""
    eps = 1e-8
    logL = np.sum(y * np.log(y_hat + eps) - y_hat, axis = 0)

    return logL

