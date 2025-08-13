#genutils

import pickle
import yaml
import numpy as np
import os
from concurrent.futures import *

def get_immediate_subdirectories(directory_path):
    """
    some ai shit...

    Returns a list of immediate subdirectories within the given directory path,
    including the full path to each subdirectory.
    """
    subdirectories = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            subdirectories.append(item_path)
    return subdirectories

def flatten_list(my_list):
    #lol
    return [new_list for little_list in my_list for new_list in little_list]

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def normalizeDataToRange(data, newRange):
    assert newRange != (0,1), 'just use normalizeData instead i think'
    norm = normalizeData(data)
    return norm * (newRange[1] - newRange[0]) + newRange[0]

def varexplained(subset, full=None):
    #samples x features
    if full==None:
        full = subset

    return np.var(subset, axis=0) / np.var(full)
   
def pload(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)
    

def pdump(object, pickle_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(object, f)

def load_yaml(yaml_path):
    with open(yaml_path, 'rb') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
                print(exc)
    return data

def nanlog(x):
    return np.log(x, out=np.zeros_like(x), where=(x!=0))

def nanaverage(A, axis, weights=1):
    return np.nansum(A*weights,axis=axis)/((~np.isnan(A))*weights).sum(axis=axis)

def multipool(function, *args, iterable, num_workers):
    #function runs on each element in iterable
    #first argument in function is the element

    result = [None] * len(iterable)
    executor = ProcessPoolExecutor(max_workers = num_workers)


    for idx, i in enumerate(iterable):
        if args is None:
            print('args is none')
            result[idx] = executor.submit(function, i)
        else:
            result[idx] = executor.submit(function, i, *args)

    wait(result, timeout=None, return_when=ALL_COMPLETED)

    output = []
    for this_result in result:
        output.append(this_result.result())

    return output

def squeezer(*args):
    output = []
    for arg in args:
        output.append(np.squeeze(arg))
    return tuple(output)

def avgstd(data, axis=None):
    return np.average(data, axis=axis), np.std(axis=axis)

def cosimil(A, B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

def pad_array(arr, m):
    """Pads a NumPy array with zeros to have m columns.

    Args:
        arr: The original NumPy array (n columns).
        m: The desired number of columns (m > n).

    Returns:
        A new NumPy array with m columns, padded with zeros, or the original
        array if m <= n.  Raises a ValueError if m < n.
    """

    n = arr.shape[1]  # Get the current number of columns

    if m == n:
        return arr # No padding needed
    elif m < n:
        raise ValueError("m must be greater than or equal to the current number of columns")
    else:
        padding_width = m - n
        padding = [(0, 0)] * arr.ndim  # Pad all dimensions with 0 rows
        padding[1] = (0, padding_width)  # Pad the columns with padding_width zeros
        padded_arr = np.pad(arr, padding, mode='constant')
        return padded_arr
