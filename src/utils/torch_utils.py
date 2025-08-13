import numpy as np
import torch

def np2seq(*args):
    output=[]
    with torch.device('cuda'):
        for x in args:
            if x is None:
                continue
            #x = x.reshape(len(x),1,-1)
            output.append(torch.tensor(x, dtype=torch.float))
        return tuple(output)

