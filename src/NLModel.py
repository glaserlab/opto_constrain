import numpy as np
from src.MPOpto import *
from src.mlp import *
from copy import deepcopy
from src.regression import *
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from src.utils.torch_utils import *


class NLModel(MPOpto):
    def __init__(self, session_path):
       super().__init__(session_path)
       print('this will use poisson loss')
       self.criterion = nn.PoissonNLLLoss(log_input=False)
       self.loss_name = "Poisson"


    def format_nlags(self, bounds=None, binsize=10, nlags=10):
        ''' generate X, Y, split by regions.'''
        if bounds is None:
            bounds = self.climbing_bounds

        region1_binned, region2_binned = self.binner(bounds=bounds, binsize=binsize, concat=False)
        
        reg1_nlags, reg1_cut = format_and_stitch_ar(region1_binned,
                nlags=nlags)
        reg2_nlags, _ = format_and_stitch_ar(region2_binned, nlags=nlags)

        return (reg1_nlags, reg2_nlags), reg1_cut

    def _generate_trainset(self, Xs, Y, regs, hidden_size, lr,
            sw=None, valid_split=False):
        criterion = self.criterion
        loss_name = self.loss_name
        trainset={}

        X1=Xs[0]
        X2=Xs[1]
        with torch.device('cuda'):
            reg1_model = MLPModel_NS(input_size=X1.shape[-1],
                    hidden_size=hidden_size, output_size=Y.shape[-1] )
            reg2_model = MLPModel_NS(input_size=X2.shape[-1], 
                    hidden_size=hidden_size, output_size=Y.shape[-1] )
            if loss_name == 'Poisson':
                activation_model = SoftPlus_model(input_size = Y.shape[-1])
            else:
                activation_model = tanh_model(input_size = Y.shape[-1])
        trainset['models'] = (reg1_model, reg2_model, activation_model)

        trainset['params'] = {}
        trainset['params']['regs'] = regs
        trainset['params']['hidden_size'] = hidden_size
        trainset['params']['lr'] = lr

        if valid_split:
            (train_data, test_data) = self._valid_split(Xs, Y, sw)
            trainset['train_data'] = train_data
            trainset['test_data'] = test_data
        else:
            X1,X2, Y = np2seq(X1,X2, Y)
            if sw is not None:
                sw = sw.to(device='cuda')
            train_data = ((X1,X2), Y, sw)
            trainset['train_data'] = train_data
            trainset['test_data'] = None

        trainset['criterion'] = self.criterion
        trainset['loss_name'] = self.loss_name
        trainset['optimizer'] = torch.optim.Adam(list(reg1_model.parameters()) +
                list(reg2_model.parameters()) + list(activation_model.parameters()),
            lr=lr, weight_decay=0)

        trainset['train_scores']=[]
        trainset['test_scores']=[]



        return trainset

    def _train_e2e_trainset(self, trainset, valid_split=True, display=True):
        regs = trainset['params']['regs']
        #hidden_size = trainset['params']['hidden_size']
        #lr = trainset['params']['lr']
        criterion = trainset['criterion']
        loss_name = trainset['loss_name']
        optimizer = trainset['optimizer']
        
        X1 = trainset['train_data'][0][0]
        X2 = trainset['train_data'][0][1]
        Y = trainset['train_data'][1]
        sw= trainset['train_data'][2]

        mlp1 = trainset['models'][0]
        mlp2 = trainset['models'][1]
        activa = trainset['models'][2]

        train_loss, train_r2=train_e2e(mlp1,mlp2,activa,
            X1, X2,Y, optimizer, criterion, loss_name,  
            mlp1_reg = regs[0], mlp2_reg = regs[1], sw=sw)

        train_output = (train_loss, train_r2)

        if valid_split:
            X1_test = trainset['test_data'][0][0]
            X2_test = trainset['test_data'][0][1]
            Y_test = trainset['test_data'][1]
            sw_test= trainset['test_data'][2]
            test_loss, test_r2, _ = evaluate_e2e(mlp1, mlp2, activa, X1_test,
                    X2_test, Y_test, loss_name, mlp1_reg=regs[0], mlp2_reg=regs[1],
                    sw=sw_test)
            test_output=(test_loss, test_r2)
            if display:
                print(f'test_loss: {test_loss}, test_r2: {test_r2}')
        else:
            test_output = None
            if display:
                print(f'train_loss: {train_loss}, train_r2: {train_r2}')

        return train_output, test_output

    
    def train(self, Xs, Y, num_epochs, regs = (1e-3,1e-3), valid_split=False, hidden_size=200, lr=1e-3,
            sw=None, early_stop=False, display=True):

        trainset = self._generate_trainset(Xs, Y, regs, hidden_size, lr, sw,
                valid_split)
        es = EarlyStopper(patience=10, min_delta=0)

        checkpoint = num_epochs//10

        for epoch in tqdm(range(num_epochs), disable=not display):
            if epoch%checkpoint==0:
                train_output, test_output = self._train_e2e_trainset(trainset,
                        display=display, valid_split=valid_split)
            else:
                train_output, test_output = self._train_e2e_trainset(trainset,
                        display=False, valid_split=valid_split)

            trainset['train_scores'].append(train_output)
            trainset['test_scores'].append(test_output)
            if early_stop:
                if es.early_stop(test_output[0]):
                    print('early stopping')
                    break

        if valid_split:
            summary_score = train_output
        else:
            summary_score = test_output

        return trainset, summary_score


    def train_sweep(self,Xs, Y, num_epochs, reg_sweep=(-3, 0),sw=None, 
            hidden_size=200, lr=1e-3, num_sweep=5, display=True):

        reg_list = np.logspace(reg_sweep[0], reg_sweep[1], num_sweep)
        best_loss=1000

        for reg in tqdm(reg_list):
            regs = (reg, reg)
            trainset, _ = self.train(Xs,Y, num_epochs, regs=regs,
                    valid_split=True, hidden_size=hidden_size, lr=lr, sw=sw, 
                    early_stop=True, display=False)
            test_data = trainset['test_data']
            Xs_test = test_data[0]
            Y_test = test_data[1]
            sw_test = test_data[2]
            _, score = self.test(trainset, Xs_test, Y_test, sw=sw_test, regularize=False)
            val_loss = score[0]
            if display:
                print(f'test_r2:{score[1]}, val_loss:{score[0]}')
            if val_loss < best_loss:
                best_loss = val_loss
                best_set = trainset
                best_regs = regs

        print(best_set['params']['regs'])
        return best_set, best_regs

    def test(self, trainset, Xs, Y, sw=None, regularize=False):
        loss_name = self.loss_name
        X1=Xs[0]
        X2=Xs[1]
        if torch.is_tensor(Y) is not True:
            #convert to tensors, only look at Y lol
            X1,X2, Y = np2seq(X1,X2, Y)
            if sw is not None:
                sw = sw.to(device='cuda')



        mlp1 = trainset['models'][0]
        mlp2 = trainset['models'][1]
        activa = trainset['models'][2]

        if regularize:
            regs = trainset['params']['regs']
        else:
            regs=(0,0)


        test_loss, test_r2, yhat = evaluate_e2e(mlp1, mlp2, activa, X1,
                X2, Y, loss_name, mlp1_reg=regs[0], mlp2_reg=regs[1],
                sw=sw)

        return yhat, (test_loss, test_r2)


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
            sw_list.append(format_single_array(temp, N=nlags))
        sw = np.hstack(sw_list)
        sw[sw >= 1] = sw_weight - 1
        sw = sw+1
        sw = sw / np.average(sw)
        return torch.tensor(sw[:, np.newaxis])


    def get_region_predics(self, trainset,  Xs):
        models = trainset['models']
        reg1_model = models[0]
        reg2_model = models[1]
        X1=Xs[0]
        X2=Xs[1]

        X1, X2 = np2seq(X1, X2)
        reg1_model.eval()
        reg2_model.eval()
       
        with torch.no_grad():
            y1 = reg1_model(X1)
            y2 = reg2_model(X2)
            reg1 = y1.cpu().detach().numpy()
            reg2 = y2.cpu().detach().numpy()

        return np.squeeze(reg1), np.squeeze(reg2)

    def get_relative_drive(self, models, Xs):
        reg1_predic, reg2_predic = self.get_region_predics(models, Xs)
        reg1_drive = np.sum(np.abs(reg1_predic))
        reg2_drive = np.sum(np.abs(reg2_predic))

        return np.log10(reg1_drive/reg2_drive)

    def _valid_split(self, Xs, Y, sw=None):
        X1=Xs[0]
        X2=Xs[1]
        X1,X1_test, X2, X2_test, Y, Y_test = train_test_split(X1, X2, Y,
                test_size=.2, shuffle=True, random_state=0)
        X1, X1_test, X2, X2_test,  Y, Y_test = np2seq(X1,X1_test, X2,
                X2_test, Y, Y_test)

        if sw is not None:
            sw, sw_test = train_test_split(sw, test_size=.2, shuffle=True,
                    random_state=0)
            sw = sw.to(device='cuda')
            sw_test = sw_test.to(device='cuda')
        else:
            sw_test = None

        Xs = (X1, X2)
        Xs_test = (X1_test, X2_test)

        return (Xs, Y, sw), (Xs_test, Y_test, sw_test)
