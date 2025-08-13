import numpy as np
from src.MPOpto import *
from src.mlp import *
from src.NLModel import *
from copy import deepcopy
from src.regression import *
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from src.utils.torch_utils import *


class NLModel_single(NLModel):
    def __init__(self, session_path):
        print('single region NL model!!!')
        super().__init__(session_path)

    def _generate_trainset(self, X, Y, regs, hidden_size, lr,
            sw=None, valid_split=False):
        criterion = self.criterion
        loss_name = self.loss_name
        trainset={}

        with torch.device('cuda'):
            reg1_model = MLPModel_NS(input_size=X.shape[-1],
                    hidden_size=hidden_size, output_size=Y.shape[-1] )
            if loss_name == 'Poisson':
                activation_model = SoftPlus_model(input_size = Y.shape[-1])
            else:
                activation_model = tanh_model(input_size = Y.shape[-1])
        trainset['models'] = (reg1_model, activation_model)

        trainset['params'] = {}
        trainset['params']['regs'] = regs
        trainset['params']['hidden_size'] = hidden_size
        trainset['params']['lr'] = lr

        if valid_split:
            (train_data, test_data) = self._valid_split(X, Y, sw)
            trainset['train_data'] = train_data
            trainset['test_data'] = test_data
        else:
            X, Y = np2seq(X, Y)
            if sw is not None:
                sw = sw.to(device='cuda')
            train_data = (X, Y, sw)
            trainset['train_data'] = train_data
            trainset['test_data'] = None

        trainset['criterion'] = self.criterion
        trainset['loss_name'] = self.loss_name
        trainset['optimizer'] = torch.optim.Adam(list(reg1_model.parameters()) + 
                list(activation_model.parameters()),
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
        
        X = trainset['train_data'][0]
        Y = trainset['train_data'][1]
        sw= trainset['train_data'][2]

        mlp1 = trainset['models'][0]
        activa = trainset['models'][1]

        train_loss, train_r2=train_e2e_single(mlp1, activa,
            X,Y, optimizer, criterion, loss_name,  
            mlp1_reg = regs, sw=sw)

        train_output = (train_loss, train_r2)

        if valid_split:
            X_test = trainset['test_data'][0]
            Y_test = trainset['test_data'][1]
            sw_test= trainset['test_data'][2]
            test_loss, test_r2, _ = evaluate_e2e_single(mlp1, activa, X_test,
                    Y_test, loss_name, mlp1_reg=regs, sw=sw_test)
            test_output=(test_loss, test_r2)
            if display:
                print(f'test_loss: {test_loss}, test_r2: {test_r2}')
        else:
            test_output = None
            if display:
                print(f'train_loss: {train_loss}, train_r2: {train_r2}')

        return train_output, test_output

    
    def train(self, X, Y, num_epochs, regs = 1e-3, valid_split=False, hidden_size=200, lr=1e-3,
            sw=None, early_stop=False, display=True):

        trainset = self._generate_trainset(X, Y, regs, hidden_size, lr, sw,
                valid_split)
        es = EarlyStopper(patience=10, min_delta=0)

        checkpoint = num_epochs//10

        for epoch in tqdm(range(num_epochs), disable = not display):
            if epoch%checkpoint==0:
                train_output, test_output = self._train_e2e_trainset(trainset,
                        display=display)
            else:
                train_output, test_output = self._train_e2e_trainset(trainset,
                        display=False)

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


    def train_sweep(self,X, Y, num_epochs, reg_sweep=(-3, 0),sw=None, 
            hidden_size=200, lr=1e-3, num_sweep=5, display=True):

        reg_list = np.logspace(reg_sweep[0], reg_sweep[1], num_sweep)
        best_loss=1000

        for reg in tqdm(reg_list, disable = not display):
            trainset, _ = self.train(X,Y, num_epochs, regs=reg,
                    valid_split=True, hidden_size=hidden_size, lr=lr, sw=sw, 
                    early_stop=True, display=False)
            test_data = trainset['test_data']
            X_test = test_data[0]
            Y_test = test_data[1]
            sw_test = test_data[2]
            _, score = self.test(trainset, X_test, Y_test, sw=sw_test, regularize=False)
            val_loss = score[0]
            if display:
                print(f'test_r2:{score[1]}, val_loss:{score[0]}')
            if val_loss < best_loss:
                best_loss = val_loss
                best_set = trainset
        print(best_set['params']['regs'])
        return best_set

    def test(self, trainset, X, Y, sw=None, regularize=False):
        loss_name = self.loss_name
        if torch.is_tensor(Y) is not True:
            #convert to tensors, only look at Y lol
            X, Y = np2seq(X, Y)
            if sw is not None:
                sw = sw.to(device='cuda')

        mlp1 = trainset['models'][0]
        activa = trainset['models'][1]

        if regularize:
            regs = trainset['params']['regs']
        else:
            regs=0


        test_loss, test_r2, yhat = evaluate_e2e_single(mlp1, activa, X,
                Y, loss_name, mlp1_reg=regs, sw=sw)

        return yhat, (test_loss, test_r2)

    def get_region_predics(self, models,  X):
        print('doessnt work for single')
        return
    def get_relative_drive(self, models, X):
        print('doesnt work for single')
        return

    def _valid_split(self, X, Y, sw=None):
        X,X_test,Y, Y_test = train_test_split(X,Y,
                test_size=.2, shuffle=True, random_state=0)
        X, X_test,Y, Y_test = np2seq(X,X_test,Y, Y_test)

        if sw is not None:
            sw, sw_test = train_test_split(sw, test_size=.2, shuffle=True,
                    random_state=0)
            sw = sw.to(device='cuda')
            sw_test = sw_test.to(device='cuda')
        else:
            sw_test = None

        return (X, Y, sw), (X_test, Y_test, sw_test)
