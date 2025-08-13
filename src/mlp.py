import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
from src.utils.evaluate import *
from src.utils.gen_utils import *
from torch.special import gammaln
import torch.nn.functional as F

# No dropout / Regularization yet -- Parameter for model?
# Using softplus which is not for the PCs
# Need to make separate model based on data type
# (Cuz using different activation function, and it might be messy if there's if statement in model architecture)
# softplus is only needed for raw spiking data
# Do we need activation function for PCs?
class MLPModel_NS(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel_NS, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    #After 

class SoftPlus_model(nn.Module):
    def __init__(self, input_size):
        super(SoftPlus_model, self).__init__()
        self.bias = nn.Parameter(torch.empty(input_size))
        self.bias = nn.init.uniform_(self.bias, -1/math.sqrt(input_size),
                1/math.sqrt(input_size))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.softplus(x+self.bias)
        return x

class tanh_model(nn.Module):
    def __init__(self, input_size):
        super(tanh_model, self).__init__()
        self.bias = nn.Parameter(torch.empty(input_size))
        self.bias = nn.init.uniform_(self.bias, -1/math.sqrt(input_size),
                1/math.sqrt(input_size))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(x+self.bias)
        return x

class EarlyStopper:
	def __init__(self, patience=1, min_delta=0):
		self.patience = patience #num epochs
		self.min_delta = min_delta #
		self.counter = 0
		self.min_validation_loss = float('inf')

	def early_stop(self, validation_loss):
		if validation_loss < self.min_validation_loss + self.min_delta:
			self.min_validation_loss = validation_loss
			self.counter = 0
		elif validation_loss > (self.min_validation_loss + self.min_delta):
			#print(f'{validation_loss} : {self.min_validation_loss} + {self.min_delta}')
			self.counter += 1
			if self.counter >= self.patience:
				return True
		return False

def custom_poiss_loss_single(pred_frs, spikes, m1, m1_reg,sw=None):
    # term1 = -torch.sum(-gammaln(spikes + 1) - pred_frs + spikes * torch.log(pred_frs + 1e-8))
    if sw is None:
        term1 = torch.mean(pred_frs - spikes * torch.log(pred_frs + 1e-8)) #log_input = False
    else:
        term1 = torch.mean((pred_frs - spikes * torch.log(pred_frs + 1e-8))*sw)
    
    lambda_reg_m1 = m1_reg

    # L2 Regularization
    m1_params = []
    m1_params.append(m1.fc1.weight)
    m1_params.append(m1.fc2.weight)
    m1_params.append(m1.fc3.weight)


    m1_len = len(flatten_list(m1_params))

    l2_norm_m1 = (lambda_reg_m1/m1_len) * sum(p.pow(2.0).sum() for p in m1_params)
 
    return term1 + l2_norm_m1

def custom_poiss_loss_separate(pred_frs, spikes, m1, m2, m1_reg, m2_reg,
        sw=None):
    # term1 = -torch.sum(-gammaln(spikes + 1) - pred_frs + spikes * torch.log(pred_frs + 1e-8))
    if sw is None:
        term1 = torch.mean(pred_frs - spikes * torch.log(pred_frs + 1e-8)) #log_input = False
    else:
        term1 = torch.mean((pred_frs - spikes * torch.log(pred_frs + 1e-8))*sw)
    
    lambda_reg_m1 = m1_reg
    lambda_reg_m2 = m2_reg

    # L2 Regularization
    m1_params = []
    m1_params.append(m1.fc1.weight)
    m1_params.append(m1.fc2.weight)
    m1_params.append(m1.fc3.weight)

    m1_len = len(flatten_list(m1_params))

    m2_params = []
    m2_params.append(m2.fc1.weight)
    m2_params.append(m2.fc2.weight)
    m2_params.append(m2.fc3.weight)

    m2_len = len(flatten_list(m2_params))


    l2_norm_m1 = (lambda_reg_m1/m1_len) * sum(p.pow(2.0).sum() for p in m1_params)
    l2_norm_m2 = (lambda_reg_m2/m2_len) *sum(p.pow(2.0).sum() for p in m2_params)
 
    l2_reg = l2_norm_m1 + l2_norm_m2

    return term1 + l2_reg


def train_e2e_single(mlp1, activa,x1, y, optimizer, criterion,loss_name, 
        mlp1_reg = 0,  sw=None):

    if loss_name is 'Poisson':
        loss_function = custom_poiss_loss_single
        eval_function = pseudo_R2
    else:
        loss_function = custom_mse_loss_single
        eval_function = r2_score_mse

    mlp1.train()
    activa.train()

    output_mlp1 = mlp1(x1)

    add_output = output_mlp1
    
    yhat = activa(add_output)
    loss = loss_function(yhat, y, mlp1, mlp1_reg, sw=sw)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    y_true_t = y.cpu().detach().numpy()
    y_pred_t = yhat.cpu().detach().numpy()

    r2_score = eval_function(y_true_t, y_pred_t,remove_lows=True)  # Calculate R2 for training
    r2_mean = np.average(r2_score, weights=np.var(y_pred_t, axis=0))
    return loss.item(), r2_mean

def train_e2e(mlp1, mlp2, activa,x1, x2, y, optimizer, criterion,loss_name, 
        mlp1_reg = 0, mlp2_reg = 0, sw=None):

    if loss_name is 'Poisson':
        loss_function = custom_poiss_loss_separate
        eval_function = pseudo_R2
    else:
        loss_function = custom_mse_loss_separate
        eval_function = r2_score_mse

    mlp1.train()
    mlp2.train()
    activa.train()

    output_mlp1 = mlp1(x1)
    output_mlp2 = mlp2(x2)

    add_output = output_mlp1 + output_mlp2 ###### Update ######
    
    yhat = activa(add_output)
    loss = loss_function(yhat, y, mlp1, mlp2, mlp1_reg, mlp2_reg,sw=sw)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    y_true_t = y.cpu().detach().numpy()
    y_pred_t = yhat.cpu().detach().numpy()

    r2_score = eval_function(y_true_t, y_pred_t,remove_lows=True)  # Calculate R2 for training
    r2_mean = np.average(r2_score, weights=np.var(y_pred_t, axis=0))
    return loss.item(), r2_mean

def evaluate_e2e_single(mlp1, activa, x1, y, loss_name, 
        mlp1_reg =0, sw=None):
    mlp1.eval()
    activa.eval()

    if loss_name == 'Poisson':
        #print('poisson r2')
        loss_function = custom_poiss_loss_single
        eval_function = pseudo_R2
    else:
        loss_function = custom_mse_loss_single
        eval_function = r2_score_mse

  
    with torch.no_grad():
        output_mlp1 = mlp1(x1)
        add_output = output_mlp1

        yhat = activa(add_output)

        loss = loss_function(yhat, y, mlp1, mlp1_reg, sw=sw)

        y_true = y.cpu().detach().numpy()
        y_pred = yhat.cpu().detach().numpy()
        r2_score = eval_function(y_true, y_pred,remove_lows=True)  # Calculate R2 for training
        r2_mean = np.average(r2_score, weights=np.var(y_pred, axis=0))
    return loss.item(), r2_mean, yhat

def evaluate_e2e(mlp1,mlp2,activa, x1, x2, y, loss_name, mlp1_reg =0, mlp2_reg = 0, sw=None):
    mlp1.eval()
    mlp2.eval()
    activa.eval()

    if loss_name == 'Poisson':
        loss_function = custom_poiss_loss_separate
        eval_function = pseudo_R2
    else:
        loss_function = custom_mse_loss_separate
        eval_function = r2_score_mse

  
    with torch.no_grad():
        output_mlp1 = mlp1(x1)
        output_mlp2 = mlp2(x2)
        add_output = output_mlp1 + output_mlp2

        yhat = activa(add_output)

        loss = loss_function(yhat, y, mlp1, mlp2, mlp1_reg, mlp2_reg,sw=sw)

        y_true = y.cpu().detach().numpy()
        y_pred = yhat.cpu().detach().numpy()
        r2_score = eval_function(y_true, y_pred,remove_lows=True)  # Calculate R2 for training
        r2_mean = np.average(r2_score, weights=np.var(y_pred, axis=0))
    return loss.item(), r2_mean, yhat


def predict_e2e_single(mlp1, activa, x1, y, loss_name):
    mlp1.eval()
    activa.eval()

    if loss_name == 'Poisson':
        eval_function = pseudo_R2
    else:
        eval_function = r2_score_mse

  
    with torch.no_grad():
        output_mlp1 = mlp1(x1)
        add_output = output_mlp1

        yhat = activa(add_output)
        y_pred = yhat.cpu().detach().numpy()
    return y_pred



def predict_e2e(mlp1,mlp2,activa, x1, x2, y, criterion,loss_name):
    mlp1.eval()
    mlp2.eval()
    activa.eval()

    if loss_name == 'Poisson':
        print('poisson r2')
        eval_function = pseudo_R2
    else:
        eval_function = r2_score_mse

  
    with torch.no_grad():
        output_mlp1 = mlp1(x1)
        output_mlp2 = mlp2(x2)
        add_output = output_mlp1 + output_mlp2

        yhat = activa(add_output)
        y_pred = yhat.cpu().detach().numpy()
    return y_pred



def get_region_predics(m1,m2,softplus_model, valid_x_cfa, valid_x_rfa, target, criterion,loss_name, remove_lows = False, m1_reg = 0, m2_reg = 0, sw=None):
    m1.eval()
    m2.eval()
    softplus_model.eval()
   
    with torch.no_grad():
        output_m1 = m1(valid_x_cfa)
        output_m2 = m2(valid_x_rfa)
        # concatenated_output = torch.cat((output_m1, output_m2), dim=2)
        reg1 = output_m1.cpu().detach().numpy()
        reg2 = output_m2.cpu().detach().numpy()

    return np.squeeze(reg1), np.squeeze(reg2)

def custom_mse_loss_single(pred_frs, spikes, m1, m2, m1_reg=1, m2_reg=1, sw=None):
    if sw is None:
        mse_loss = torch.mean(torch.pow(pred_frs - spikes, 2))
    else:

        mse_loss = torch.mean(torch.pow(pred_frs-spikes,2) * sw)
    lambda_reg_m1 = m1_reg

    # L2 Regularization
    m1_params = []
    m1_params.append(m1.fc1.weight)
    m1_params.append(m1.fc2.weight)
    m1_params.append(m1.fc3.weight)

    l2_norm_m1 = lambda_reg_m1 * sum(p.pow(2.0).sum() for p in m1_params)
 
    return mse_loss + l2_norm_m1



def custom_mse_loss_separate(pred_frs, spikes, m1, m2, m1_reg=1, m2_reg=1, sw=None):
    if sw is None:
        mse_loss = torch.mean(torch.pow(pred_frs - spikes, 2))
    else:

        mse_loss = torch.mean(torch.pow(pred_frs-spikes,2) * sw)
    lambda_reg_m1 = m1_reg
    lambda_reg_m2 = m2_reg

    # L2 Regularization
    m1_params = []
    m1_params.append(m1.fc1.weight)
    m1_params.append(m1.fc2.weight)
    m1_params.append(m1.fc3.weight)

    m2_params = []
    m2_params.append(m2.fc1.weight)
    m2_params.append(m2.fc2.weight)
    m2_params.append(m2.fc3.weight)


    l2_norm_m1 = lambda_reg_m1 * sum(p.pow(2.0).sum() for p in m1_params)
    l2_norm_m2 = lambda_reg_m2 *sum(p.pow(2.0).sum() for p in m2_params)
 
    l2_reg = l2_norm_m1 + l2_norm_m2

    return mse_loss + l2_reg


