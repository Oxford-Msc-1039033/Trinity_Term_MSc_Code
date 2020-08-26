
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:07:50 2020

@author: Tom
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import cal_cc_norm


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"   
device = torch.device(dev)  
    
class nrfmodel(nn.Module):
    def __init__(self, input_size, h1_size, weight_scaling, p=0, use_dropout = False):
        super(nrfmodel, self).__init__()
        self.layer1 = nn.Linear(input_size, h1_size, bias=True)
        self.layer2 = nn.Linear(h1_size, 1, bias=True)
        self.output_mod = nn.Linear(1,1, bias = True)
        self.use_dropout = use_dropout
        
        self.dropout =  nn.Dropout(p=p) 
        self.sigmoid = nn.Sigmoid() 
        #scale the weights
        with torch.no_grad():
            self.layer1.weight.mul_(weight_scaling)
            self.layer2.weight.mul_(weight_scaling*10)        
    def forward(self, x):  
        if self.use_dropout:
            h1 = self.sigmoid(self.layer1(self.dropout(x)))
        else:
            h1 = self.sigmoid(self.layer1(x))
        out = self.sigmoid(self.layer2(h1))
        out = self.output_mod(out)
        return  out

def train(stimuli, target_cluster, input_fht,  neuron = 0, dropout = False, h1_size = 10, weight_scaling = 0.0001, p = 0.5, gamma = 1e-04, optim = 'adam', 
          epochs = 30000, lr = 0.0001, idx=0, input_area = 'NA'):
    
    y = torch.unsqueeze(torch.from_numpy(target_cluster[neuron].y_t).type(torch.FloatTensor),0).t()  #target output values
    y = y.to(device)
    loss_array = np.zeros(epochs)                #array to store loss values on the training set
    tune_mse_array = np.zeros(epochs)           #array to store loss values on the validation/tuning set
    tune_ccnorm_array = np.zeros(epochs)  
    report_ccnorm_array = np.zeros(epochs)
    report_mse_array = np.zeros(epochs)
    loss = 1
    cc_norm_max = -np.inf
    mse_loss_min = np.inf
    epoch = 0
    ccnorm_epoch = -1
    
    count = 0
    inputs = np.empty([0,6400])
    #Remove recordings from same area with same unique pen
    if input_area.lower() == target_cluster[neuron].brainArea.lower():
        for i in range(input_fht.shape[0]):
            if target_cluster[i].unique_penetration_idx != target_cluster[neuron].unique_penetration_idx:
                inputs= np.concatenate((inputs, input_fht[i,:,:]),0)
        inputs = inputs.T                 
        inputs = torch.from_numpy(inputs).type(torch.FloatTensor)     #Convert to torch.Tensor 
        inputs = inputs.to(device)                                    #Move the tensor to cuda.
                
    else:
        for i in range(input_fht.shape[0]):
            inputs= np.concatenate((inputs, input_fht[i,:,:]),0)
        inputs = inputs.T                 #each row corresponds to one training pattern and the columns correspond to the historical input neurons.
        inputs = torch.from_numpy(inputs).type(torch.FloatTensor)     #Convert to torch.Tensor 
        inputs = inputs.to(device)                                    #Move the tensor to cuda.
                
    

    model = nrfmodel(inputs.shape[1], h1_size, weight_scaling, p=p, use_dropout = dropout)
    model.to(device)
    
    if optim.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    elif optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
        
    mse_loss = nn.MSELoss()
    mse_loss.to(device)

    while epoch < epochs:
        optimizer.zero_grad()   #reset gradients
        
        model.train()                        
        outputs = model(inputs[stimuli.train_idx.astype(int)-1,:])
        
        loss = mse_loss(outputs, torch.unsqueeze(y[stimuli.train_idx.astype(int)-1, 0],1)) +\
                gamma*model.layer1.weight.norm(p=1) + gamma*model.layer2.weight.norm(p=1)
       # Backward and optimize
        loss.backward()        
        optimizer.step()
         
        model.eval()
        with torch.no_grad():
            tune_out = model(inputs[stimuli.tune_idx.astype(int)-1,:])
            tune_mse_loss = mse_loss(tune_out, torch.unsqueeze(y[stimuli.tune_idx.astype(int)-1, 0],1)).detach()
            report_out = model(inputs[stimuli.report_idx.astype(int)-1,:])
            report_mse_loss = mse_loss(report_out, torch.unsqueeze(y[stimuli.report_idx.astype(int)-1, 0],1)).detach()
       
        cc_abs, tune_cc_norm, cc_max = cal_cc_norm.calculate(target_cluster[neuron].y_dt[:,stimuli.tune_idx.astype(int)-1], 
                                                        tune_out[:,0].cpu().detach().numpy())
        
        if tune_cc_norm == np.nan and epoch == 0:
            print('SP for Neuron {:d} is negative or equal to zero'.format(neuron))
        rep_cc_abs, rep_cc_norm, rep_cc_max = cal_cc_norm.calculate(target_cluster[neuron].y_dt[:,stimuli.report_idx.astype(int)-1], 
                                                        report_out[:,0].cpu().detach().numpy())
        
        loss_array[epoch] = loss.item()
        tune_mse_array[epoch] = tune_mse_loss
        tune_ccnorm_array[epoch] = tune_cc_norm
        report_ccnorm_array[epoch] = rep_cc_norm
        report_mse_array[epoch] = report_mse_loss
                       
        if (cc_norm_max <= tune_cc_norm) and epoch> 2000:       #checking vs previous highest cc_norm.
            ccnorm_model = type(model)(inputs.shape[1], h1_size, weight_scaling, p=p) 
            ccnorm_model.load_state_dict(model.state_dict())
            ccnorm_epoch = epoch
            cc_norm_max = tune_cc_norm
            count = 0
        elif epoch > 2000:
            count +=1 
            
        if (tune_mse_loss <= mse_loss_min) and epoch> 2000:       #checking vs previous lowest mse_tune.
            mse_model = type(model)(inputs.shape[1], h1_size, weight_scaling, p=p) 
            mse_model.load_state_dict(model.state_dict())
            mse_epoch = epoch
            mse_loss_min = tune_mse_loss

#        if (epoch+1)%100==0:
#            print ('Run: {}, Epoch: {}, Loss:{:.4f}... ccnorm:{:.4f}... mse:{:.4f}'.format(idx+1, epoch+1, loss.item(), tune_cc_norm, tune_mse_loss)) 
        epoch += 1
        
    try:                         #Create a model save in case if cc_norm = nan
        ccnorm_model
    except NameError:
        ccnorm_model = type(model)(inputs.shape[1], h1_size, weight_scaling, p=p) 
        ccnorm_model.load_state_dict(model.state_dict())
        
    return inputs, loss_array, tune_mse_array, tune_ccnorm_array, report_mse_array, report_ccnorm_array,  ccnorm_epoch, ccnorm_model, mse_epoch, mse_model
