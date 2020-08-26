# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 10:18:24 2020

@author: Tom
"""

# -*- coding: utf-8 -*-
import numpy as np
import scipy.io
import copy_mat2py
import cal_cc_norm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import time
import train_nrf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import DivergingNorm

#import TESTING_train_a2a_nrf

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  


#Load the ic,mgb, and ac data. Note - this also includes the matlab nrf model data.
mat = scipy.io.loadmat('data/B1_clusters_ic_a2a_nrf.mat', squeeze_me=True)
clusters_ic_mat = mat['clusters_ic']
clusters_ic = []
for i in range(clusters_ic_mat.size):
    clusters_ic.append(mat2py_copy.cluster_unpacker(i, clusters_ic_mat))
    
mat = scipy.io.loadmat('data/B1_clusters_mgb_a2a_nrf.mat', squeeze_me=True)
clusters_mgb_mat = mat['clusters_mgb']
clusters_mgb = []
for i in range(clusters_mgb_mat.size):
    clusters_mgb.append(mat2py_copy.cluster_unpacker(i, clusters_mgb_mat))
    
mat = scipy.io.loadmat('data/B1_clusters_ac_a2a_nrf.mat', squeeze_me=True)
clusters_ac_mat = mat['clusters_ac']
clusters_ac = []
for i in range(clusters_ac_mat.size):
    clusters_ac.append(mat2py_copy.cluster_unpacker(i, clusters_ac_mat))

#Import the tensorized data
mat = scipy.io.loadmat('data/B1_X_fht.mat', squeeze_me=True)
x_fht = mat['X_fht'][()]
mat = scipy.io.loadmat('data/B1_ic_fht.mat', squeeze_me=True)
ic_fht = mat['ic_fht'][()]
mat = scipy.io.loadmat('data/B1_mgb_fht.mat', squeeze_me=True)
mgb_fht = mat['mgb_fht'][()]
mat = scipy.io.loadmat('data/B1_ac_fht.mat', squeeze_me=True)
ac_fht = mat['ac_fht'][()]

#Load stimulus data
mat = scipy.io.loadmat('data/A3_stimuli_with_idxes.mat', squeeze_me=True)
stimuli_mat = mat['stimuli']
stimuli = mat2py_copy.stimulus_unpacker(stimuli_mat)

del mat
del stimuli_mat
del clusters_ic_mat
del clusters_ac_mat
del clusters_mgb_mat

### Initializing the nrf model ###
weight_scaling = 1e-5
momentum_rate = 0.9
h1_size = 10
epochs = 40000
#lr = 1e-02      #use for lbfhs
lr = 1e-04      #use for adam/sgd
gamma_values = [1e-03, 1e-04, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10]
dropout = False
p = 0.5
filename = "B1_nrf_ic2mgb_MSE_L1squared.pkl" 

areas = ['coch', 'ic', 'mgb', 'ac']

output_area = 'mgb'     #Have output_clusters on outermost loop
#output_clusters = clusters_ic
if output_area == 'coch':
    print('cannot select cochleagram as output area')
elif output_area == 'ic':
    output_clusters = clusters_ic
elif output_area == 'mgb':
    output_clusters = clusters_mgb
elif output_area == 'ac':
    output_clusters = clusters_ac


input_area = 'ic'     #Input_clusters on next layer loop

if input_area == 'coch':
    input_fht = x_fht
elif input_area == 'ic':
    input_fht = ic_fht
elif input_area == 'mgb':
    input_fht = mgb_fht
elif input_area == 'ac':
    input_fht = ac_fht
    
if input_area == 'coch':
    print('cannot select clusters for cochleagram')
elif input_area == 'ic':
    input_clusters = clusters_ic
elif input_area == 'mgb':
    input_clusters = clusters_mgb
elif input_area == 'ac':
    input_clusters = clusters_ac


#Define the training, tuning, and reporting sets via their indices.
stimuli.train_idx = stimuli.all_valid_data.no_folds.fit_idx[0:int(len(stimuli.all_valid_data.no_folds.fit_idx)*0.9)]
stimuli.tune_idx = stimuli.all_valid_data.no_folds.fit_idx[int(len(stimuli.all_valid_data.no_folds.fit_idx)*0.9):]
stimuli.report_idx = stimuli.all_valid_data.no_folds.pred_idx

#Inside neuron for loop:
tic = time.time()
for neuron in range(len(output_clusters)):
    print('Training Neuron {:d} out of {:d}'.format(neuron+1, len(output_clusters)))
    output_clusters[neuron].initialize_gamma(gamma_values= gamma_values, input_area = input_area)
    if output_clusters[neuron].mat.ic.cc_max <= 0.2:         #skip noisy neurons
        output_clusters[neuron].py_trained = False
        print('Skipped neuron {}, cc_max:{:.4f}'.format(neuron+1, output_clusters[neuron].mat.ic.cc_max))
        with open(filename, 'wb') as f:
            pickle.dump(output_clusters, f)
        continue

    for idx, gamma_val in enumerate(gamma_values):
        print('Training for gamma value: {:d} out of {:d}'.format(idx+1, len(gamma_values)))
        inputs, loss_array, tune_mse_array, tune_ccnorm_array, report_mse_array, report_ccnorm_array,  ccnorm_epoch, ccnorm_model, mse_epoch, mse_model  = train_a2a_nrf_copy.train(stimuli, output_clusters, input_fht, neuron = neuron, 
                                                            dropout= dropout,  h1_size=h1_size, epochs=epochs, idx=idx, lr = lr, input_area = input_area, gamma=gamma_val, 
                                                            optim='adam', p=p)
        
        ccnorm_model.to(device)                #Move saved model to cuda to evaluate performance on report set.
        ccnorm_model.eval()
        mse_model.to(device)
        mse_model.eval()
        with torch.no_grad():                          
            ccnorm_out = ccnorm_model(inputs[stimuli.report_idx.astype(int)-1,:])
            mse_out = mse_model(inputs[stimuli.report_idx.astype(int)-1,:])
        
        mse_out = mse_out[:,0].cpu().detach().numpy()
        ccnorm_out = ccnorm_out[:,0].cpu().detach().numpy()
        
        ccnorm_ccabs, ccnorm_ccnorm ,cc_max = cal_cc_norm.calculate(output_clusters[neuron].y_dt[:,stimuli.report_idx.astype(int)-1], 
                             ccnorm_out)
        mse_ccabs, mse_ccnorm ,cc_max = cal_cc_norm.calculate(output_clusters[neuron].y_dt[:,stimuli.report_idx.astype(int)-1], 
                             mse_out)

        print('Best CCnorm Epochs: {:d}'.format(ccnorm_epoch))
        print('Best CCnorm Model: CC_abs: {:.4f}, CC_norm: {:.4f}, CC_max:{:.4f}'.format(ccnorm_ccabs, ccnorm_ccnorm,cc_max))
        print('Best MSE Epochs: {:d}'.format(mse_epoch))
        print('Best MSE Model: CC_abs: {:.4f}, CC_norm: {:.4f}, CC_max:{:.4f}'.format(mse_ccabs, mse_ccnorm,cc_max))
        x_mat, y_mat, z_mat = output_clusters[neuron].mat.mgb.cc_abs, output_clusters[neuron].mat.mgb.cc_norm, output_clusters[neuron].mat.mgb.cc_max
        print('Matlab NRF Model: CC_abs: {:.4f}, CC_norm: {:.4f}, CC_max:{:.4f}'.format(x_mat,y_mat,z_mat))
        
        ccnorm_model.cpu()    #Need to move the model back to cpu to save it. 
        mse_model.cpu()
        
        output_clusters[neuron].add_py_data(gamma_idx = idx, input_area = input_area, dropout=dropout, p=p, gamma_value = gamma_val , loss_history = loss_array,
                    ccnorm_history = report_ccnorm_array, tune_ccnorm_history = tune_ccnorm_array,  mse_history = report_mse_array, tune_mse_history = tune_mse_array,
                    cc_max = cc_max, ccnorm_epoch = ccnorm_epoch , ccnorm_cc_abs = ccnorm_ccabs , ccnorm_yhat = ccnorm_out, ccnorm_model = ccnorm_model,
                    mse_cc_abs = mse_ccabs, mse_yhat=mse_out, mse_epoch = mse_epoch, mse_model = mse_model)
    elapsed = time.time() - tic
    print('\n Time Elapsed: {:.4f}'.format(elapsed))
    output_clusters[neuron].py_trained = True
    #saving after each neuron just in case
    with open(filename, 'wb') as f:
        pickle.dump(output_clusters, f)
        
with open(filename, 'wb') as f:
    pickle.dump(output_clusters, f)
    
#file2 = open(filename, 'rb')
#new_d = pickle.load(file2)
#file2.close()    
    
elapsed = time.time() - tic
print('Time Elapsed: {:.4f}'.format(elapsed))

#plot loss and test loss
#t = np.arange(loss_array.shape[0])
#end_point = loss_array.shape[0]
#
#fig, ax  =plt.subplots(figsize=(9, 9))   
#ax.plot(t[:end_point], loss_array[:end_point], c='k', linewidth = 1, label='train')
#ax.plot(t[:end_point], report_mse_array[:end_point], c='olive', linewidth = 1, label='report_mse')
#ax.plot(t[:end_point], tune_mse_array[:end_point], c='g', linewidth = 1, label='tune_mse')
#ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
#ax2.plot(t[:end_point], tune_ccnorm_array[:end_point], color='mediumblue', linewidth = 1, label='tune cc_norm')
#ax2.plot(t[:end_point], report_ccnorm_array[:end_point], color='red',  linewidth = 1, label='report cc_norm')
#ax.set_xlabel('Epochs', fontsize = 20)
#ax2.set_ylabel('ccnorm', fontsize = 20)
#ax.set_ylabel('loss', color = 'k')
#ax.legend()
#
#fig2 =plt.figure(figsize=(9, 9))   
#ax2 = fig2.add_subplot(1,1,1) 
#ax2.plot(t[:end_point], tune_ccnorm_array[:end_point], color='mediumblue', linewidth = 1, label='tune cc_norm')
#ax2.plot(t[:end_point], report_ccnorm_array[:end_point], color='red',  linewidth = 1, label='report cc_norm')
#ax2.set_xlabel('Epochs', fontsize = 20)
#ax2.set_ylabel('cc_norm', fontsize = 20)
#ax2.legend()
#
#fig.savefig('Loss.png')
#fig2.savefig('cc_norm.png')
#
#
#def show_strf_report(strf, neuron='NA', gamma='NA', epochs='NA',  area ='NA', input_area='NA', cc_norm='NA', save_fig = False):
#
#    my_dpi = 120
#    mx = max([np.max(strf), np.abs(np.min(strf))])
#    fig, ax = plt.subplots()
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes('right', size='5%', pad=0.05)
#    im = ax.imshow(strf, norm=DivergingNorm(0), cmap = 'seismic', vmin=-mx, vmax=mx)
#    fig.colorbar(im, cax=cax, orientation = 'vertical')
#    xcoords = np.arange(14.5,150,15)
#    for xc in xcoords:
#        ax.axvline(x=xc, c='k', linewidth=0.1)
#    ax.set_title('neuron: {}, gamma:{}, \n epochs:{}, cc norm: {:.4f}'.format(neuron, gamma, epochs, cc_norm), fontsize=5)
#    #plt.show()
#    if save_fig:
#        plt.savefig('RFs_best_report/report_rf_{}2{}_neuron{}.png'.format(input_area, area, neuron), 
#                    dpi=my_dpi*20, bbox_inches='tight')
#
#
#strf = np.zeros((448, 10*15))
#data = output_clusters[44].py.ic[0].mse_model
#model = data.model
#gamma = gamma_values[1]
#best_epoch = data.best_epoch
#cc_norm = data.ccnorm
#for ii in range(448):
#    for jj in range(h1_size):
#        sign = np.sign(model.layer2.weight.data.numpy()[0,jj])
#        strf[ii,jj*15:(jj+1)*15] = sign*model.layer1.weight.data.numpy()[jj,ii*15:(ii+1)*15]
#show_strf_report(strf, neuron=43, gamma=gamma, epochs=best_epoch, cc_norm=cc_norm, input_area='ic', area='mgb', save_fig=False)
#
#'''
#plot by epoch data
#'''
#t = np.arange(loss_array.shape[0])
#end_point = loss_array.shape[0]
#data = output_clusters[44].py.ic[0]
#
#fig, ax  =plt.subplots(figsize=(9, 9))   
#ax.plot(t[:end_point], data.loss_history[:end_point], c='k', linewidth = 1, label='train')
#ax.plot(t[:end_point], data.mse_history[:end_point], c='olive', linewidth = 1, label='report_mse')
#ax.plot(t[:end_point], data.tune_mse_history[:end_point], c='g', linewidth = 1, label='tune_mse')
#ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
#ax2.plot(t[:end_point], data.tune_ccnorm_history[:end_point], color='mediumblue', linewidth = 1, label='tune cc_norm')
#ax2.plot(t[:end_point], data.ccnorm_history[:end_point], color='red',  linewidth = 1, label='report cc_norm')
#ax.set_xlabel('Epochs', fontsize = 20)
#ax2.set_ylabel('ccnorm', fontsize = 20)
#ax.set_ylabel('loss', color = 'k')
#ax.legend()