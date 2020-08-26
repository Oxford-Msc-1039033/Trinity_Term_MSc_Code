# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:53:57 2020

@author: thble
"""

'''
Calculating CC Norm on Training Set
'''

import numpy as np
import scipy.io
import mat2py_copy
import cal_cc_norm
import matplotlib.pyplot as plt
import pickle
import math
import torch

ac2ic_fut = False
input_area = 'ic'
'''
note - if doing mapping from ac need to make sure all .ic attributes are renamed to .ac, and vice versa
'''

mat = scipy.io.loadmat('C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\hierachy-py\\data\\B1_X_fht.mat', squeeze_me=True)
x_fht = mat['X_fht'][()]
mat = scipy.io.loadmat('C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\hierachy-py\\data\\B1_ic_fht.mat', squeeze_me=True)
ic_fht = mat['ic_fht'][()]
mat = scipy.io.loadmat('C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\hierachy-py\\data\\B1_mgb_fht.mat', squeeze_me=True)
mgb_fht = mat['mgb_fht'][()]
mat = scipy.io.loadmat('C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\hierachy-py\\data\\B1_ac_fht.mat', squeeze_me=True)
ac_fht = mat['ac_fht'][()]

#ic2mgb data:
# filename = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\my-data\\08-12-2020 (ic2mgb full L1squared MSE Dropout )\\data\\B1_nrf_ic2mgb_MSE_dropout_L1squared.pkl'
# ic2ac data:
filename = "C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\my-data\\08-16-2020 (ic2ac)\\data\\B1_nrf_ic2ac.pkl"
# ac2ic0 data:
#filename = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\my-data\\08-18-2020 (ac2ic n_fut_0)\\data\\B1_nrf_ac2ic_nfut0.pkl'
# ac2ic5 data:
#filename = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\my-data\\08-16-2020 (ac2ic n_fut_5)\\data\\B1_nrf_ac2ic.pkl'
file1 = open(filename, 'rb')
clusters_data = pickle.load(file1)
file1.close()
#need to add py_trained info to the neurons that didnt get it in t_2_ac2ic:
for j in range(len(clusters_data)):
    try:   
        clusters_data[j].py_trained
    except:
        clusters_data[j].py_trained = False

'''
Determining what neurons were trained
'''
trained_neurons = np.array([])
for i in range(len(clusters_data)):
    if clusters_data[i].py_trained:
        trained_neurons = np.append(trained_neurons, i)
print('Trained Neurons: \n')
print(trained_neurons)
trained_neurons = trained_neurons.astype(int)

'''
load useful data into np arrays
'''
py_data_max = np.array([])         #array storing best cc_norm for each neuron
py_full_data = np.array([])        #array storing the cc_norms for each gamma for each neuron
mse_py_color_array = np.zeros(len(clusters_data))      #Stores colour values for data point in py_data_max
ccnorm_py_color_array = np.zeros(len(clusters_data))


num_epochs = len(clusters_data[trained_neurons[0]].py.ic[0].ccnorm_h)
labels = []
for gamma_val in clusters_data[trained_neurons[0]].py.ic:
    labels.append(gamma_val.gamma_value) 

gamma_bins = np.zeros(len(labels))  

colors = []
cmap = plt.cm.get_cmap('gist_ncar')
#cmap = plt.cm.get_cmap('gist_rainbow')
z = np.array([])

for i in range(len(labels)):
    z = np.append(z, (i)/(len(labels)+1))

mse_data = np.zeros((len(clusters_data), len(labels)))        #3D array storing the cc_norm for run x neuron x gamma
mse_data[:] = np.nan
mse_data_tune = np.zeros((len(clusters_data), len(labels)))  
mse_data_tune[:] = np.nan
mse_data_epochs = np.zeros((len(clusters_data), len(labels)))
mse_data_epochs[:] = np.nan
mse_data_tune_mean = np.zeros((len(clusters_data), len(labels))) 

ccnorm_data = np.zeros((len(clusters_data), len(labels)))        #3D array storing the cc_norm for run x neuron x gamma
ccnorm_data[:] = np.nan
ccnorm_data_tune = np.zeros((len(clusters_data), len(labels)))  
ccnorm_data_tune[:] = np.nan
ccnorm_data_epochs = np.zeros((len(clusters_data), len(labels)))
ccnorm_data_epochs[:] = np.nan
ccnorm_data_tune_mean = np.zeros((len(clusters_data), len(labels))) 

#Load data into mse_data:
for j in range(len(clusters_data)):
    # mat_data = np.append(mat_data, clusters_data[j].mat.ic.cc_norm)
    if not clusters_data[j].py_trained:
                continue
    for k in range(len(labels)):
        mse_data_epochs[j,k] = clusters_data[j].py.ic[k].mse_model.best_epoch
        mse_data[j,k] = clusters_data[j].py.ic[k].mse_model.ccnorm
        mse_data_tune[j,k] = clusters_data[j].py.ic[k].tune_ccnorm_h[clusters_data[j].py.ic[k].mse_model.best_epoch]
        mse_data_tune_mean[j,k] = np.nanmean(clusters_data[j].py.ic[k].tune_ccnorm_h[clusters_data[j].py.ic[k].mse_model.best_epoch-100:clusters_data[j].py.ic[k].mse_model.best_epoch+100])
        
        ccnorm_data_epochs[j,k] = clusters_data[j].py.ic[k].ccnorm_model.best_epoch
        ccnorm_data[j,k] = clusters_data[j].py.ic[k].ccnorm_model.ccnorm
        ccnorm_data_tune[j,k] = clusters_data[j].py.ic[k].tune_ccnorm_h[clusters_data[j].py.ic[k].ccnorm_model.best_epoch]
        ccnorm_data_tune_mean[j,k] = np.nanmean(clusters_data[j].py.ic[k].tune_ccnorm_h[clusters_data[j].py.ic[k].ccnorm_model.best_epoch-100:clusters_data[j].py.ic[k].ccnorm_model.best_epoch+100])


mse_py_color_array_mean = np.zeros(len(clusters_data))      #Stores colour values for data point in py_data_max

#Calculating best gamma for each neuron for mse
mse_report_data_mean =  np.zeros(ccnorm_data.shape[0]) 
mse_report_data_mean[:] = np.nan
mse_tune_data_mean =  np.zeros(ccnorm_data.shape[0]) 
mse_tune_data_mean[:] = np.nan
mse_best_idxs = np.zeros(ccnorm_data.shape[0]) 
mse_best_idxs[:] = np.nan

for j in trained_neurons:
    k = np.where(mse_data_tune_mean[j,:] == np.nanmax(mse_data_tune_mean[j,:]))[0]       #selecting the best gamma
    if k.shape[0] == 0:                   #In the case of all nan.slice (i.e negative SP for neuron)
        mse_py_color_array_mean[j] = 0
        print('all runs for this neuron had cc_norm_tune of np.nan')
        continue
    gamma_bins[k] += 1
    mse_py_color_array_mean[j] = (int(k))/(len(labels)+1)
    mse_report_data_mean[j] =  mse_data[j, k]
    mse_tune_data_mean[j] = mse_data_tune[j,k]
    mse_best_idxs[j] = k

'''
generate the training set ccnorm data
'''
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

mat = scipy.io.loadmat('C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\michael-data\\hierarchy\\data\\A3_stimuli_with_idxes.mat', squeeze_me=True)
stimuli_mat = mat['stimuli']
stimuli = mat2py_copy.stimulus_unpacker(stimuli_mat)

if not ac2ic_fut:
    if input_area == 'ic':
        input_fht = ic_fht
    elif input_area == 'ac':
        input_fht = ac_fht
    inputs = np.empty([0,6400])
    for i in range(input_fht.shape[0]):
        inputs= np.concatenate((inputs, input_fht[i,:,:]),0)
    inputs = inputs.T                 #each row corresponds to one training pattern and the columns correspond to the historical input neurons.
    inputs = torch.from_numpy(inputs).type(torch.FloatTensor)     #Convert to torch.Tensor 
    
    stimuli.train_idx = stimuli.all_valid_data.no_folds.fit_idx[0:int(len(stimuli.all_valid_data.no_folds.fit_idx)*0.9)]
    stimuli.tune_idx = stimuli.all_valid_data.no_folds.fit_idx[int(len(stimuli.all_valid_data.no_folds.fit_idx)*0.9):]
    stimuli.report_idx = stimuli.all_valid_data.no_folds.pred_idx

elif ac2ic_fut:
    inputs = np.loadtxt('data/ac2ic_X_tfh_train.csv', delimiter=',')
    inputs = torch.from_numpy(inputs).type(torch.FloatTensor)
    
    stimuli.train_idx = np.loadtxt('data/ac2ic_train_idx.csv', delimiter=',')+1
inputs = inputs.to(device)

data_train = np.zeros((len(clusters_data), len(labels)))       
data_train[:] = np.nan
data_train_ccabs = np.zeros((len(clusters_data), len(labels)))       
data_train_ccabs[:] = np.nan
data_train_ccmax = np.zeros((len(clusters_data), len(labels)))       
data_train_ccmax[:] = np.nan
for neuron in trained_neurons:
    print('evaluating neruon {}'.format(neuron))
    for k in range(len(labels)):
        model = clusters_data[neuron].py.ic[k].mse_model.model
        model.to(device)
        model.eval()
        with torch.no_grad():
            if ac2ic_fut:
                out = model(inputs)
            else:
                out = model(inputs[stimuli.train_idx.astype(int)-1,:])
        ccabs, ccnorm , ccmax = cal_cc_norm.calculate(clusters_data[neuron].y_dt[:,stimuli.train_idx.astype(int)-1], 
                             out[:,0].cpu().detach().numpy())
        data_train[neuron, k] = ccnorm
        data_train_ccabs[neuron,k] = ccabs
        data_train_ccmax[neuron,k] = ccmax

best_train_ccabs = np.zeros(len(clusters_data))
best_train_ccabs[:] = np.nan
best_train_ccmax = np.zeros(len(clusters_data))
best_train_ccmax[:] = np.nan
for j in range(len(clusters_data)):
    if not math.isnan(mse_best_idxs[j]):
        best_train_ccabs[j] = data_train_ccabs[j, mse_best_idxs[neuron].astype(int)]
        best_train_ccmax[j]= data_train_ccmax[j, mse_best_idxs[neuron].astype(int)]
    
fig, ax1 = plt.subplots(figsize=(10,10)) 
xx = np.arange(-2,2,0.1)  
ax2 = ax1.twinx()
for j, neuron in enumerate(trained_neurons):
    ax1.scatter(j, best_train_ccabs[j], marker='o', c = 'tab:blue')
    ax2.scatter(j, best_train_ccmax[j], marker='o', color='tab:red')  
ax1.set_xlabel('Neuron', fontsize = 20)
ax1.set_ylabel('CC Abs', fontsize = 20, color = 'tab:blue')
ax1.set_ylim(-0.05, 1)
ax2.set_ylim(-0.05, 1)
ax1.tick_params(axis='y', labelcolor='tab:blue')
color = 'tab:blue'
ax2.set_ylabel('CC Max', color='tab:red', fontsize = 20)  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor='tab:red')

'''
direct comparison
'''
cmap = plt.cm.get_cmap('gist_ncar')

fig, ax1 = plt.subplots(figsize=(10,10)) 
xx = np.arange(-2,2,0.1)  
# ax1.scatter( best_train_ccmax,best_train_ccabs, marker='o', c = cmap(mse_py_color_array_mean))
ax1.scatter( best_train_ccmax,best_train_ccabs, marker='o', c = 'k')
ax1.plot(xx,xx)
ax1.set_xlabel('CCmax', fontsize = 20)
ax1.set_ylabel('CCabs', fontsize = 20)
ax1.set_ylim(-0.125, 1)
ax1.set_xlim(-0.125, 1)

outliers = []
for i in trained_neurons:
    if best_train_ccmax[i]>0.2 and best_train_ccabs[i]<0.2:
        print(i)
        outliers.append(i)
        
for i in outliers:
    epochs = clusters_data[i].py.ic[mse_best_idxs[i].astype(int)].mse_model.best_epoch
    print(epochs)

fig, ax1 = plt.subplots(figsize=(10,10)) 
for i in outliers:
    ax1.scatter(best_train_ccabs[i], mse_report_data_mean[i], c='k')
ax1.set_xlabel('Training CCabs', fontsize = 20)
ax1.set_ylabel('Report CCnorm', fontsize = 20)
    
#fig.savefig('C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Dissertation-Figures\\trainingsetdata_ic2ac.png', bbox_inches='tight')