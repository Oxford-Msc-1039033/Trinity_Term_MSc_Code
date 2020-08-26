# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:18:18 2020

@author: Tom
"""

import numpy as np
import scipy.io
import mat2py_copy
import matplotlib.pyplot as plt
import pickle
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import DivergingNorm
import torch

mapping = 'ic2mgb'

if mapping == 'ic2mgb':
    input_area = 'ic'
    clusters_path = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\my-data\\08-12-2020 (ic2mgb full L1squared MSE Dropout )\\data\\B1_nrf_ic2mgb_MSE_dropout_L1squared.pkl'
    
    a2a_ln_path = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\michael-data\\hierarchy\\data\\B1_clusters_mgb_a2a_kernels.mat'
    mat = scipy.io.loadmat(a2a_ln_path, squeeze_me=True)
    LN_full_info = mat['clusters_mgb']
    
    coch_ln_path = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\michael-data\\hierarchy\\data\\B1_clusters_mgb_coch_kernels.mat'
    mat = scipy.io.loadmat(coch_ln_path, squeeze_me=True)
    LN_full_info_coch = mat['clusters_mgb'] 
    
if mapping == 'ic2ac':
    input_area = 'ic'
    clusters_path = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\my-data\\08-16-2020 (ic2ac)\\data\\B1_nrf_ic2ac.pkl'
    a2a_ln_path = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\michael-data\\hierarchy\\data\\B1_clusters_ac_a2a_kernels.mat'
    mat = scipy.io.loadmat(a2a_ln_path, squeeze_me=True)
    LN_full_info = mat['clusters_mgb']
    
    coch_ln_path = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\michael-data\\hierarchy\\data\\B1_clusters_ac_coch_kernels.mat'
    mat = scipy.io.loadmat(coch_ln_path, squeeze_me=True)
    LN_full_info_coch = mat['clusters_mgb'] 

if mapping == 'ac2ic0':
    input_area = 'ac'
    clusters_path = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\my-data\\08-18-2020 (ac2ic n_fut_0)\\data\B1_nrf_ac2ic_nfut0.pkl'
    a2a_ln_path = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\michael-data\\hierarchy\\data\\B1_clusters_ic_a2a_kernels.mat'
    mat = scipy.io.loadmat(a2a_ln_path, squeeze_me=True)
    LN_full_info = mat['clusters_mgb']
    
    coch_ln_path = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\michael-data\\hierarchy\\data\\B1_clusters_ic_coch_kernels.mat'
    mat = scipy.io.loadmat(coch_ln_path, squeeze_me=True)
    LN_full_info_coch = mat['clusters_mgb'] 
    
if mapping == 'ac2ic0':
    input_area = 'ac'
    clusters_path = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\my-data\\08-16-2020 (ac2ic n_fut_5)\\data\B1_nrf_ac2ic.pkl'
    a2a_ln_path = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\michael-data\\hierarchy\\data\\B1_clusters_ic_a2a_kernels.mat'
    mat = scipy.io.loadmat(a2a_ln_path, squeeze_me=True)
    LN_full_info = mat['clusters_mgb']
    
    coch_ln_path = 'C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\michael-data\\hierarchy\\data\\B1_clusters_ic_coch_kernels.mat'
    mat = scipy.io.loadmat(coch_ln_path, squeeze_me=True)
    LN_full_info_coch = mat['clusters_mgb'] 

count = 0
LN_data_coch = np.array([])          #this is for a different kernel - maybe make a library that grabs the appropriate ones.
for i in range(LN_full_info.size):
    current = np.array([])
    current = np.append(current, LN_full_info_coch[i]['all_data_coch_kernels'][()]['separable_kernel'][()]['ln_model'][()]['cc_norm_pred'])
    current = np.append(current, LN_full_info_coch[i]['all_data_coch_kernels'][()]['rank_n_kernel'][()]['ln_model'][()]['cc_norm_pred'])
    current = np.append(current, LN_full_info_coch[i]['all_data_coch_kernels'][()]['elasticnet_kernel'][()]['ln_model'][()]['cc_norm_pred'])
    if count == 0:
        LN_data_coch = np.expand_dims(current, axis = 1)
    else:
        LN_data_coch = np.append(LN_data_coch, np.expand_dims(current, axis = 1), axis = 1)
    count +=1
LN_data_coch = np.nanmax(LN_data_coch, axis = 0)
        
LN_data = np.zeros(len(LN_full_info))         
for i in range(len(LN_full_info)):
    if input_area == 'ic':
        LN_data[i] = LN_full_info[i]['all_data_a2a_kernels'][()]['ic_kernel'][()]['cc_norm_pred']
    elif input_area == 'ac':
        LN_data[i] = LN_full_info[i]['all_data_a2a_kernels'][()]['ac_kernel'][()]['cc_norm_pred']
        
file1 = open(clusters_path, 'rb')
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
np.savetxt('trained_neurons_ic2ac.csv', trained_neurons, delimiter=',')   #save the indxs trained on.


mat_data = np.array([])
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
    mat_data = np.append(mat_data, clusters_data[j].mat.ic.cc_norm)
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

#Calculating best gamma for each neuron across all runs for ccnorm
ccnorm_report_data =  np.zeros(ccnorm_data.shape[0]) 
ccnorm_report_data[:] = np.nan
ccnorm_best_idxs_tune = np.zeros((ccnorm_data.shape[0]))    #holds the run and gamma idxs of chosen models
ccnorm_best_idxs_tune[:] = np.nan
ccnorm_best_idxs_report = np.zeros((ccnorm_data.shape[0]))
ccnorm_best_idxs_report[:] = np.nan
 
for j in trained_neurons:
    if clusters_data[j].py_trained:
        k = np.where(ccnorm_data_tune[j,:] == np.nanmax(ccnorm_data_tune[j,:]))[0]       #selecting the best gamma
        if k.shape[0] > 1:
            k = np.array([k[-1]])
        k_report = np.where(ccnorm_data[j,:] == np.nanmax(ccnorm_data[j,:]))[0]       #selecting the best gamma
        
        if k.shape[0] == 0:                   #In the case of all nan.slice (i.e negative SP for neuron)
            ccnorm_py_color_array[j] = 0
            print('all runs for this neuron had cc_norm_tune of np.nan')
            ccnorm_best_idxs_tune[j] = np.nan
            continue
        
        if not k_report.shape[0] == 0:
            ccnorm_best_idxs_report[j] =  k_report
    
        gamma_bins[k] += 1
        ccnorm_py_color_array[j] = (int(k))/(len(labels)+1)
        ccnorm_report_data[j] =  ccnorm_data[j, k]
        ccnorm_best_idxs_tune[j] = k

#Calculating best gamma for each neuron across all runs for mse
mse_report_data =  np.zeros(ccnorm_data.shape[0]) 
mse_report_data[:] = np.nan
mse_best_idxs_tune = np.zeros((mse_data.shape[0]))    #holds the run and gamma idxs of chosen models
mse_best_idxs_tune[:] = np.nan
mse_best_idxs_report = np.zeros((mse_data.shape[0]))
mse_best_idxs_report[:] = np.nan

for j in trained_neurons:
    k = np.where(mse_data_tune[j,:] == np.nanmax(mse_data_tune[j,:]))[0]       #selecting the best gamma
    k_report = np.where(mse_data[j,:] == np.nanmax(mse_data[j,:]))[0]       #selecting the best gamma
    if k.shape[0] == 0:                   #In the case of all nan.slice (i.e negative SP for neuron)
        mse_py_color_array[j] = 0
        print('all runs for this neuron had cc_norm_tune of np.nan')
        mse_best_idxs_tune[j] = np.nan
        continue
    
    if not k_report.shape[0] == 0:
        mse_best_idxs_report[j] =  k_report

    gamma_bins[k] += 1
    mse_py_color_array[j] = (int(k))/(len(labels)+1)
    mse_report_data[j] =  mse_data[j, k]
    mse_best_idxs_tune[j] = k

count = 0
count_matrix = np.zeros((mse_data.shape[1], mse_data.shape[1]))
for j in range(mse_data.shape[0]):
    if (not math.isnan(mse_best_idxs_tune[j])) and (not math.isnan(mse_best_idxs_report[j])):
        count_matrix[int(mse_best_idxs_tune[j]), int(mse_best_idxs_report[j])] +=1
    if mse_best_idxs_tune[j] == mse_best_idxs_report[j]:
        count +=1
max_count = np.nanmax(count_matrix)


'''
best average gamma comparison between tuning and reporting sets
'''
fig1 =plt.figure(figsize=(9, 9)) 
xx = np.arange(-2,12,0.1)  
ax1 = fig1.add_subplot(1,1,1) 
for i in range(count_matrix.shape[1]):
    for j in range(count_matrix.shape[1]):
        if not count_matrix[i,j] ==0:
            ax1.scatter(i,j, marker='o', c = cmap((count_matrix[i,j])/(max_count+1)))
for k in range(1,int(max_count),3):
    ax1.scatter(np.nan, np.nan, marker='o', c = cmap((k)/(max_count+1)), label = k)
ax1.plot(xx,xx)
ax1.set_xlabel('Tuning Set', fontsize = 20)
ax1.set_ylabel('Reporting Set', fontsize = 20)
ax1.set_ylim([-0.1, 9+0.05])
ax1.set_xlim([-0.1, 9+0.05])
ax1.set_xticks(np.arange(mse_data.shape[1]))
ax1.set_yticks(np.arange(mse_data.shape[1]))
ax1.set_xticklabels(labels) 
ax1.set_yticklabels(labels) 
ax1.set_title('Best average gamma value per neuron for tuning vs reporting set', fontsize=20)
ax1.legend()    
#
#max_lim =  np.nanmax(np.append(mat_data, gamma_j)).astype(np.float)
#min_lim = np.nanmin(np.append(mat_data, gamma_j)).astype(np.float)
#xx = np.arange(-2, 2, 0.01)
#figmax =plt.figure(figsize=(9, 9))   
#axmax = figmax.add_subplot(1,1,1) 
#for j in range(mse_data.shape[0]):
#    if not math.isnan(gamma_j[j]):
#        axmax.scatter(mat_data[j], gamma_j[j], marker='o', c = cmap(py_color_array[j]))
#for k in range(len(labels)):
#    axmax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
#axmax.plot(xx,xx)
#axmax.set_xlabel('Matlab NRF', fontsize = 20)
#axmax.set_ylabel('Python', fontsize = 20)
#axmax.set_ylim([min_lim-0.05,max_lim+0.05])
#axmax.set_xlim([min_lim-0.05,max_lim+0.05])
#axmax.set_title('Best NRF vs Matlab NRF', fontsize=20)
#axmax.legend()
#
#
##
#max_lim =  np.nanmax(np.append(LN_datans, gamma_j)).astype(np.float)
#min_lim = np.nanmin(np.append(LN_datans, gamma_j)).astype(np.float)
#fig =plt.figure(figsize=(9, 9))   
#ax = fig.add_subplot(1,1,1) 
#for j in range(mse_data.shape[0]):
#    if not math.isnan(gamma_j[j]):
#        ax.scatter(LN_datans[j], gamma_j[j], marker='o', c = cmap(py_color_array[j]))
#for k in range(len(labels)):
#    ax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
#ax.plot(xx,xx)
#ax.set_xlabel('Matlab LN', fontsize = 20)
#ax.set_ylabel('Python', fontsize = 20)
#ax.set_ylim([min_lim-0.05,max_lim+0.05])
#ax.set_xlim([min_lim-0.05,max_lim+0.05])
#ax.set_title('Best NRF vs Matlab LN (PRED)', fontsize=20)
#ax.legend()

#Plotting average cc_norm by gamma val
#fig1 =plt.figure(figsize=(9, 9))   
#ax1 = fig1.add_subplot(1,1,1) 
#for k in range(len(labels)):
#    color_vector = np.zeros((1, 4))
#    for n in range(4):
#        color_vector[:,n] = cmap(z[k])[n]
#    ax1.scatter(k*np.ones(mse_data.shape[0]), np.nanmax(mse_data, axis = 0)[:,k], marker='o', c = color_vector , label = labels[k])
#    ax1.scatter(k, np.nanmean(np.nanmax(mse_data, axis = 0)[:,k]), marker='o', color = 'k' ,s=80)
#ax1.set_xlabel('gamma val', fontsize = 20)
#ax1.set_ylabel('cc norm', fontsize = 20)
#ax1.set_xlim([0-0.5,len(gamma_bins)-0.5])
#ax1.set_xticks(np.arange(0,len(labels),1))
#ax1.set_xticklabels(labels)
#ax1.set_title('mean cc norm by best gamma value', fontsize=20)
#ax1.legend()

#fig1.savefig('cc_norm_by_gamma_cmap.png')
#figmax.savefig('mat_vs_py_max_cmap.png')
#fig.savefig('mat_vs_py_all_cmap.png')

#######
'''
TUNING SET RAINBOW PLOT
'''

#cmap_grey = plt.cm.get_cmap('gist_rainbow')
#fig3 =plt.figure(figsize=(9, 9))   
#ax3 = fig3.add_subplot(1,1,1) 
#k = 9
#for j in range(mse_data.shape[0]):
#    for k in range(mse_data.shape[1]):
#        ax3.scatter(j*np.ones(mse_data_tune.shape[0]), mse_data_tune[j,k], marker='o', c = cmap((int(k)+1)/len(labels)))
#for j in range(mse_data.shape[0]):
#    for k in range(mse_data.shape[1]):
#        for i in range(mse_data.shape[0]):
#            if i == best_idxs_tune[j,0] and k == best_idxs_tune[j,1]:
#                ax3.scatter(j, mse_data_tune[j,k], marker='o',s=100 ,c = 'k')
#for k in range(len(labels)):
#    ax3.scatter(np.nan, np.nan, marker='o' , c = cmap((int(k)+1)/len(labels)) , label = 'gamma %s' % labels[k])
#ax3.set_xlabel('Neuron', fontsize = 20)
#ax3.set_ylabel('cc_norm', fontsize = 20)
#ax3.set_title('Plot of TUNE cc_norm for models', fontsize=20)
#ax3.legend()


###############
#
#cmap_grey = plt.cm.get_cmap('YlOrRd')
#fig4 =plt.figure(figsize=(9, 9))   
#ax4 = fig4.add_subplot(1,1,1) 
#for i in range(len(labels)):
#    ax4.scatter(i, py_full_data[100 ,i], marker='o', c = cmap_grey(z[i]))
#for i in range(len(labels)):
#    ax4.scatter(np.nan, np.nan, marker='o', c = cmap_grey(z[i]) , label = labels[i])
#ax4.set_xlabel('gamma', fontsize = 20)
#ax4.set_ylabel('cc_norm', fontsize = 20)
#ax4.set_xticks(np.arange(len(labels)))
#ax4.set_xticklabels(labels)
#ax4.set_title('cc norm vs gamma for neuron 48'.format(clusters_mgb[neurons[0]].py.ic[0].hyper_type, clusters_mgb[neurons[0]].py.ic[0].num_hidden), fontsize=20)
#ax4.legend()

#######
'''
ploting best strfs
'''
#importing BF data
#append the unique_id for each unit.
mat = scipy.io.loadmat('C:\\Users\\thble\\OneDrive\\Trinity_term_project\\Python-Code\\michael-data\\hierarchy\\data\\B1_clusters_ic_a2a_nrf.mat', squeeze_me=True)
clusters_ic_mat = mat['clusters_ic']
ic_unique_ids = []
for d in clusters_ic_mat:
    ic_unique_ids.append('%s-e%03d-p%03d-c%03d' % (d['brainArea'].lower(), d['expt'], d['pen'], d['clusterIdx'])
                         )
with open('data/res-bfs.pkl', 'rb') as f:
    bf_data = pickle.load(f)
    
bf_idx = []
bf_hz = []
for x in bf_data:
    if x['unique_id'][0:2] == input_area:
        bf_idx.append(x['coch_kernels']['SeparableKernel']['bf_idx'])
        bf_hz.append(x['coch_kernels']['SeparableKernel']['bf_hz'])
order = np.argsort(bf_idx)

def show_strf(strf, neuron='NA', gamma='NA', epochs='NA',  area ='NA', input_area='NA', cc_norm='NA', save_fig = False):
    my_dpi = 120
    mx = max([np.max(strf), np.abs(np.min(strf))])
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(strf, norm=DivergingNorm(0), cmap = 'seismic', vmin=-mx, vmax=mx)
    fig.colorbar(im, cax=cax, orientation = 'vertical')
    xcoords = np.arange(14.5,150,15)
    for xc in xcoords:
        ax.axvline(x=xc, c='k', linewidth=0.1)
    ax.set_title('neuron: {}, gamma:{}, \n epochs:{}, cc norm: {:.4f}'.format(neuron, gamma, epochs, cc_norm), fontsize=5)
    plt.show()
    if save_fig:
        plt.savefig('RFs/rf_mse_{}2{}_neuron{}_unordered.png'.format(input_area, area, neuron), 
                    dpi=my_dpi*20, bbox_inches='tight')

# for j in range(mse_data.shape[0]):
# for j in [trained_neurons[11]]:
#     for k in range(mse_data.shape[1]):
#         if  k == mse_best_idxs_tune[j] and (not math.isnan(mse_data[j,k])):
#             model = clusters_data[j].py.ic[k].mse_model.model
#             cc_norm = clusters_data[j].py.ic[k].mse_model.ccnorm
#             gamma = clusters_data[j].py.ic[k].gamma_value
#             epochs = clusters_data[j].py.ic[k].mse_model.best_epoch
#             strf = np.zeros((448, 10*15))
#             for ii in range(448):
#                 for jj in range(10):
#                     sign = np.sign(model.layer2.weight.data.numpy()[0,jj])
#                     strf[ii,jj*15:(jj+1)*15] = sign*model.layer1.weight.data.numpy()[jj,ii*15:(ii+1)*15]
#             #strf = strf[order,:]
#             show_strf(strf, neuron=j, gamma=gamma, epochs=epochs, cc_norm=cc_norm, input_area='ic', area='ac', save_fig=True)
#             plt.close('all') 

'''
saving epochs for each 'best' model'
''' 
epochs_data = np.zeros(mse_data.shape[0])
epochs_data[:] = np.nan
epochs_color_array = np.zeros(mse_data.shape[0])
for j in range(mse_data.shape[0]):   
    for k in range(mse_data.shape[1]):
        if k == mse_best_idxs_tune[j] and (not math.isnan(mse_data[j,k])): 
            epochs_color_array[j] = (int(k)+1)/(len(labels)+2)
            epochs_data[j] = mse_data_epochs[j,k]

fig, ax =plt.subplots(figsize=(9, 9)) 
for j in range(mse_data.shape[0]):
    if not math.isnan(epochs_data[j]):
        ax.scatter(j, epochs_data[j], marker='o', c = cmap(epochs_color_array[j]))
for k in range(len(labels)):
    ax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
#ax.plot(xx,xx)
ax.set_xlabel('Neuron', fontsize = 20)
ax.set_ylabel('Best Epoch', fontsize = 20)
#ax.set_ylim([min_lim-0.05,max_lim+0.05])
#ax.set_xlim([min_lim-0.05,max_lim+0.05])
ax.set_title('Plot of best epoch per neuron', fontsize=20)
ax.legend()     


'''
plot ccnorm_report vs epochs trained
'''
fig, ax =plt.subplots(figsize=(9, 9)) 
for j in range(mse_data.shape[0]):
    if not math.isnan(epochs_data[j]):
        ax.scatter(epochs_data[j], mse_report_data[j], marker='o', c = cmap(epochs_color_array[j]))
for k in range(len(labels)):
    ax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
ax.set_xlabel('Best_Epochs', fontsize = 20)
ax.set_ylabel('CC norm (report)', fontsize = 20)

ax.set_title('best epoch vs CC nrom', fontsize=20)
ax.legend()          
              
                
######
                
'''
Plotting Neural responses on tuning set
'''
#mat = scipy.io.loadmat('C:\\Users\\Tom\\OneDrive\\Trinity_term_project\\Python-Code\\michael-data\\hierarchy\\data\A3_stimuli_with_idxes.mat', squeeze_me=True)
#stimuli_mat = mat['stimuli']
#stimuli = mat2py_copy.stimulus_unpacker(stimuli_mat)
#
##Define the training, tuning, and reporting sets via their indices.
#stimuli.train_idx = stimuli.all_valid_data.no_folds.fit_idx[0:int(len(stimuli.all_valid_data.no_folds.fit_idx)*0.9)]
#stimuli.tune_idx = stimuli.all_valid_data.no_folds.fit_idx[int(len(stimuli.all_valid_data.no_folds.fit_idx)*0.9):]
#stimuli.report_idx = stimuli.all_valid_data.no_folds.pred_idx
#
#neuron = trained_neurons[0]
#y_hat = clusters_data[neuron].py.ic[int(mse_best_idxs_report[neuron,1])].mse_model.yhat[:,0].cpu().detach().numpy()
##model = clusters_data[neuron].py.ic[int(best_idxs_tune[neuron,1])].model
#       
#y = clusters_data[neuron].y_t[stimuli.report_idx.astype(int)-1]
#
#fig, ax =plt.subplots(figsize=(9, 9)) 
#ax.plot(25*np.arange(y.size), y_hat, c='r', label = 'predicted')
##scaling = 1e-5
##ax.plot(25*np.arange(y.size), scaling*y+np.mean(y_hat) - np.mean(scaling*y), c='b', label = 'recorded')
#ax.plot(25*np.arange(y.size), y, c='b', label = 'recorded')
#ax.set_xlabel('Time (ms)', fontsize = 20)
#ax.set_ylabel(r' Average Spikes per Bin (25ms$^{-1}$)', fontsize = 20)
#ax.set_title('Predicted and Recorded PSTH', fontsize=20)
#ax.set_xlim([0, 25*y.size])
#ax.legend()       
#
#import cal_cc_norm
#cc_abs, cc_norm ,cc_max = cal_cc_norm.calculate(clusters_data[neuron].y_dt[:,stimuli.report_idx.astype(int)-1],  y_hat) 


######
'''
ploting ST of stimulus
'''
#fig, ax =plt.subplots() 
#divider = make_axes_locatable(ax)
#cax = divider.append_axes('right', size='3%', pad=0.05)
##im = ax.imshow(stimuli.X_ft[:,100:200], cmap = 'jet', origin='lower')
#im = ax.imshow(stimuli.X_ft, cmap = 'jet', origin='lower')
#ax.set_xticks(np.arange(0,100,20))
#ax.set_yticks(np.arange(0,25,12))
#ax.set_xticklabels(25*np.arange(0,100,20)) 
#ax.set_yticklabels([1000, 8000, 64000]) 
#ax.set_xlabel('Time (ms)', fontsize=30)
#ax.set_ylabel('Frequency (kHz)', fontsize=30)
#cbar = plt.colorbar(im, cax=cax, orientation = 'vertical')
#cbar.set_label('Level (dB SPL)', fontsize=25)

''' 
plotting ccnorm history
'''

def low_pass_filter(data, n_h = 10):
    out = np.zeros(len(data))
    out[:]=np.nan
    for i in range(len(data)):
        if i-n_h<0:
            lpf_i = np.nanmean(data[:i+n_h+1])
        elif i+n_h+1 >len(data):
            lpf_i = np.nanmean(data[i-n_h:])

        else:
            lpf_i = np.nanmean(data[i-n_h:i+n_h+1])
        out[i] = lpf_i
    return out

def plot_data(neuron = None, data= None, labels = None, colors = None, title = None, yaxis = None, legend=False,
              highlight_min = False, highlight_max = False, show_selected = False, selected_idxs = None):
    
    if data.ndim ==2:
        x = np.arange(data.shape[1])
    elif data.ndim == 1:
        x = np.arange(data.shape[0])
    else:
        print('incorrect shape for data matrix. needs to be 1 or 2 dim')
        
    if not labels:
        labels = ['data0', 'data1','data2','data3','data4','data5','data6','data7','data8',]
    if not colors:
        colors = np.arange(9)/9
    cmap = plt.cm.get_cmap('gist_ncar')
    fig, ax =plt.subplots(figsize=(9, 9)) 
    if data.ndim ==2:
        for i in range(data.shape[0]):
            ax.plot(x, data[i,:], label = labels[i], color = cmap(colors[i]), zorder = i)
        for i in range(data.shape[0]):
            if highlight_min:
                idx = np.where(data[i,:] == np.nanmin(data[i,:]))[0]
                ax.scatter(x[idx], data[i,idx], marker='o', s = 50, c= np.expand_dims(np.array(cmap(colors[i])), axis=0))
            if highlight_max:
                idx = np.where(data[i,:] == np.nanmax(data[i,:]))[0]
                ax.scatter(x[idx], data[i,idx], marker='o', s = 50, c= np.expand_dims(np.array(cmap(colors[i])), axis=0))
            if show_selected:
                ax.scatter(x[int(selected_idxs[i])], data[i,int(selected_idxs[i])], marker='o', s = 200, 
                             c= np.expand_dims(np.array(cmap(colors[i])), axis=0), zorder = 10+i)
    elif data.ndim == 1:
        ax.plot(x, data, label = labels, color = cmap(colors[0]))
    
    ax.set_xlabel('Epoch', fontsize = 20)
    if not yaxis:
        ax.set_ylabel('variable', fontsize = 20)
    else:
        ax.set_ylabel(yaxis, fontsize = 20)
    if legend:
        ax.legend(title = 'Gamma', bbox_to_anchor=(0.975, 0.025), 
                  loc='lower right', borderaxespad=0.,  prop={'size':12}, framealpha = 1)
        plt.rcParams['legend.title_fontsize'] = 15
    if title:
        ax.set_title(title, fontsize=20)
    plt.show()
    
neuron = trained_neurons[0]
#for neuron in trained_neurons.astype(int):
data_tune = np.zeros((len(labels),num_epochs))
data_tune[:] = np.nan
for i in range(len(labels)):
    data_tune[i,:] = clusters_data[neuron].py.ic[i].tune_ccnorm_h
    
plot_data(data = data_tune, labels = labels,  show_selected = True,
          selected_idxs =mse_data_epochs[neuron,:], legend=True, yaxis = 'CC Norm')

data_report = np.zeros((len(labels),num_epochs))
data_report[:] = np.nan
for i in range(len(labels)):
    data_report[i,:] = clusters_data[neuron].py.ic[i].ccnorm_h
    
plot_data(data=data_report, labels = labels, show_selected = True,
          selected_idxs =mse_data_epochs[neuron,:] , yaxis = 'CC Norm')

data_tune = np.zeros((len(labels),num_epochs))
data_tune[:] = np.nan
for i in range(len(labels)):
    data_tune[i,:] = low_pass_filter(clusters_data[neuron].py.ic[i].tune_ccnorm_h, n_h=30)
    
plot_data(data = data_tune, labels = labels,  show_selected = True,
          selected_idxs =mse_data_epochs[neuron,:],  yaxis = 'CC Norm')

data_report = np.zeros((len(labels),num_epochs))
data_report[:] = np.nan
for i in range(len(labels)):
    data_report[i,:] = low_pass_filter(clusters_data[neuron].py.ic[i].ccnorm_h, n_h=30)
    
plot_data(data=data_report, labels = labels, show_selected = True,
          selected_idxs =mse_data_epochs[neuron,:],  yaxis = 'CC Norm')


'''
Avereged ccnorm selection method - MSE Method
'''
mse_py_color_array_mean = np.zeros(len(clusters_data))      #Stores colour values for data point in py_data_max
ccnorm_py_color_array_mean = np.zeros(len(clusters_data))

#Calculating best gamma for each neuron for ccnorm
ccnorm_report_data_mean =  np.zeros(ccnorm_data.shape[0]) 
ccnorm_report_data_mean[:] = np.nan
ccnorm_tune_data_mean =  np.zeros(ccnorm_data.shape[0]) 
ccnorm_tune_data_mean[:] = np.nan
 
for j in trained_neurons:
    if clusters_data[j].py_trained:
        k = np.where(ccnorm_data_tune_mean[j,:] == np.nanmax(ccnorm_data_tune_mean[j,:]))[0]       #selecting the best gamma
        if k.shape[0] > 1:
            k = np.array([k[-1]])        
        if k.shape[0] == 0:                   #In the case of all nan.slice (i.e negative SP for neuron)
            ccnorm_py_color_array_mean[j] = 0
            print('all runs for this neuron had cc_norm_tune of np.nan')
            continue
    
        gamma_bins[k] += 1
        ccnorm_py_color_array_mean[j] = (int(k))/(len(labels)+1)
        ccnorm_report_data_mean[j] =  ccnorm_data[j, k]
        ccnorm_tune_data_mean[j] = ccnorm_data_tune[j,k]

#Calculating best gamma for each neuron for mse
mse_report_data_mean =  np.zeros(ccnorm_data.shape[0]) 
mse_report_data_mean[:] = np.nan
mse_tune_data_mean =  np.zeros(ccnorm_data.shape[0]) 
mse_tune_data_mean[:] = np.nan

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
 
    
'''
vs a2a LN
'''
max_lim = 0
relevant_data = np.array([])
for j in range(mse_data.shape[0]):
    if clusters_data[j].py_trained:
        if not math.isnan(LN_data[j]) and not math.isnan(mse_report_data_mean[j]):
            max_lim = np.nanmax([max_lim,LN_data[j], mse_report_data_mean[j]])
            relevant_data = np.append(relevant_data, LN_data[j])
min_lim = 0
for j in range(mse_data.shape[0]):
    if clusters_data[j].py_trained:
        if not math.isnan(LN_data_coch[j]) and not math.isnan(mse_report_data_mean[j]):
            min_lim = np.nanmin([min_lim,LN_data[j], mse_report_data[j]])
            
fig =plt.figure(figsize=(9, 9)) 
xx = np.arange(-2,2,0.1)  
ax = fig.add_subplot(1,1,1) 
for j in range(mse_data.shape[0]):
    if not math.isnan(mse_report_data[j]):
        ax.scatter(LN_data[j], mse_report_data_mean[j], marker='o', c = cmap(mse_py_color_array[j]))
for k in range(len(labels)):
    ax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
ax.axvline(x=np.nanmedian(relevant_data), c='r', linewidth=0.5)
ax.axhline(y=np.nanmedian(mse_report_data_mean), c='r', linewidth=0.5)
ax.scatter(np.nanmedian(relevant_data), np.nanmedian(mse_report_data_mean), marker = 'o', s = 100, c='r')
ax.plot(xx,xx)
ax.set_xlabel('LN ic-gram model', fontsize = 20)
ax.set_ylabel('Pytorch ic-gram model', fontsize = 20)
ax.set_ylim([min_lim-0.05,max_lim+0.05])
ax.set_xlim([min_lim-0.05,max_lim+0.05])
#ax.set_title('ic2ac model vs Matlab ic2ac LN', fontsize=20)
ax.legend() 

'''
vs a2a NRF
'''
            
fig =plt.figure(figsize=(9, 9)) 
xx = np.arange(-2,2,0.1)  
ax = fig.add_subplot(1,1,1) 
for j in range(mse_data.shape[0]):
    if not math.isnan(mse_report_data[j]):
        ax.scatter(LN_data[j], mse_report_data_mean[j], marker='o', c = cmap(mse_py_color_array[j]))
for k in range(len(labels)):
    ax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
ax.axvline(x=np.nanmedian(relevant_data), c='r', linewidth=0.5)
ax.axhline(y=np.nanmedian(mse_report_data_mean), c='r', linewidth=0.5)
ax.scatter(np.nanmedian(relevant_data), np.nanmedian(mse_report_data_mean), marker = 'o', s = 100, c='r')
ax.plot(xx,xx)
ax.set_xlabel('LN ic-gram model', fontsize = 20)
ax.set_ylabel('Pytorch ic-gram model', fontsize = 20)
ax.set_ylim([min_lim-0.05,max_lim+0.05])
ax.set_xlim([min_lim-0.05,max_lim+0.05])
#ax.set_title('ic2ac model vs Matlab ic2ac LN', fontsize=20)
ax.legend() 
  
max_lim =  np.nanmax(np.append(mse_report_data, mse_report_data_mean)).astype(np.float)
min_lim = np.nanmin(np.append(mse_report_data, mse_report_data_mean)).astype(np.float)
fig =plt.figure(figsize=(9, 9)) 
xx = np.arange(-2,2,0.1)  
ax = fig.add_subplot(1,1,1) 
for j in range(mse_data.shape[0]):
    if not math.isnan(mse_report_data[j]):
        ax.scatter(mse_report_data[j], mse_report_data_mean[j], marker='o', c = cmap(mse_py_color_array[j]))
for k in range(len(labels)):
    ax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
ax.plot(xx,xx)
ax.set_xlabel('Original Method CCnorm', fontsize = 20)
ax.set_ylabel('Mean Method CCnorm', fontsize = 20)
ax.set_ylim([min_lim-0.05,1+0.05])
ax.set_xlim([min_lim-0.05,1+0.05])
ax.set_title('Mean vs Original MSE Selection Method ic2ac', fontsize=20)
ax.legend()

'''
Avereged ccnorm selection method - CCnorm method
'''
    
max_lim =  np.nanmax(np.append(LN_data, mse_report_data_mean)).astype(np.float)
min_lim = np.nanmin(np.append(LN_data, mse_report_data_mean)).astype(np.float)
fig =plt.figure(figsize=(9, 9)) 
xx = np.arange(-2,2,0.1)  
ax = fig.add_subplot(1,1,1) 
for j in range(mse_data.shape[0]):
    if not math.isnan(mse_report_data[j]):
        ax.scatter(LN_data[j], ccnorm_report_data_mean[j], marker='o', c = cmap(mse_py_color_array_mean[j]))
for k in range(len(labels)):
    ax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
ax.plot(xx,xx)
ax.set_xlabel('Matlab LN CCnorm', fontsize = 20)
ax.set_ylabel('Mean Method CCnorm', fontsize = 20)
ax.set_ylim([min_lim-0.05,1+0.05])
ax.set_xlim([min_lim-0.05,1+0.05])
ax.set_title('Mean Method ic2ac model vs Matlab ic2ac LN', fontsize=20)
ax.legend()

max_lim =  np.nanmax(np.append(ccnorm_report_data, ccnorm_report_data_mean)).astype(np.float)
min_lim = np.nanmin(np.append(ccnorm_report_data, ccnorm_report_data_mean)).astype(np.float)
fig =plt.figure(figsize=(9, 9)) 
xx = np.arange(-2,2,0.1)  
ax = fig.add_subplot(1,1,1) 
for j in range(mse_data.shape[0]):
    if not math.isnan(mse_report_data[j]):
        ax.scatter(ccnorm_report_data[j], ccnorm_report_data_mean[j], marker='o', c = cmap(mse_py_color_array[j]))
for k in range(len(labels)):
    ax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
ax.plot(xx,xx)
ax.set_xlabel('Original Method CCnorm', fontsize = 20)
ax.set_ylabel('Mean Method CCnorm', fontsize = 20)
ax.set_ylim([min_lim-0.05,1+0.05])
ax.set_xlim([min_lim-0.05,1+0.05])
ax.set_title('Mean vs Original Selection Method ic2ac CCnorm', fontsize=20)
ax.legend()


'''
MSE vs CCnorm
'''

max_lim =  np.nanmax(np.append(ccnorm_report_data_mean, mse_report_data_mean)).astype(np.float)
min_lim = np.nanmin(np.append(ccnorm_report_data_mean, mse_report_data_mean)).astype(np.float)
fig =plt.figure(figsize=(9, 9)) 
xx = np.arange(-2,2,0.1)  
ax = fig.add_subplot(1,1,1) 
for j in range(mse_data.shape[0]):
    if not math.isnan(ccnorm_report_data_mean[j]) and not math.isnan(mse_report_data_mean[j]):
        ax.scatter(ccnorm_report_data_mean[j], mse_report_data_mean[j], marker='o', c = cmap(ccnorm_py_color_array_mean[j]))
for k in range(len(labels)):
    ax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
ax.plot(xx,xx)
ax.set_xlabel('CCnorm Early Stopping', fontsize = 20)
ax.set_ylabel('MSE Early Stopping', fontsize = 20)
ax.set_ylim([min_lim-0.05,max_lim+0.05])
ax.set_xlim([min_lim-0.05,max_lim+0.05])
ax.axvline(x=np.nanmedian(ccnorm_report_data_mean), c='r', linewidth=0.5)
ax.axhline(y=np.nanmedian(mse_report_data_mean), c='r', linewidth=0.5)
ax.scatter(np.nanmedian(ccnorm_report_data_mean), np.nanmedian(mse_report_data_mean), marker = 'o', s = 100, c='r')
# ax.set_title('Comparison of CCnorm and MSE method for Dropout L1 squared model', fontsize=20)
ax.legend(title = 'Gamma', bbox_to_anchor=(0.975, 0.025), 
                  loc='lower right', borderaxespad=0.,  prop={'size':12}, framealpha = 1)
plt.rcParams['legend.title_fontsize'] = 15

'''
plot tune vs report generalization - MSE
'''
max_lim =  1.0
min_lim = -0.2
xx = np.arange(-2,2,0.1) 

fig, ax =plt.subplots(figsize=(9, 9)) 
for j in range(mse_data.shape[0]):  
    ax.scatter(mse_tune_data_mean[j], mse_report_data_mean[j], marker='o', c = cmap(mse_py_color_array_mean[j]))
for k in range(len(labels)):
    ax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
    
ax.plot(xx,xx)
ax.set_xlabel('Tuning Set', fontsize = 20)
ax.set_ylabel('Reporting Set', fontsize = 20)
ax.set_ylim([min_lim-0.05,max_lim+0.05])
ax.set_xlim([min_lim-0.05,max_lim+0.05])
ax.legend()  

'''
plot tune vs report generalization - CCNorm
'''
max_lim =  1.0
min_lim = -0.2
xx = np.arange(-2,2,0.1) 

fig, ax =plt.subplots(figsize=(9, 9)) 
for j in range(mse_data.shape[0]):  
    ax.scatter(ccnorm_tune_data_mean[j], ccnorm_report_data_mean[j], marker='o', c = cmap(mse_py_color_array_mean[j]))
for k in range(len(labels)):
    ax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
    
ax.plot(xx,xx)
ax.set_xlabel('Tuning Set', fontsize = 20)
ax.set_ylabel('Reporting Set', fontsize = 20)
ax.set_ylim([min_lim-0.05,max_lim+0.05])
ax.set_xlim([min_lim-0.05,max_lim+0.05])
ax.legend()  

#######
'''
Reporting set RAINBOW PLOT
'''
cmap_grey = plt.cm.get_cmap('gist_rainbow')
fig3 =plt.figure(figsize=(9, 9))   
ax3 = fig3.add_subplot(1,1,1) 

for j, neuron in enumerate(trained_neurons):
    for k in range(mse_data.shape[1]):
        ax3.scatter(j, mse_data[neuron,k], marker='o', c = cmap((int(k))/(len(labels)+1)))
for j, neuron in enumerate(trained_neurons):
    for k in range(mse_data.shape[1]):
        if mse_data[neuron,k] == mse_report_data_mean[neuron]:
            ax3.scatter(j, mse_data[neuron,k], marker='o',s=100 , c = 'k')
for k in range(len(labels)):
    ax3.scatter(np.nan, np.nan, marker='o' , c = cmap(int(k)/(len(labels)+1)) , label = 'gamma %s' % labels[k])
ax3.set_xlabel('Neuron', fontsize = 30)
ax3.set_ylabel('CC Norm', fontsize = 30)
#ax3.set_title('Reporting Set CC Norm For All Models', fontsize=30)
ax3.legend()

























####
'''
old code
'''
############ PLots  
'''
NRF vs ic2ac LN
'''
# max_lim = 0
# relevant_data = np.array([])
# for j in range(mse_data.shape[0]):
#     if clusters_data[j].py_trained:
#         if not math.isnan(LN_data[j]) and not math.isnan(mse_report_data[j]):
#             max_lim = np.nanmax([max_lim,LN_data[j], mse_report_data[j]])
#             relevant_data = np.append(relevant_data, LN_data[j])
# min_lim = 0
# for j in range(mse_data.shape[0]):
#     if clusters_data[j].py_trained:
#         if not math.isnan(LN_data_coch[j]) and not math.isnan(mse_report_data[j]):
#             min_lim = np.nanmin([min_lim,LN_data[j], mse_report_data[j]])
            
# fig =plt.figure(figsize=(9, 9)) 
# xx = np.arange(-2,2,0.1)  
# ax = fig.add_subplot(1,1,1) 
# for j in range(mse_data.shape[0]):
#     if not math.isnan(mse_report_data[j]):
#         ax.scatter(LN_data[j], mse_report_data[j], marker='o', c = cmap(mse_py_color_array[j]))
# for k in range(len(labels)):
#     ax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
# ax.axvline(x=np.nanmedian(relevant_data), c='r', linewidth=0.5)
# ax.axhline(y=np.nanmedian(mse_report_data), c='r', linewidth=0.5)
# ax.scatter(np.nanmedian(relevant_data), np.nanmedian(mse_report_data), marker = 'o', s = 100, c='r')
# ax.plot(xx,xx)
# ax.set_xlabel('Matlab LN CCnorm', fontsize = 20)
# ax.set_ylabel('Model CCnorm', fontsize = 20)
# ax.set_ylim([min_lim-0.05,max_lim+0.05])
# ax.set_xlim([min_lim-0.05,max_lim+0.05])
# ax.set_title('ic2ac model vs Matlab ic2ac LN', fontsize=20)
# ax.legend()

# '''
# NRF vs ic2ac NRF
# '''
# max_lim = 0
# relevant_data = np.array([])
# for j in range(mse_data.shape[0]):
#     if clusters_data[j].py_trained:
#         if not math.isnan(LN_data_coch[j]) and not math.isnan(mse_report_data[j]):
#             max_lim = np.nanmax([max_lim,mat_data[j], mse_report_data[j]])
#             relevant_data = np.append(relevant_data, mat_data[j])
# min_lim = 0
# for j in range(mse_data.shape[0]):
#     if clusters_data[j].py_trained:
#         if not math.isnan(LN_data_coch[j]) and not math.isnan(mse_report_data[j]):
#             min_lim = np.nanmin([min_lim,mat_data[j], mse_report_data[j]])
            
# fig =plt.figure(figsize=(9, 9)) 
# xx = np.arange(-2,2,0.1)  
# ax = fig.add_subplot(1,1,1) 
# for j in range(mse_data.shape[0]):
#     if not math.isnan(mse_report_data[j]):
#         ax.scatter(mat_data[j], mse_report_data[j], marker='o', c = cmap(mse_py_color_array[j]))
# for k in range(len(labels)):
#     ax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
# ax.plot(xx,xx)
# ax.axvline(x=np.nanmedian(relevant_data), c='r', linewidth=0.5)
# ax.axhline(y=np.nanmedian(mse_report_data), c='r', linewidth=0.5)
# ax.scatter(np.nanmedian(relevant_data), np.nanmedian(mse_report_data), marker = 'o', s = 100, c='r')
# ax.set_xlabel('matlab nrf ccnorm', fontsize = 20)
# ax.set_ylabel('python nrf ccnorm', fontsize = 20)
# ax.set_ylim([min_lim-0.05,max_lim+0.05])
# ax.set_xlim([min_lim-0.05,max_lim+0.05])
# ax.set_title('ic2ac model vs Matlab ic2ac NRF', fontsize=20)
# ax.legend()

# '''
# model vs coch2ac LN
# '''
# max_lim = 0
# relevant_data = np.array([])
# for j in range(mse_data.shape[0]):
#     if clusters_data[j].py_trained:
#         if not math.isnan(LN_data_coch[j]) and not math.isnan(mse_report_data[j]):
#             max_lim = np.nanmax([max_lim,LN_data_coch[j], mse_report_data[j]])
#             relevant_data = np.append(relevant_data, LN_data_coch[j])
# min_lim = 0
# for j in range(mse_data.shape[0]):
#     if clusters_data[j].py_trained:
#         if not math.isnan(LN_data_coch[j]) and not math.isnan(mse_report_data[j]):
#             min_lim = np.nanmin([min_lim,LN_data_coch[j], mse_report_data[j]])
            
# fig =plt.figure(figsize=(9, 9)) 
# xx = np.arange(-2,2,0.1)  
# ax = fig.add_subplot(1,1,1) 
# for j in range(mse_data.shape[0]):
#     if not math.isnan(mse_report_data[j]):
#         ax.scatter(LN_data_coch[j], mse_report_data[j], marker='o', c = cmap(mse_py_color_array[j]))
# for k in range(len(labels)):
#     ax.scatter(np.nan, np.nan, marker= 'o', c = cmap(z[k]), label = labels[k])
# ax.plot(xx,xx)
# ax.axvline(x=np.nanmedian(relevant_data), c='r', linewidth=0.5)
# ax.axhline(y=np.nanmedian(mse_report_data), c='r', linewidth=0.5)
# ax.scatter(np.nanmedian(relevant_data), np.nanmedian(mse_report_data), marker = 'o', s = 100, c='r')
# ax.set_xlabel('cochleagram ccnorm', fontsize = 20)
# ax.set_ylabel('model ccnorm', fontsize = 20)
# ax.set_ylim([min_lim-0.05,max_lim+0.05])
# ax.set_xlim([min_lim-0.05,max_lim+0.05])
# ax.set_title('ic2ac model vs LN cochleagram', fontsize=20)
# ax.legend()