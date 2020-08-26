# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 19:53:26 2020

@author: thble
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import DivergingNorm
import pickle

def low_pass_filter(data, n_h = 10):
    '''
    applies a smoothing operator to an input array
    '''
    
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

def relevant_info(LN_data, mse_report_data_mean, clusters_data):
    max_lim = 0
    relevant_data = np.array([])
    for j in range(len(clusters_data)):
        if clusters_data[j].py_trained:
            if not math.isnan(LN_data[j]) and not math.isnan(mse_report_data_mean[j]):
                max_lim = np.nanmax([max_lim,LN_data[j], mse_report_data_mean[j]])
                relevant_data = np.append(relevant_data, LN_data[j])
    min_lim = 0
    for j in range(len(clusters_data)):
        if clusters_data[j].py_trained:
            if not math.isnan(LN_data[j]) and not math.isnan(mse_report_data_mean[j]):
                min_lim = np.nanmin([min_lim,LN_data[j], mse_report_data_mean[j]])
    return min_lim, max_lim, relevant_data


def show_strf(strf, bf_hz, neuron='NA', gamma='NA', epochs='NA',  area ='NA', input_area='NA', cc_norm='NA', save_fig = False):
    my_dpi = 120
    mx = max([np.max(strf), np.abs(np.min(strf))])
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size='5%', pad=0.2)
    im = ax.imshow(strf, cmap = 'seismic', vmin=-mx, vmax=mx, origin='lower')
    ax.set_yticks([0, int(len(bf_hz)/3),int(2*len(bf_hz)/3), len(bf_hz)-1])
    ax.set_yticklabels([1000, int(bf_hz[int(len(bf_hz)/3)]),int(bf_hz[int(2*len(bf_hz)/3)]) , 64000], rotation=90)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xticks([])
    for label in ax.yaxis.get_ticklabels():
        label.set_verticalalignment('center')
    ax.set_ylabel('Best Frequency', fontsize =6)
    ax.set_xlabel('Time/Hidden Unit', fontsize =6)
    # fig.colorbar(im, cax=cax, orientation = 'vertical')
    cbar = plt.colorbar(im, cax=cax, orientation = 'horizontal', pad = 0.15)
    cbar.set_label('Weight strength', fontsize = 6)
    cbar.ax.tick_params(labelsize=8) 
    xcoords = np.arange(14.5,150,15)
    for xc in xcoords:
        ax.axvline(x=xc, c='k', linewidth=0.1)
    ax.set_title('Neuron: {} \n CC Norm: {:.4f}'.format(neuron, cc_norm), fontsize=6)
    plt.show()
    if save_fig:
        plt.savefig('RFs/rf_{}2{}_neuron{}.png'.format(input_area, area, neuron), 
                    dpi=my_dpi*20, bbox_inches='tight')
        
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
    
def show_stimuli(stimuli = None, x_range = None, short = False):
    fig, ax =plt.subplots() 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    if short:
        im = ax.imshow(stimuli.X_ft[:,x_range[0]:x_range[1]], cmap = 'jet', origin='lower')
    elif not short:
        im = ax.imshow(stimuli.X_ft, cmap = 'jet', origin='lower')
    ax.set_xticks(np.arange(0,100,20))
    ax.set_yticks(np.arange(0,25,12))
    ax.set_xticklabels(25*np.arange(0,100,20)) 
    ax.set_yticklabels([1000, 8000, 64000]) 
    ax.set_xlabel('Time (ms)', fontsize=30)
    ax.set_ylabel('Frequency (kHz)', fontsize=30)
    cbar = plt.colorbar(im, cax=cax, orientation = 'vertical')
    cbar.set_label('Level (dB SPL)', fontsize=25)
    return fig
    
def load_bf_data(input_area=None):
    with open('data/res-bfs.pkl', 'rb') as f:
        bf_data = pickle.load(f)
    
    bf_idx = []
    bf_hz = []
    for x in bf_data:
        if x['unique_id'][0:2] == input_area:
            bf_idx.append(x['coch_kernels']['SeparableKernel']['bf_idx'])
            bf_hz.append(x['coch_kernels']['SeparableKernel']['bf_hz'])
    
    return bf_idx, bf_hz, bf_data