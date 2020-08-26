# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:12:54 2020

@author: Tom
"""
import numpy as np
import scipy.io

class Myobject(object):  #Code to initialize objects
    pass
       

class stimulus_unpacker(object):
    def __init__(self, x):
        self.freqs = x['freqs'][()][:]     
        self.X_ft = x['X_ft'][()][:,:]
        #self.all_valid_data = x['all_valid_data'][()] #need the [()] at end to access subfields or structs
        #self.all_valid_data = set_all_valid_data(x['all_valid_data'][()])
        self.all_valid_data = Myobject()
        self.all_valid_data.valid_idxes = x['all_valid_data'][()]['valid_idxes'][()]
        self.all_valid_data.ten_folds = []
        for i in range(x['all_valid_data'][()]['ten_folds'][()].size):
            self.all_valid_data.ten_folds.append(Myobject())
            self.all_valid_data.ten_folds[i].fit_idx =  x['all_valid_data'][()]['ten_folds'][()][i]['fit_idx'][()]
            self.all_valid_data.ten_folds[i].pred_idx =  x['all_valid_data'][()]['ten_folds'][()][i]['pred_idx'][()]
        self.all_valid_data.no_folds = Myobject()
        self.all_valid_data.no_folds.fit_idx = x['all_valid_data'][()]['no_folds'][()]['fit_idx'][()]
        self.all_valid_data.no_folds.pred_idx = x['all_valid_data'][()]['no_folds'][()]['pred_idx'][()]
        
class cluster_unpacker(object):
    def __init__(self, number, x):
        self.number = number
        self.brainAreaIdx = x[number]['brainAreaIdx']
        self.brainArea = x[number]['brainArea']
        self.y_t = x[number]['y_t'][()]
        self.y_dt = x[number]['y_dt'][()]
        self.unique_penetration_idx = x[number]['unique_penetration_idx']
        
        self.mat = Myobject()             #Matlab NRF model data
        self.mat.ic = Myobject()          #Note -  we are just saving the cc_norm ect data from the reporting set,
        self.mat.mgb = Myobject()       #        though we do save all of yhat.
        self.mat.ac = Myobject()

        self.mat.ic.cc_norm = x[number]['all_data_a2a_nrf'][()]['ic_nrf'][()]['cc_norm_pred']
        self.mat.ic.cc_abs = x[number]['all_data_a2a_nrf'][()]['ic_nrf'][()]['cc_abs_pred']
        self.mat.ic.cc_max = x[number]['all_data_a2a_nrf'][()]['ic_nrf'][()]['cc_max_pred']
        self.mat.ic.yhat = x[number]['all_data_a2a_nrf'][()]['ic_nrf'][()]['y_hat']
        
        self.mat.mgb.cc_norm = x[number]['all_data_a2a_nrf'][()]['mgb_nrf'][()]['cc_norm_pred']
        self.mat.mgb.cc_abs = x[number]['all_data_a2a_nrf'][()]['mgb_nrf'][()]['cc_abs_pred']
        self.mat.mgb.cc_max = x[number]['all_data_a2a_nrf'][()]['mgb_nrf'][()]['cc_max_pred']
        self.mat.mgb.yhat = x[number]['all_data_a2a_nrf'][()]['mgb_nrf'][()]['y_hat']
        
        self.mat.ac.cc_norm = x[number]['all_data_a2a_nrf'][()]['ac_nrf'][()]['cc_norm_pred']
        self.mat.ac.cc_abs = x[number]['all_data_a2a_nrf'][()]['ac_nrf'][()]['cc_abs_pred']
        self.mat.ac.cc_max = x[number]['all_data_a2a_nrf'][()]['ac_nrf'][()]['cc_max_pred']
        self.mat.ac.yhat = x[number]['all_data_a2a_nrf'][()]['ac_nrf'][()]['y_hat']
        
        self.py = Myobject()              #Python NRF model data
        self.py.coch = []                 #Each element of list will be an object holding data for one particular hyperparameter value
        self.py.ic = []
        self.py.mgb = []
        self.py.ac = []
        
    def initialize_gamma(self, gamma_values= None, input_area = None):
        '''
        Creates a list of attributes ready to store model data for each gamma value.
        '''       
        if input_area.lower() == 'coch':
            for gamma_idx in range(len(gamma_values)):
                self.py.coch.append(Myobject())
        if input_area.lower() == 'ic':
            for gamma_idx in range(len(gamma_values)):
                self.py.ic.append(Myobject())
        if input_area.lower() == 'mgb':
            for gamma_idx in range(len(gamma_values)):
                self.py.mgb.append(Myobject())
        if input_area.lower() == 'ac':
            for gamma_idx in range(len(gamma_values)):
                self.py.ac.append(Myobject())
            
        
    def add_py_data(self, gamma_idx = None, input_area = None, dropout=None, gamma_value = None , loss_history = None,
                    ccnorm_history = None, tune_ccnorm_history = None,  mse_history = None, tune_mse_history = None,
                    cc_max = None, ccnorm_epoch = None , ccnorm_cc_abs = None , ccnorm_yhat = None, ccnorm_model = None,
                    mse_cc_abs = None, mse_yhat=None, mse_epoch = None, mse_model = None, p=None):
        if input_area.lower() == 'coch':
            self.py.coch[gamma_idx].dropout = dropout
            if dropout:
                self.py.coch[gamma_idx].p = p            
            self.py.coch[gamma_idx].gamma_value = gamma_value
            self.py.coch[gamma_idx].ccnorm_model = Myobject()
            self.py.coch[gamma_idx].mse_model = Myobject()
            self.py.coch[gamma_idx].loss_h = loss_history
            self.py.coch[gamma_idx].tune_ccnorm_h = tune_ccnorm_history
            self.py.coch[gamma_idx].ccnorm_h = ccnorm_history
            self.py.coch[gamma_idx].tune_mse_h = tune_mse_history
            self.py.coch[gamma_idx].mse_h = mse_history

            self.py.coch[gamma_idx].ccnorm_model.ccnorm = ccnorm_history[ccnorm_epoch]
            self.py.coch[gamma_idx].ccnorm_model.ccabs = ccnorm_cc_abs
            self.py.coch[gamma_idx].ccnorm_model.ccmax = cc_max
            self.py.coch[gamma_idx].ccnorm_model.yhat = ccnorm_yhat
            self.py.coch[gamma_idx].ccnorm_model.model = ccnorm_model
            self.py.coch[gamma_idx].ccnorm_model.best_epoch = ccnorm_epoch
            
            self.py.coch[gamma_idx].mse_model.ccnorm = ccnorm_history[mse_epoch]
            self.py.coch[gamma_idx].mse_model.ccabs = mse_cc_abs
            self.py.coch[gamma_idx].mse_model.ccmax = cc_max
            self.py.coch[gamma_idx].mse_model.yhat = mse_yhat
            self.py.coch[gamma_idx].mse_model.model = mse_model
            self.py.coch[gamma_idx].mse_model.best_epoch = mse_epoch
            
            
        if input_area.lower() == 'ic':
            self.py.ic[gamma_idx].dropout = dropout
            if dropout:
                self.py.ic[gamma_idx].p = p            
            self.py.ic[gamma_idx].gamma_value = gamma_value
            self.py.ic[gamma_idx].ccnorm_model = Myobject()
            self.py.ic[gamma_idx].mse_model = Myobject()
            self.py.ic[gamma_idx].loss_h = loss_history
            self.py.ic[gamma_idx].tune_ccnorm_h = tune_ccnorm_history
            self.py.ic[gamma_idx].ccnorm_h = ccnorm_history
            self.py.ic[gamma_idx].tune_mse_h = tune_mse_history
            self.py.ic[gamma_idx].mse_h = mse_history

            self.py.ic[gamma_idx].ccnorm_model.ccnorm = ccnorm_history[ccnorm_epoch]
            self.py.ic[gamma_idx].ccnorm_model.ccabs = ccnorm_cc_abs
            self.py.ic[gamma_idx].ccnorm_model.ccmax = cc_max
            self.py.ic[gamma_idx].ccnorm_model.yhat = ccnorm_yhat
            self.py.ic[gamma_idx].ccnorm_model.model = ccnorm_model
            self.py.ic[gamma_idx].ccnorm_model.best_epoch = ccnorm_epoch
            
            self.py.ic[gamma_idx].mse_model.ccnorm = ccnorm_history[mse_epoch]
            self.py.ic[gamma_idx].mse_model.ccabs = mse_cc_abs
            self.py.ic[gamma_idx].mse_model.ccmax = cc_max
            self.py.ic[gamma_idx].mse_model.yhat = mse_yhat
            self.py.ic[gamma_idx].mse_model.model = mse_model
            self.py.ic[gamma_idx].mse_model.best_epoch = mse_epoch

        if input_area.lower() == 'mgb':
            self.py.mgb[gamma_idx].dropout = dropout
            if dropout:
                self.py.mgb[gamma_idx].p = p
            self.py.mgb[gamma_idx].gamma_value = gamma_value
            self.py.mgb[gamma_idx].ccnorm_model = Myobject()
            self.py.mgb[gamma_idx].mse_model = Myobject()
            self.py.mgb[gamma_idx].loss_h = loss_history
            self.py.mgb[gamma_idx].tune_ccnorm_h = tune_ccnorm_history
            self.py.mgb[gamma_idx].ccnorm_h = ccnorm_history
            self.py.mgb[gamma_idx].tune_mse_h = tune_mse_history
            self.py.mgb[gamma_idx].mse_h = mse_history

            self.py.mgb[gamma_idx].ccnorm_model.ccnorm = ccnorm_history[ccnorm_epoch]
            self.py.mgb[gamma_idx].ccnorm_model.ccabs = ccnorm_cc_abs
            self.py.mgb[gamma_idx].ccnorm_model.ccmax = cc_max
            self.py.mgb[gamma_idx].ccnorm_model.yhat = ccnorm_yhat
            self.py.mgb[gamma_idx].ccnorm_model.model = ccnorm_model
            self.py.mgb[gamma_idx].ccnorm_model.best_epoch = ccnorm_epoch
            
            self.py.mgb[gamma_idx].mse_model.ccnorm = ccnorm_history[mse_epoch]
            self.py.mgb[gamma_idx].mse_model.ccabs = mse_cc_abs
            self.py.mgb[gamma_idx].mse_model.ccmax = cc_max
            self.py.mgb[gamma_idx].mse_model.yhat = mse_yhat
            self.py.mgb[gamma_idx].mse_model.model = mse_model
            self.py.mgb[gamma_idx].mse_model.best_epoch = mse_epoch

        if input_area.lower() == 'ac':
            self.py.ac[gamma_idx].dropout = dropout
            if dropout:
                self.py.ac[gamma_idx].p = p
            self.py.ac[gamma_idx].gamma_value = gamma_value
            self.py.ac[gamma_idx].ccnorm_model = Myobject()
            self.py.ac[gamma_idx].mse_model = Myobject()
            self.py.ac[gamma_idx].loss_h = loss_history
            self.py.ac[gamma_idx].tune_ccnorm_h = tune_ccnorm_history
            self.py.ac[gamma_idx].ccnorm_h = ccnorm_history
            self.py.ac[gamma_idx].tune_mse_h = tune_mse_history
            self.py.ac[gamma_idx].mse_h = mse_history

            self.py.ac[gamma_idx].ccnorm_model.ccnorm = ccnorm_history[ccnorm_epoch]
            self.py.ac[gamma_idx].ccnorm_model.ccabs = ccnorm_cc_abs
            self.py.ac[gamma_idx].ccnorm_model.ccmax = cc_max
            self.py.ac[gamma_idx].ccnorm_model.yhat = ccnorm_yhat
            self.py.ac[gamma_idx].ccnorm_model.model = ccnorm_model
            self.py.ac[gamma_idx].ccnorm_model.best_epoch = ccnorm_epoch
            
            self.py.ac[gamma_idx].mse_model.ccnorm = ccnorm_history[mse_epoch]
            self.py.ac[gamma_idx].mse_model.ccabs = mse_cc_abs
            self.py.ac[gamma_idx].mse_model.ccmax = cc_max
            self.py.ac[gamma_idx].mse_model.yhat = mse_yhat
            self.py.ac[gamma_idx].mse_model.model = mse_model
            self.py.ac[gamma_idx].mse_model.best_epoch = mse_epoch

            
        
