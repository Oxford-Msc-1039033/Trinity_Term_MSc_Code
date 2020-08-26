# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:51:51 2020

@author: Tom


Code to create X_fht for ac2ic nrf mapping with n_h=10, n_fut = 5
"""
import numpy as np

import sys
sys.path.insert(0,'C:\\Users\\Tom\\OneDrive\\Trinity_term_project\\Python-Code\\my-code\\hierarchy-py')

import numpy as np
from matplotlib import pyplot as plt
from tomlib.tom_nrf import *
from hierarchy import Clusters, Stimulus
from tomlib.nrf_utils import select_xVal_idxes
from benlib.strf import select_idxes
import benlib.torchnrf as bennrf

n_hist = 10
n_fut = 5

stim = Stimulus()
clusters_ic = Clusters('ic', stim.segment_lengths)
clusters_mgb = Clusters('mgb', stim.segment_lengths)
clusters_ac = Clusters('ac', stim.segment_lengths)

fit_idx, report_idx = stim.no_folds

ac_data = clusters_ac.get_all_y_t()
X_fit = select_idxes(ac_data, fit_idx)
X_report  = select_idxes(ac_data, report_idx)

X_tfh_fit = tensorize_segments(X_fit, n_hist, n_fut=n_fut)
n_t, n_f, n_h = X_tfh_fit.shape
X_tfh_fit = X_tfh_fit.reshape(n_t, n_f*n_h)
X_tfh_train = X_tfh_fit[:int(0.9*len(fit_idx)),:]
X_tfh_tune = X_tfh_fit[int(0.9*len(fit_idx)):,:]

X_tfh_report = tensorize_segments(X_report, n_hist, n_fut=n_fut)
n_t, n_f, n_h = X_tfh_report.shape
X_tfh_report = X_tfh_report.reshape(n_t, n_f*n_h)

train_idx = fit_idx[:int(0.9*len(fit_idx))]
tune_idx = fit_idx[int(0.9*len(fit_idx)):]

np.savetxt('ac2ic_train_idx.csv', train_idx, delimiter=',')   #save the indxs trained on.
np.savetxt('ac2ic_tune_idx.csv', tune_idx, delimiter=',')   #save the indxs trained on.
np.savetxt('ac2ic_report_idx.csv', report_idx, delimiter=',')   #save the indxs trained on.
np.savetxt('ac2ic_X_tfh_train.csv', X_tfh_train, delimiter=',')   #save the indxs trained on.
np.savetxt('ac2ic_X_tfh_tune.csv', X_tfh_tune, delimiter=',')   #save the indxs trained on.
np.savetxt('ac2ic_X_tfh_report.csv', X_tfh_report, delimiter=',')   #save the indxs trained on.

