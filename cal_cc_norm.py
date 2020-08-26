# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:03:16 2020

@author: Tom
"""
import numpy as np

def calculate(R, yhat):
    N, T = R.shape[0], R.shape[1]
    y = np.mean(R,axis=0)
    Ey = np.mean(y)
    Eyhat = np.mean(yhat)
    Vy = np.sum(np.multiply((y-Ey),(y-Ey)))/T
    Vyhat = np.sum(np.multiply((yhat-Eyhat),(yhat-Eyhat)))/T
    Cyyhat = np.sum(np.multiply((y-Ey),(yhat-Eyhat)))/T
    SP = (np.var(np.sum(R,axis=0), ddof=1)-np.sum(np.var(R,axis=1,ddof=1)))/(N*(N-1))
    
    CCabs =Cyyhat/np.sqrt(Vy*Vyhat)
       
    if SP<=0:
        #print('SP less than or equal to zero - CCmax and CCnorm cannot be calculated.')
        CCnorm = np.nan
        CCmax = 0 
    else:
        CCnorm = Cyyhat/np.sqrt(SP*Vyhat)
        CCmax = np.sqrt(SP/Vy)
    return CCabs, CCnorm, CCmax
