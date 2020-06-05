# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:41:46 2020

@author: Wei_X
"""
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification

p = print

def _bband(data: pd.DataFrame, window: int = 50, width: float = 0.001):
    '''
    Basic bollinger band
    
    params: DataFrame => entire dataframe which close price resides
    params: int => span0 is how many to roll
    params: int => width to set 
    '''
    avg = data['close'].ewm(span = window).mean()
    std = avg * width
    upper = avg + std
    lower = avg - std
    return avg, upper, lower

def _side_pick(data: pd.DataFrame):
    for i in np.arange(data.index.shape[0]):
        if (data['close'].iloc[i] >= data['upper'].iloc[i]):
            data['side'].iat[i] = -1
        elif (data['close'].iloc[i] <= data['lower'].iloc[i]):
            data['side'].iat[i] = 1
    return data

def bband_as_side(data: pd.DataFrame, window: int = 100, width: int = 0.001):
    data['avg'], data['upper'], data['lower'] = _bband(data = data, window = window, width = width)
    data['side'] = np.nan
    data = _side_pick(data = data)
    
    upper = data[data['upper'] <= data['close']] # short signal
    lower = data[data['lower'] >= data['close']] # long signal
    
    p("Bollinger Band results:\n")
    p("Num of times upper limit touched: {0}\nNum of times lower limit touched: {1}"
      .format(upper.count()[0], 
              lower.count()[0]))
    
    return data.dropna()
    
def make_data(n_features=40, n_informative=10, n_redundant=10, n_samples=10000):
    # generate a random dataset for a classification problem    
    trnsX, cont = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, random_state=0, shuffle=False)
    df0 = pd.date_range(periods=n_samples, freq=pd.tseries.offsets.BDay(), end=pd.datetime.today())
    trnsX = pd.DataFrame(trnsX, index=df0)
    cont = pd.Series(cont, index=df0).to_frame('bin')
    df0 = ['I_%s' % i for i in range(n_informative)] + ['R_%s' % i for i in range(n_redundant)]
    df0 += ['N_%s' % i for i in range(n_features - len(df0))]
    trnsX.columns = df0
    cont['w'] = 1.0 / cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index)
    return trnsX, cont