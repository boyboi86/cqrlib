# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:41:46 2020

@author: Wei_X
"""
import numpy as np
import pandas as pd

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
    
    