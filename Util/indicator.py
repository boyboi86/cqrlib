# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:41:46 2020

@author: Wei_X
"""

import pandas as pd

def bband(data: pd.Series, span0: int = 50, width: int = 1):
    '''
    Basic bollinger band
    
    params: DataFrame => entire dataframe which close price resides
    params: int => span0 is how many to roll
    params: int => width to set 
    '''
    
    ma = data['close'].ewm(span0).mean()
    std = data['close'].ewm(span0).std()
    
    upper = ma + (std * width)
    lower = ma + (std * width)
    
    return ma, upper, lower