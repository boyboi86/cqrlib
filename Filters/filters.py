# -*- coding: utf-8 -*-
"""
Cumsum filters for mean reversion
 based on Marco Lopez AFML and filtering strategy research by Fama and Blume[1966]
"""

import numpy as np
import pandas as pd

'''

This is the original cumsum event in AFML pg 39 snippet 2.4
def cumsum_events(df: pd.Series, limit: float):
    idx, _up, _dn = [], 0, 0
    diff = df.diff()
    for i in diff.index[1:]:
        _up, _dn = max(0, _up + diff.loc[i]), min(0, _dn + diff.loc[i])
        if _up > limit:
            _up = 0; idx.append(i)
        elif _dn < - limit:
            _dn = 0; idx.append(i)
        
    return pd.DatetimeIndex(idx)
'''
def cumsum_events(data: pd.Series, limit: float):
    '''
    params: pd.Series => time series input
    params: pd.Series => threshold before logging datetime index
    
    This is not the original cumsum event in AFML pg 39 snippet 2.4
    Most part of the code has been rewritten in numpy instead
    
    Logic is the same, but kindly go through the code before using it.
    
    Currently using the below:
    
    #numba 0.49.1
    #numpy 1.17.3
    #pandas 1.0.3
    '''
    if data.isnull().values.any():
        raise ValueError('Data contain NaNs, kindly review data input!')
    idx, _up, _dn, count = [], 0, 0, 0
    diff = data.diff().dropna().to_numpy()
    datetimeidx = data.index[1:].to_numpy()
    df0 = list(zip(datetimeidx, diff))
    for i in np.arange(len(diff)):
        _up, _dn = max(0, _up + df0[i][1]), min(0, _dn + df0[i][1])
        p("up", _up, "_dn", _dn)
        if np.greater(_up, limit).bool() == True:
            _up = 0
            idx.append((df0)[i][0])
        elif np.less(_dn, -limit).bool() == True:
            _dn = 0 
            idx.append((df0)[i][0])
    return pd.DatetimeIndex(idx)


def cumsum_filter(df: pd.DataFrame, limit: float):
    if 'Close' in df.columns:
        column = df['Close']
    df0 = cumsum_events(column, limit)
    return df.reindex(df0)