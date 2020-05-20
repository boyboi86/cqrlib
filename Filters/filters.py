import numpy as np
import pandas as pd
#from numba import njit 

def sym_csf(data: pd.Series, limit: float):
    '''
    params: pd.Series => time series input price only accepts ndarray, list, pd.series
    params: pd.Series => threshold before logging datetime index
    
    AFML pg 39 snippet 2.4
    This func will give absolute return based on input price.
    As for limit, use original price series to derive standard deviation as an estimate.
    This is to ensure stationary series is homoscedastic.
    
    Logic is the same, but kindly go through the code before using it.
    
    WARNING!!: DO NOT EVER CHANGE data.diff() into data.apply(np.log).diff(), use only absolute price.
    
    np.log(data) does allow memory preservation but! that is not what we are looking for.
    
    In addition, np.log(data) contains additive properties. This will distort your data structure.
    
    The main use use for this func is to ensure your data structure is homoscedastic as much as possible, which is pivotal for mean-reversion strategy.
    IF you change this line, 99.99% you will only get a Heteroscedastic data structure, no matter what you do at a later stage.
    
    This will haunt you at the later stage as you develop your mean-reversion strategy.
    
    The above claim is tested and proven (sort of..)
    
    This filter will ensure your data structure maintain a good data structural shape , 
    which is key to mean-reversion strategy and to ensure your data structure is NOT too random.
    
    If you are not sure  what I mean pls go read up on "time-series stationarity" and run a white test using both log price and abs price.
    
    REPEAT 10 times: "you will not change this func"
    
    This filter return datatimeindex only.
    '''
    if isinstance(data, (str, float, int)):
        raise ValueError('Data must be numpy ndarray or pandas series!')
    elif data.isnull().values.any():
        raise ValueError('Data contain NaNs, kindly review data input!')

    if isinstance(limit, (list, np.ndarray, pd.Series)):
        limit = float(limit.mean())
        UserWarning('Func does not accept numpy array, convert limit to mean value as estimate')
    elif isinstance(limit, (int, float)):
        limit = float(limit)
        UserWarning('Convert limit to float value as estimate')
    else:
        raise ValueError('Limit is neither numpy ndarray, pandas series nor float!')
    
    idx, _up, _dn = [], 0, 0
    diff = data.diff()
    for i in diff.index[1:]:
        _up, _dn = max(0, float(_up + diff.loc[i])), min(0, float(_dn + diff.loc[i]))
        if _up >= limit:
            _up = 0; idx.append(i)
        elif _dn <= - limit:
            _dn = 0; idx.append(i)
        
    return pd.DatetimeIndex(idx)

'''
@njit
def _cumsum_events(df0, _up, _dn, limit, count, idx):
    for i in np.arange(count):
        _up, _dn = max(0, _up + df0[i][1]), min(0, _dn + df0[i][1])
        p("up", _up, "_dn", _dn)
        if np.greater(_up, limit).bool() == True:
            _up = 0
            idx.append((df0)[i][0])
        elif np.less(_dn, -limit).bool() == True:
            _dn = 0 
            idx.append((df0)[i][0])
    return _up, _dn, idx

def cumsum_events(data: pd.Series, limit: float):

    params: pd.Series => time series input price only accepts ndarray, list, pd.series
    params: pd.Series => threshold before logging datetime index
    
    This is not the original cumsum event in AFML pg 39 snippet 2.4
    Most part of the code has been rewritten in numpy instead.
    
    This func will give absolute return based on input price.
    As for limit, use original price series to derive standard deviation as an estimate.
    This is to ensure stationary series is homoscedastic.
    
    Logic is the same, but kindly go through the code before using it.
    
    There is no imbalance in this algo, but there is a floor and ceiling which is set at zero.
    
    This filter return datatimeindex only.
    
    Currently using the below:
    
    #numba 0.49.1
    #numpy 1.17.3
    #pandas 1.0.3

    if isinstance(data, (str, float, int)):
        raise ValueError('Data must be numpy ndarray or pandas series!')
    elif data.isnull().values.any():
        raise ValueError('Data contain NaNs, kindly review data input!')

    if isinstance(limit, (list, np.ndarray, pd.core.series.Series)):
        limit = float(limit.mean())
        UserWarning('Func does not accept numpy array, convert limit to mean value as estimate')
    elif isinstance(limit, (int, float)):
        limit = float(limit)
        UserWarning('Convert limit to float value as estimate')
    else:
        raise ValueError('Limit is neither numpy ndarray, pandas series nor float!')
    
    idx, _up, _dn = [], 0, 0
    diff = data.diff().dropna().to_numpy()
    datetimeidx = data.index[1:].to_numpy()
    df0 = list(zip(datetimeidx, diff))
    for i in np.arange(len(diff)):
        _up, _dn = max(0, _up + df0[i][1]), min(0, _dn + df0[i][1])
        if np.greater(_up, limit):
            _up = 0
            idx.append((df0)[i][0])
        elif np.less(_dn, -limit):
            _dn = 0 
            idx.append((df0)[i][0])
    return pd.DatetimeIndex(idx)
'''