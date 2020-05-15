import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from numba import njit
from statsmodels.tsa.stattools import adfuller

p = print

@njit
def getWeights(d: float, size: int):
    w=[1.]
    for k in np.arange(1,size):
        w_ = -w[-1]/k*(d-k+1)
        w.append(w_)
    w=np.array(w[::-1]).reshape(-1,1)
    return w

def fracDiff(data: pd.Series, d: float, thres: float = 1e-2):
    
    if isinstance(data, (str, float, int, dict, tuple)):
        raise ValueError('Data must be numpy ndarray or pandas data!')
    elif data.isnull().values.any():
        raise ValueError('Data contain NaNs, kindly review data input!')
    else:
        data = data.to_frame()
        
    if isinstance(d, (list, str, dict, tuple, pd.Series, np.ndarray, pd.DataFrame)):
        raise ValueError('d val must be positive float value within range(0,2) i.e. 0.05')
    elif d <= 0 or d >= 2:
        warnings.warn('d val must be positive float value, otherwise it will result in infinite observations i.e. 0.05')
    else:
        d = float(d)
        
    if isinstance(thres, (list, str, dict, tuple, pd.Series, np.ndarray, pd.DataFrame)):
        raise ValueError('thres val must be float, kindly run func min_value provided see minimal d value for stationary i.e. 0.1!')
    elif thres <= 1e-2:
        warnings.warn('thres val <= 1.e-2 may not suit non-trend series, may take up longer than expected to calculate val i.e. 1e-5')
    else:
        thres = float(thres)
     
    w=getWeights(d, data.shape[0])
    w_=np.cumsum(abs(w))
    w_/=w_[-1]
    skip = w_[w_>thres].shape[0]
    df={}
    for name in data.columns:
        seriesF,df_=data[[name]].fillna(method='ffill').dropna(),pd.Series(index=data.index, dtype=float)
        for iloc in range(skip,seriesF.shape[0]):
            loc=seriesF.index[iloc]
            if not np.isfinite(data.loc[loc,name]): continue
            df_[loc]=np.dot(w[-(iloc+1):,:].T,seriesF.loc[:loc])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df

@njit
def getWeights_FFD(d: float, thres: float):
    w,k=[1.],1
    while True:
        w_ = -w[-1]/k*(d-k+1)
        if abs(w_) < thres:
            break
        w.append(w_); k+=1
    return np.array(w[::-1]).reshape(-1,1)

# need to refactor to optimze if not will take forever if threshold too low
def fracDiff_FFD(data: pd.Series, d: float, thres: float = 1e-2):
    '''
    This method will control weights until an acceptable threhold before applying to series
    to counter exponential decrease as compared to fracDiff calculation.
    
    params: data => non-stationary close price series
    params: FFD => boolean type with 2 possible values only fracDiff_FFD (True) or fracDiff (False)
    params: d => minimal d value to pass ADF test
    params: threshold => threshold value to control weights before fitting
    '''
    
    if isinstance(data, (str, float, int, dict, tuple)):
        raise ValueError('Data must be numpy ndarray or pandas data!')
    elif data.isnull().values.any():
        raise ValueError('Data contain NaNs, kindly review data input!')
    else:
        data = data.to_frame()
        
    if isinstance(d, (list, str, dict, tuple, pd.Series, np.ndarray, pd.DataFrame)):
        raise ValueError('d val must be float, hypothesis test critical size i.e. 0.05')
    elif d <= 0 or d >= 2:
        warnings.warn('d val must be non-zero positive float within range(0, 2) i.e. 0.05')
    else:
        d = float(d)
        
    if isinstance(thres, (list, str, dict, tuple, pd.Series, np.ndarray, pd.DataFrame)):
        raise ValueError('thres val must be float, kindly run func min_value provided see minimal d value for stationary i.e. 0.1!')
    elif thres <= 1e-2:
        warnings.warn('thres val <= 1.e-2 may not suit non-trend series, may take up longer than expected to calculate val i.e. 1e-5')
    else:
        thres = float(thres)
        
    w, df = getWeights_FFD(d, thres), {}
    width = len(w)-1
    for name in data.columns:
        seriesF,df_=data[[name]].fillna(method='ffill').dropna(),pd.Series(index=data.index, dtype=float)
        for iloc in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc - width], seriesF.index[iloc]
            if not np.isfinite(data.loc[loc1,name]): continue
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df


def min_value(data: pd.Series, FFD = True, thres: float = 1e-2, pval_threshold: float = 0.05,
              num: int = 100):
    '''
    params: data => non-stationary close price series
    params: FFD => boolean type with 2 possible values only fracDiff_FFD (True) or fracDiff (False)
    params: thres => threshold value for d value weights
    params: pval_threshold => p value for ADF hypothesis test critical size 0.1, 0.05, 0.01
    params: num=> number of samples to check for ADF test, control grandularity of possible d val 
    '''
    func = fracDiff_FFD
    if FFD != True: func = fracDiff
    functype = ['Fixed Window FD', 'Expanding Window FD']
    
    if isinstance(data, (str, float, int, dict, tuple)):
        raise ValueError('Data must be numpy ndarray or pandas data!')
    elif data.isnull().values.any():
        raise ValueError('Data contain NaNs, kindly review data input!')
    else:
        data = data.to_frame()
        
    if isinstance(func, (str, float, int, dict, tuple, pd.Series, np.ndarray, pd.DataFrame)):
        raise ValueError('func input is function input choose either fracDiff_FFD or fracDiff provided!')
    elif FFD != True and FFD != False:
        raise ValueError('FFD must be boolean arguement, default will be True i.e. True or False')
    elif func != fracDiff_FFD or func != fracDiff:
        raise ValueError('func takes arg only fracDiff_FFD or fracDiff, default will be fracDiff_FFD')
    else:
        if func == fracDiff_FFD: p('Function used: {0}'.format(functype[0]))
        else: p('Function used: {0}'.format(functype[1]))
        
    if isinstance(pval_threshold, (list, str, dict, tuple, pd.Series, np.ndarray, pd.DataFrame)):
        raise ValueError('pval_threshold val must be float, hypothesis test critical size i.e. 0.05')
    elif pval_threshold <= 0 or pval_threshold >= 0.1:
        raise ValueError('pval_threshold val must be non-zero positive float within range(0.01, 0.1) i.e. 0.05')
    else:
        pval_threshold = float(pval_threshold)
        
    if isinstance(thres, (list, str, dict, tuple, pd.Series, np.ndarray, pd.DataFrame)):
        raise ValueError('thres val must be float, kindly run func min_value provided see minimal d value for stationary i.e. 0.1!')
    elif thres <= 1e-2:
        warnings.warn('thres val <= 1.e-2 may not suit non-trend series, may take up longer than expected to calculate val i.e. 1e-5')
    else:
        thres = float(thres)
        
    if isinstance(num, (list, str, dict, tuple, pd.Series, np.ndarray, pd.DataFrame)):
        raise ValueError('num val must be postive integer or non-decimal positive float i.e. 100!')
    elif num < 100:
        warnings.warn('num val must be more than or equal 100 default val is 100 i.e. 200')
    elif num > 10000:
        warnings.warn('num val is more than 10,000 which take up more than expected time to calculate.')
    else:
        thres = int(thres)
    
    d_domain = np.linspace(start = 0, 
                           stop = 2, 
                           num = num, 
                           endpoint = True, 
                           retstep = False, 
                           dtype = float)
    
    for d in d_domain:
        df1 = np.log(data).resample('1D').last() # pls note downcast to daily obs
        df2 = func(df1, d, thres = thres).dropna()
        df2 = adfuller(df2.squeeze(), maxlag=1, regression='c', autolag=None)
        try:
            if df2[1] <= pval_threshold:
                return d
        except:
            p('Something is wrong! Most likely required d value more than 2!!')


def plot_min_ffd(close_prices: pd.Series, max_d = 2, pval_threshold: float = 0.05):
    '''    
    params: close_price: pd.data
    max_d: maximum value to differentiate (optional)
    pval_threshold: p-value to pass ADF test
    '''
    if pval_threshold > 0.05:
        print('p-value for ADF should be less than 0.05 to confirm for stationarity')
        return
    close_prices = close_prices.to_frame()
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    for d in np.linspace(0, max_d, 200):
        df1 = np.log(close_prices[['close']]).resample('1D').last()  # downcast to daily obs        
        df1.dropna(inplace=True)
        df2 = fracDiff_FFD(data = df1, d = d, thres = 1e-2).dropna()
        corr = np.corrcoef(df1.loc[df2.index, 'close'], df2['close'])[0, 1]
        df2 = adfuller(df2['close'], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr]  # with critical value
        try:
            if df2[1] <= pval_threshold:
                return d
        except:
            p('Something is wrong! Most likely required d value more than 2!!')
    out[['adfStat', 'corr']].plot(secondary_y='adfStat', figsize=(10, 8))
    plt.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    return