import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlfinlab.features import fracdiff
from statsmodels.tsa.stattools import adfuller

p = print

def min_ffd_value(data: pd.Series, pval_threshold: float = 0.05):
    """Minimal value of d which makes pandas series stationary.

    Arguments:
        unstationary_pdseries {pd.Series} -- Pandas series for which you want to find d.
        d_domain {np.array} -- sequence of possible values of d.

    Keyword Arguments:
        pvalue_threshold {float} -- P value threshold (default: {0.05})

    Returns:
        [float] -- Minimum value of d that makes series staionary.
    """
    d_domain = np.linspace(start = 0, 
                           stop = 2, 
                           num=100, 
                           endpoint=True, 
                           retstep=False, 
                           dtype=float)
    
    for d in d_domain:
        df1 = np.log(data).resample('1D').last()
        df1.dropna(inplace=True)
        df2 = fracdiff.frac_diff_ffd(df1, 
                                     diff_amt = d, 
                                     thresh = 0.01).dropna()
        df2 = adfuller(df2.squeeze(), maxlag=1, regression='c', autolag=None)
        try:
            if df2[1] <= pval_threshold:
                return d
        except:
            p('Something is wrong! Most likely required d value more than 2!!')


def plot_min_ffd(close_prices: pd.Series, max_d = 2, pval_threshold: float = 0.05):
    '''    
    params: close_price: pd.Series
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
        df2 = fracdiff.frac_diff_ffd(df1, 
                                     diff_amt = d, 
                                     thresh = 0.01).dropna()
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