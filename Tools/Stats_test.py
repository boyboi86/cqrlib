import numpy as np
import pandas as pd
import scipy.stats as si

from arch.unitroot import ADF, KPSS

import statsmodels.stats.diagnostic as sm
import statsmodels.api as smi

p = print

#==============================================================================
# Model volatility comparison
# Input data array output to test sample period stability
# optimal signal period test include original data along with test samples
#==============================================================================

vol_arr = []

def vol_stability_test(vol_arr):
    _vol = []
    for i in vol_arr:
        _vol.append(np.std(i))
        p("{0} with std {1}").format(vol_arr[0], _vol[0])
    _arr = sorted(zip(vol_arr, _vol), key=lambda x: x[1])
    return _arr

def jb(x,test=True): 
    np.random.seed(12345678)
    if test: 
        p(si.jarque_bera(x)[0])
    return si.jarque_bera(x)[1]

def shapiro(x,test=True): 
    np.random.seed(12345678)
    if test: 
        print(si.shapiro(x)[0])
    return si.shapiro(x)[1]
    
#==============================================================================
# Normality Test
# Against Skewness as well as Kurtosis of series
# If it is not single model, it could be used for goodness of fit comparison
#==============================================================================

def normality_test(series_input, test_limit = 0.05):
    p("=" * 33)
    count = len(series_input)
    p(("Total Sample: {0}").format(count))
    if count >= 2000:
        p("\nShapiro Wilk Test recommends at most 2000 sample")
        p("Otherwise test results might not be reliable\n")
    elif count <= 1900:
        p("Jarque-Bera Test recommends at least 2000 sample\n")
        p("Otherwise test results might not be reliable\n")
    else:
        p("Sample Size Suffice for Normality Test\n")
    p("Jarque-Bera Test Results")
    jb_ = jb(series_input,test=True)
    if jb_ >= test_limit:
        p("Null Hypothesis Accepted: Not Normal Distributed\n")
    else:
        p("Null Hypothesis Rejected: Normal Distributed\n")
    p("Shapiro Wilk Test Results")
    sw_ = shapiro(series_input,test=True)
    if sw_ <= test_limit:
        p("Null Hypothesis Rejected: Not Normal Distributed\n")
    else:
        p("Null Hypothesis Accepted: Normal Distributed\n")
    p("=" * 33)

#==============================================================================
# Stationarity Test
# Auto-regression effects
# If too strong, might require 2nd order differentiation
# Otherwise EWMA may be considered especially if [a,b] > 1
# If it is not single model, it could be used for goodness of fit comparison
#==============================================================================

def unit_test(data: pd.Series):
    adf = ADF(data)
    kpss = KPSS(data)
    adf.trend = 'c'
    kpss.trend = 'ct'
    if (adf.pvalue <= 0.05 and kpss.pvalue >= 0.05):
        p("\nADF & KPSS: Strong evidence process is stationary\n")
    elif adf.pvalue <= 0.05:
        p(adf.summary())
        p('\nReject Null hypothesis: Process is stationary\n')
    elif kpss.pvalue >= 0.05:
        p("=" * 33)
        p(kpss.summary())
        p('\nFail to reject Null hypothesis: Process is stationary\n')
    else:
        p('\nProcess has unit root therefore not stationary\n')

def white_test(data: pd.DataFrame):
    '''
    params: data => close price series
    White test is meant to test if series is homoscedastic or heteroscedastic
    null hypothsis: homoscedastic (> 0.05)
    alt hypothsis: heteroscedastic
    
    If homoscedastic would meant series have constant dispersion, ststistically good for mean reverting.
    Test uses OLS non-log. So do take note when using it.
    '''
    data['std1'] = data['close'].std()
    p("Residual: {0}\nExog count: {1}".format(data['std1'][0], data['close'].count()))
    data.dropna(inplace= True)
    X = smi.tools.tools.add_constant(data['close'])
    results = smi.regression.linear_model.OLS(data['std1'], X).fit()
    resid = results.resid
    exog = results.model.exog
    p("White-Test p-Value: {0}".format(sm.het_white(resid, exog)[1]))
    if sm.het_white(resid, exog)[1] > 0.05:
        p("White test outcome at 5% signficance: homoscedastic")
        p("Reject null hypothesis at critical size: 0.05")
    else:
        p("White test outcome at 5% signficance: heteroscedastic")
    return data.drop(columns = ['std1'])