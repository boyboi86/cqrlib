# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:54:18 2020

@author: Wei_X
"""

import numpy as np
import scipy.stats as si
from arch.unitroot import ADF, KPSS

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

def unit_test(series_input):
    adf = ADF(series_input)
    kpss = KPSS(series_input)
    adf.trend = 'ct'
    kpss.trend = 'ct'
    if (adf.pvalue <= 0.05 and kpss.pvalue >= 0.05):
        p("=" * 33)
        p("\nADF & KPSS: Strong evidence process is stationary\n")
        p("=" * 33)
    elif adf.pvalue <= 0.05:
        p("=" * 33)
        p(adf.summary())
        p('\nReject Null hypothesis: Process is stationary\n')
        p("=" * 33)
    elif kpss.pvalue >= 0.05:
        p("=" * 33)
        p(kpss.summary())
        p('\nFail to reject Null hypothesis: Process is stationary\n')
        p("=" * 33)
    else:
        p("=" * 33)
        p('\nProcess has unit root therefore not stationary\n')
        p("=" * 33)