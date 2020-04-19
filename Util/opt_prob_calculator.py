# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:51:09 2020

@author: Wei_X
"""

#==============================================================================
# Credit Spread probability calculator using guass distribution based on spot
# Credit Spread Premium Calculator based on actual market pricing input included
#==============================================================================

import numpy as np
import pandas as pd
import scipy.stats as si

# CBOE benchmark
period = 5
tickers = ["^GSPC", "^VIX"]
p = print

def ProbCalculate(spot, strike, window, sigma):
    sigma = 0.01 * sigma * np.sqrt(window / 365)
    z = np.log(strike/spot) / sigma
    BelowStrike = si.norm.cdf(z)
    return BelowStrike

def PutProbCal(spot, strike, window):
    ProbITM = round(ProbCalculate(spot, strike, window) * 100, 4)
    width = abs(spot - strike)
    p("=" * 33)
    p(("Probability of below strike: {0}%").format(ProbITM))
    p(("Width of strike: {0}").format(round(width, 4)))
    p("=" * 33)
    return ProbITM
    
def CallProbCal(spot, strike, window):
    ProbOTM = ProbCalculate(spot, strike, window)
    width = abs(spot - strike)
    ProbITM = round((1 - ProbOTM) * 100, 4)
    p("=" * 33)
    p(("Probability of above strike: {0}%").format(round(ProbITM, 4)))
    p(("Width of strike: {0}").format(round(width, 4)))
    p("=" * 33)
    return ProbITM

def PoPCalculator(premium, fees, Lstrike, Hstrike):
    width = Hstrike - Lstrike
    pop = 1 - (premium - fees) / width
    p("=" * 33)
    p(("Probability of Profit: {0} width: {1}").format(round(pop, 4), width))
    p("=" * 33)
    
def riskpremiumPut(premium, spot ,Lstrike, Hstrike, DTM, threshold):
    width = Hstrike - Lstrike 
    probITM = PutProbCal(spot, Hstrike , DTM)
    rp = premium/ probITM * width
    if rp > threshold:
        p("Transaction valid")
    else:
        p("Transaction invalid")
        
def riskpremiumCall(premium, spot ,Lstrike, Hstrike, DTM, threshold):
    width = Hstrike - Lstrike 
    probITM = CallProbCal(spot, Hstrike , DTM)
    rp = premium/ probITM * width
    if rp > threshold:
        p("Transaction valid")
    else:
        p("Transaction invalid")  
    
