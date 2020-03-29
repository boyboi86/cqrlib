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
from calldata import sampledata

# CBOE benchmark
period = 5
tickers = ["^GSPC", "^VIX"]
p = print


df = sampledata(period, tickers)

#==============================================================================
# Probability based on spot vs strike
# for future option underlying is the future price itself
# get ln return to derive ave rolling period std
# finally annualized according to option leftover period
# This however will underestimate volatility therefore just use VIX by 50%
#==============================================================================

def AnnualizedVolatility(window):
    df["LnReturn"] = np.log(df["^GSPC"]) - np.log(df["^GSPC"]).shift(1)
    df.dropna(inplace=True)
    df["Volatility"] = pd.DataFrame.rolling(df["LnReturn"] , window).std()
    df.dropna(inplace=True)
    df["AdjVol"] = df["Volatility"] * np.sqrt(window)
    return df["AdjVol"][-1]
    
    


#==============================================================================
# some use ATM option IV = SPX VIX as input valid only for EOD
# Rmb ppf 68% 95% 99.7% [1std, 2std, 3std]
# Assume 365 days trading left on option
#==============================================================================

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

#==============================================================================
# 2 cloest strike from the spot price
# Width of strike is writer's closet short strikes from spot
# The below works for both iron condor and vertical spread
#==============================================================================
    
def PoPCalculator(premium, fees, Lstrike, Hstrike):
    width = Hstrike - Lstrike
    pop = 1 - (premium - fees) / width
    p("=" * 33)
    p(("Probability of Profit: {0} width: {1}").format(round(pop, 4), width))
    p("=" * 33)

PoPCalculator(147.5, 0.26, 2600, 2800)
#==============================================================================
# Prob of profit formula for vertical spread credit
# theortical price not valid for actual spread
# put price and call price has to be retrieved from CBOE/ marketmaker for IV
# Valid only for EOD since insuffice data
# threshold is for min risk premium decimal 5% = 0.05
# DTM = Date to maturity
#==============================================================================
    
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
    
