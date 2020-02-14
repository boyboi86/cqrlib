# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:28:51 2020

@author: Wei_X
"""

from bsm_model import IV

# Read SPY CVS file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import ProbPlot
from arch.univariate import ConstantMean, GARCH, Normal
from arch.unitroot import ADF, KPSS
import datetime as dt
import pandas_datareader.data as pdr

p = print

#IV(option_type, sigma, S, K, r, T, q)
#p(IV('c', 0.3, 3, 3, 6, 30/365, 0.6))

# CBOE benchmark
period = 5
tickers = ["^VIX", "^GSPC"]

# =============================================================================
# return dataframe
# ============================================================================= 

def sampledata(period, tickers):
    df = pd.DataFrame() # dataframe to store close price of each ticker
# =============================================================================
# Sample Period yearly basis
# =============================================================================
    period = period
    sampleperiod = 365 * period
    startdate = dt.date.today() - dt.timedelta(days = sampleperiod)
        
    # =============================================================================
    # Ticker symbol inputs
    # =============================================================================
    tickers = tickers
    # =============================================================================
    # return dataframe
    # ============================================================================= 
    #df = pd.DataFrame() # dataframe to store close price of each ticker
            
            
    attempt = 0 # initializing passthrough variable
    drop = [] # initializing list to store tickers whose close price was successfully extracted
    while len(tickers) != 0 and attempt <= 5:
        tickers = [j for j in tickers if j not in drop] # removing stocks whose data has been extracted from the ticker list
        for i in range(len(tickers)):
            try:
                temp = pdr.get_data_yahoo(tickers[i], startdate, dt.date.today())
                temp.dropna(inplace = True)
                df[tickers[i]] = temp["Adj Close"]
                drop.append(tickers[i])       
            except:
                p(tickers[i]," :failed to fetch data...retrying")
                continue
        attempt+=1
    return df

df = sampledata(period, tickers)
