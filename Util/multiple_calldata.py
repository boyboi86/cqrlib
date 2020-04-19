# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:28:51 2020

@author: Wei_X
"""

# Read SPY CVS file

import pandas as pd
import datetime as dt
import pandas_datareader.data as pdr

p = print

# =============================================================================
# return dataframe
# ============================================================================= 

def multiple_sampledata(period, tickers):
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
