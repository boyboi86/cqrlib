# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 21:50:20 2020

@author: Wei_X
"""

# Import necesary libraries
import pandas as pd
import pandas_datareader.data as pdr
import scipy.stats as si
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

p = print
period = 5
tickers = ["^VIX", "^GSPC"]

# =============================================================================
# return dataframe
# ============================================================================= 
df = pd.DataFrame() # dataframe to store close price of each ticker

def sampledata(period, tickers):
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
# Since CBOE VIX is already IV of S&P500
# Measure percentile scores
# Length of period 30d, 60d, 90d, 180d, 360d, 720d

revdf = df[::-1]

# =============================================================================
# Calculate rolling percentile scoring
# Only Series no Dataframe
# ============================================================================= 
def rolling_percentileofscore(series, window):

    def _percentile(arr):
        score = arr[-1]
        vals = arr[:-1]
        return si.percentileofscore(vals, score)

    _ = series.dropna()
    if _.empty:
        return pd.Series(np.nan, index=series.index)
    else:
        return _.rolling(window).apply(_percentile, raw=True).reindex(series.index) 

df['period90'] = rolling_percentileofscore(series = df['^VIX'], window = 90)
df['period180'] = rolling_percentileofscore(series = df['^VIX'], window = 180)
df['period252'] = rolling_percentileofscore(series = df['^VIX'], window = 252)
df['period360'] = rolling_percentileofscore(series = df['^VIX'], window = 360)
df['period504'] = rolling_percentileofscore(series = df['^VIX'], window = 504)
df['period600'] = rolling_percentileofscore(series = df['^VIX'], window = 600)
df['pct_rtn'] = df['^GSPC'].pct_change()

df1 = df.dropna()
df1 = df1.drop(['^VIX','^GSPC'], axis = 1)
periods = [df1['period90'], df1['period180'], df1['period252'], df1['period360'], df1['period504'], df1['period600']]
labels = ['period90', 'period180', 'period252', 'period360', 'period504', 'period600']

def multipleplotkde(periods, labels):
    periods = periods
    plt.figure(figsize=(15,8))
    for period in periods:
        period.plot.kde()
        
    # Plot formatting
    plt.legend(labels)
    plt.title('Multiple Periods KDE Plot')
    plt.xlabel('Actual Percentile Calculated')
    plt.ylabel('Density')
    plt.show()
    
multipleplotkde(periods, labels)
# for 90d range given long short should be min 10% and max 85% for most signal
# while 600d range given long short should be around min 6% max 80% and for most signal 