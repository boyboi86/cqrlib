# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:30:43 2020

@author: Wei_Xiang

Pls note the following maybe applicable to options in directional strategy 
but futures might be a cheaper alternative when factoring tax (60/40) and fees involved

Although the below seems to be popular with crypto traders
Theory & formula from Marcos Lopez's book (Advances in Financial Machine Learning)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt
from calldata import sampledata

p = print
tickers = ["ES=F"]
period = 3

df = sampledata(period, tickers)

def datacalling(period, ticker):
    df = pd.DataFrame()
    sampleperiod = 365 * period
    startdate = dt.date.today() - dt.timedelta(days = sampleperiod)
    try:
        temp = pdr.get_data_yahoo(ticker, startdate, dt.date.today())
        temp.dropna(inplace = True)
        df[ticker] = temp["Adj Close"]
        df["vol"] = temp["Volume"]
        print("Data retrieved..")
        return temp, df
    except:
        print("Data cannot be retrieved..")
    

df = sampledata(period, tickers)
#==============================================================================
# calculate log returns for dataframe
# can choose to return pd.series instead
# Can play ard without log return use pct_chg()
#==============================================================================

def returns(data):
    df["returns"] = df["ES=F"].pct_change() * np.sqrt(365)
    df["net_rtn"] = df["returns"].cumsum()
    df["log_ret"] = (np.log(df["ES=F"]) - np.log(df["ES=F"].shift(1))) * np.sqrt(365)
    df["net_lgrtn"] = df["log_ret"].cumsum()
    df.dropna(inplace=True)
    return df

#==============================================================================
# Try out basic Tick imbalance without Vol
# E0 [θT ] =E0 [T](P[bt = 1] − P[bt = −1])
# Where practice E0 [T] = ewma of T
# where in practice (P[bt = 1] − P[bt = −1]) = ewma cumsum signal
# Testout Lambda for ewma to fit S&P futures
#==============================================================================
def ema_tick(imbalance, weighted_count, weighted_sum, weighted_sum_T, limit, alpha, T_count):
    weighted_sum_T = limit + (1 - alpha) * weighted_sum_T
    weighted_sum = limit / (1.0 * T_count) + (1 - alpha) * weighted_sum
    weighted_count = 1 + (1 - alpha) * weighted_count
    imbalance = weighted_sum_T * weighted_sum/ weighted_count ** 2
    return imbalance, weighted_count, weighted_sum, weighted_sum_T


def tick_bar(data, set_limit, alpha):
    df = returns(data)
    df["signals"] = pd.Series()
    df["net_signal"] = pd.Series()
    bt_arr = []
    imb_arr = []
    rtn = []
    weighted_sum_T = 0
    weighted_sum = 0
    weighted_count = 0
    T_count = 0
    T_up = 0
    T_dn = 0
    T_imb_sum = 0
    imbalance = 0
    for i, value in enumerate(df["returns"]):
        T_count += 1
        if value > 0:
            df["signals"][i] = 1
            bt_arr.append(1)
            T_imb_sum += 1
            T_up += 1
            df["net_signal"][i] = T_imb_sum
        else:
            df["signals"][i] = -1
            bt_arr.append(-1)
            T_imb_sum -= 1
            T_dn -= 1
            df["net_signal"][i] = T_imb_sum
        upper_limit = max(T_imb_sum, T_up)
        #lower_limit = min(T_imb_sum, T_dn)
        if upper_limit >= set_limit:
            imbalance, weighted_count, weighted_sum, weighted_sum_T = ema_tick(imbalance, weighted_count, weighted_sum, weighted_sum_T, upper_limit, alpha, T_count)
            imb_arr.append(imbalance) # exclude ewma without hitting threshold
            rtn.append(df["net_lgrtn"][i])
            T_up = 0
       # elif lower_limit <= -1 * set_limit:
       #     imbalance, weighted_count, weighted_sum, weighted_sum_T = ema_tick(imbalance, weighted_count, weighted_sum, weighted_sum_T, lower_limit, alpha, T_count)
       #     imb_arr.append(imbalance) # exclude ewma without hitting threshold
       #     rtn.append(df["net_lgrtn"][i])
       #     T_dn = 0
        else:
            imb_arr.append(imbalance) # exclude ewma without hitting threshold
            rtn.append(df["net_lgrtn"][i])
            
            
    return df, imb_arr, rtn, bt_arr

df, imb_arr, rtn, bt_arr = tick_bar(df, 3, 0.7)

#==============================================================================
# Try out basic Tick imbalance without Vol
# T index threshold set 2
# Where practice E0 [T] = ewma of T
# where in practice (P[bt = 1] − P[bt = −1]) = ewma cumsum signal
# Testout Lambda for ewma to fit S&P futures
# Checking correlation, signal generation is somewhat reliable sell/buy
#==============================================================================
def corr(imb_arr, rtn):
    imb = pd.Series(imb_arr)
    rt = pd.Series(rtn)
    corr = rt.rolling(7).corr(imb)
    return corr

corr = corr(imb_arr, rtn)

def graphplot():
    plt.figure(figsize=(15,8))
    plt.plot(rtn) #reduce plot days 90
    plt.plot(imb_arr)
        
    # Plot formatting
    plt.legend(["Accumulative return", "Imbalance Bars", "Directional change signal"])
    plt.title("Tick Imbalance Bar Vs Annualized Return (S&P 500 E-mini)")
    plt.xlabel("time days")
    plt.ylabel("log returns")
    plt.show()

graphplot()

def corplot():
    plt.figure(figsize=(15,8))
    plt.plot(corr)
        
    # Plot formatting
    plt.title("Tick Imbalance Bar Vs Annualized Return (S&P 500 E-mini)")
    plt.show()

corplot()