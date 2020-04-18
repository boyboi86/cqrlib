# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:30:43 2020

@author: Wei_Xiang

When tested against XSP future. Super reliable Signal. Require further testing.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as si
import pandas_datareader.data as pdr
import datetime as dt

p = print
tickers = "ES=F"
period = 0.6

#==============================================================================
# calculate log returns for dataframe
# if we use pct change in vol it's more of liquidity check/ market sentiment
# instead we will use overall volume for that day as signal magnitude for imbalance
# while tick signal will still be based on price change
# Tick bar is bias upwards hence it will generate more signal during up tick
# but there will be a drop in signal when overall price trend is downwards
#==============================================================================
def datacalling(period, ticker):
    df = pd.DataFrame()
    sampleperiod = 365 * period
    startdate = dt.date.today() - dt.timedelta(days = sampleperiod)
    try:
        temp = pdr.get_data_yahoo(ticker, startdate, dt.date.today())
        temp.dropna(inplace = True)
        df[ticker] = temp["Adj Close"]
        df["V"] = temp["Volume"]
        df["DV"] = temp["Adj Close"] * temp["Volume"]
        print("Data retrieved..")
        return df
    except:
        print("Data cannot be retrieved..")
    

df = datacalling(period, tickers)
#==============================================================================
# calculate log returns for dataframe
# can choose to return pd.series instead
# Can play ard without log return use pct_chg()
#==============================================================================

def returns(data, tickers):
    b_t = []
    _ = df[tickers].pct_change()
    _.dropna(inplace=True)
    for i, value in enumerate(_):
        b_t.append(value)
    return b_t

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

def imbalance_bar(data, tickers, set_limit, alpha):
    b_t = returns(data, tickers)
    bt_arr = []
    imb_arr = []
    weighted_sum_T = 0
    weighted_sum = 0
    weighted_count = 0
    bt_count = 0
    bt_up = 0
    b_imb_sum = 0
    b_sum = 0
    imbalance = 0
    for i, value in enumerate(b_t):
        bt_count += 1
        if value >= 0:
            b_sum += b_t[i]
            b_imb_sum += 1
            bt_up += 1
            bt_arr.append(b_sum)
        else:
            b_imb_sum -= 1
            b_sum += b_t[i]
            bt_arr.append(b_sum)
            
        upper_limit = max(b_imb_sum, bt_up)
        if upper_limit >= set_limit:
            imbalance, weighted_count, weighted_sum, weighted_sum_T = ema_tick(imbalance, weighted_count, weighted_sum, weighted_sum_T, upper_limit, alpha, bt_count)
            imb_arr.append(imbalance) # exclude ewma without hitting threshold
            if upper_limit == bt_up:
                bt_up = 0
            else:
                b_imb_sum = 0
        else:
            imb_arr.append(0.0)    
    return bt_arr, imb_arr, b_t

bt_arr, b_imb_arr, b_t = imbalance_bar(df, tickers, 4, 0.9)
vt_arr, v_imb_arr, v_t = imbalance_bar(df, "V", 4, 0.9)
dt_arr, d_imb_arr, d_t = imbalance_bar(df, "DV", 4, 0.9)

#==============================================================================
# Try out basic Tick imbalance without Vol
# T index threshold set 2
# Where practice E0 [T] = ewma of T
# where in practice (P[bt = 1] − P[bt = −1]) = ewma cumsum signal
# Testout Lambda for ewma to fit S&P futures
# Checking correlation, signal generation is somewhat reliable sell/buy
#==============================================================================

def sig(imb_type, imb_arr, bt_arr):
    total = 0
    total_sum = len(bt_arr)
    week_count = []
    week_count_sum = 0
    for i, value in enumerate(imb_arr):
        if value > 0:
            imb_arr[i] = bt_arr[i]
            total += 1
            week_count_sum += 1
        if i % 7 == 0:
            week_count.append(week_count_sum)
            week_count_sum = 0
    print("==================================================")
    print(("Statistic for {0}").format(imb_type))
    print(("Signal generated {0}, {1}% of sample period").format(total, round(total * 100/total_sum, 3)))
    print(("Ave signal {0} times per week").format(round(np.mean(week_count), 3)))
    print(("Median signal {0} times per week").format(round(np.median(week_count), 3)))
    print(("Std {0} times per week").format(round(np.std(week_count), 3)))
    print("==================================================")
    return imb_arr, total, week_count

def imb_bar_check(imb_type, imb_type_arr, bt_arr):
    imb_bars_arr = []
    imb_bars_count = []
    imb_week_count = []
    for i in range(len(imb_type_arr)):
        imb_arr, imb_total, imb_week_count = sig(imb_type[i], imb_type_arr[i], bt_arr)
        imb_bars_arr.append(imb_arr)
        imb_bars_count.append(imb_total)
        imb_week_count.append(imb_week_count)
    return imb_bars_arr, imb_bars_count, imb_week_count

imb_type = ["Tick Imbalance", "Volume Imbalance", "Dollar Imbalance"]
imb_type_arr = [b_imb_arr, v_imb_arr, d_imb_arr]
imb_bars_arr, imb_bars_count, imb_week_count = imb_bar_check(imb_type, imb_type_arr, bt_arr)

b_imb = pd.Series(imb_bars_arr[0])
v_imb = pd.Series(imb_bars_arr[1])
d_imb = pd.Series(imb_bars_arr[2])

def autocorr_check(pd_series):
    print(pd_series.autocorr())
    print(pd_series.autocorr(2))

def jb(x,test=True): 
    np.random.seed(12345678)
    if test: print(si.jarque_bera(x)[0])
    print(si.jarque_bera(x)[1])

def shapiro(x,test=True): 
    np.random.seed(12345678)
    if test: print(si.shapiro(x)[0])
    print(si.shapiro(x)[1])


def graphplot():
    plt.figure(figsize=(15,8))
    plt.plot(bt_arr)
    plt.scatter(b_imb.index[b_imb != 0], b_imb[b_imb != 0], color = "red", marker = "x")
    plt.scatter(v_imb.index[v_imb != 0], v_imb[v_imb != 0], color = "green", marker = "x")
    plt.scatter(d_imb.index[d_imb != 0], d_imb[d_imb != 0], color = "black", marker = "x")
    # Plot formatting
    plt.legend(["Price change", "Tick Imbalance Bars", "Vol Imbalance Bars", "Dollar Imbalance Bars"])
    plt.title("Imbalance Bar Buy Signal Vs Price Change (S&P 500 E-mini)")
    plt.xlabel("Time (Days)")
    plt.ylabel("Percentage change")
    plt.show()

#==============================================================================
# Future spread rollovers (multiple futures Natural Gas/ Cotton etc)
# assume a basket of asset similar to ETFs
# not all tradable but those that are tradable at t have to be include
# asset_arr = every of all tradable assets weightage within basket of futures
# weightage/ allocation based on number of contract held
# FX USD/ ???
# open_arr = opening price for next series t - 1 .. t+1 for all tradable asset that was weighted
# close_arr = close price for current series t - 1 .. t for all tradable asset that was weighted
#==============================================================================
'''
def weight_adj(open_arr, asset_arr, exchange_rate):
    total_alloc = np.cumsum(abs(asset_arr))
    weight_arr = []
    for i, value in enumerate(asset_arr):
        adj_weight = value/total_alloc * exchange_rate * open_arr[i][2]
        weight_arr.append(adj_weight)
    return weight_arr

def price_chng(open_arr, close_arr):
    for i in range(len(close_arr)):
        if i < 1:
            close_arr[i] - open_arr[i]

def future_spread_adj(open_arr, close_arr, asset_arr, exchange_rate):
    weight_arr = weight_adj(open_arr, asset_arr, exchange_rate)
'''
