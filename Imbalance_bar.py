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
        if upper_limit >= set_limit:
            imbalance, weighted_count, weighted_sum, weighted_sum_T = ema_tick(imbalance, weighted_count, weighted_sum, weighted_sum_T, upper_limit, alpha, T_count)
            imb_arr.append(imbalance) # exclude ewma without hitting threshold
            rtn.append(df["net_lgrtn"][i])
            T_up = 0
        else:
            imb_arr.append(imbalance) # exclude ewma without hitting threshold
            rtn.append(df["net_lgrtn"][i])
            
            
    return df, imb_arr, rtn, bt_arr

#==============================================================================
# Try out basic Tick imbalance without Vol
# E0 [θT ] =E0 [T](P[bt = 1] − P[bt = −1])
# Where practice E0 [T] = ewma of T
# where in practice (P[bt = 1] − P[bt = −1]) = ewma cumsum signal
# Testout Lambda for ewma to fit S&P futures
# To be run seperately from above
#==============================================================================

def returns(data, tickers):
    b_t = []
    v_t = []
    _ = df[tickers].pct_change() * np.sqrt(365)
    _.dropna(inplace=True)
    for i, value in enumerate(_):
        b_t.append(value)
        v_t.append(df["vol"][i])
    return b_t, v_t

b_t, v_t = returns(df, tickers)

def ema_tick(imbalance, weighted_count, weighted_sum, weighted_sum_T, limit, alpha, T_count):
    weighted_sum_T = limit + (1 - alpha) * weighted_sum_T
    weighted_sum = limit / (1.0 * T_count) + (1 - alpha) * weighted_sum
    weighted_count = 1 + (1 - alpha) * weighted_count
    imbalance = weighted_sum_T * weighted_sum/ weighted_count ** 2
    return imbalance, weighted_count, weighted_sum, weighted_sum_T


def vol_bar(data, tickers, set_limit, alpha):
    b_t, v_t = returns(data, tickers)
    vt_arr = []
    bt_arr = []
    imb_arr = []
    weighted_sum_T = 0
    weighted_sum = 0
    weighted_count = 0
    vt_count = 0
    vt_up = 0
    vt_dn = 0
    v_imb_sum = 0
    imbalance = 0
    for i, value in enumerate(b_t):
        vt_count += v_t[i]
        if value > 0:
            bt_arr.append(1)
            vt_arr.append(v_t[i])
            v_imb_sum += v_t[i]
            vt_up += v_t[i]
            
        else:
            bt_arr.append(-1)
            vt_arr.append(-1 * v_t[i])
            v_imb_sum -= v_t[i]
            vt_dn -= v_t[i]
            
        upper_limit = max(v_imb_sum, vt_up)
        if upper_limit >= set_limit:
            imbalance, weighted_count, weighted_sum, weighted_sum_T = ema_tick(imbalance, weighted_count, weighted_sum, weighted_sum_T, upper_limit, alpha, vt_count)
            imb_arr.append(imbalance) # exclude ewma without hitting threshold
            vt_up = 0
        else:
            imb_arr.append(imbalance)    
    return vt_arr, bt_arr, imb_arr, v_t, b_t

