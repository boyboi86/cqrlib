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

df, imb_arr, rtn, bt_arr = tick_bar(df, 3, 0.7)
