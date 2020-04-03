# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:30:43 2020

@author: Wei_Xiang

Pls note the following maybe applicable to options in directional strategy 
but futures might be a cheaper alternative when factoring tax (60/40) and fees involved.

This is considered advance charting methods or what they called imbalance/ information driven bars. 

Although the below seems to be popular with crypto traders
Theory & formula from Marcos Lopez's book (Advances in Financial Machine Learning)
"""
def returns(data, tickers):
    b_t = []
    v_t = []
    d_t = []
    data["value"] = data["vol"] * data[tickers]
    _ = df[tickers].pct_change()
    _v = df["vol"].pct_change()
    _d = data["value"].pct_change()
    _.dropna(inplace=True)
    _v.dropna(inplace=True)
    _d.dropna(inplace=True)
    for i, value in enumerate(_):
        b_t.append(value)
        v_t.append(_v[i])
        d_t.append(_d[i])
    return b_t, v_t, d_t

#==============================================================================
# E0 [θT ] =E0 [T](P[bt = 1] − P[bt = −1])
# Where practice E0 [T] = ewma of T
# where in practice (P[bt = 1] − P[bt = −1]) = ewma cumsum signal
#==============================================================================

def ema_tick(imbalance, weighted_count, weighted_sum, weighted_sum_T, limit, alpha, T_count):
    weighted_sum_T = limit + (1 - alpha) * weighted_sum_T
    weighted_sum = limit / (1.0 * T_count) + (1 - alpha) * weighted_sum
    weighted_count = 1 + (1 - alpha) * weighted_count
    imbalance = weighted_sum_T * weighted_sum/ weighted_count ** 2
    return imbalance, weighted_count, weighted_sum, weighted_sum_T

def tick_bar(data, tickers, set_limit, alpha):
    b_t, v_t, d_t = returns(data, tickers)
    bt_arr = []
    imb_arr = []
    weighted_sum_T = 0
    weighted_sum = 0
    weighted_count = 0
    bt_count = 0
    bt_up = 0
    b_imb_sum = 0
    imbalance = 0
    for i, value in enumerate(v_t):
        bt_count += 1
        if value > 0:
            b_imb_sum += value
            bt_up += value
            bt_arr.append(b_imb_sum)
        else:
            b_imb_sum += value
            bt_arr.append(b_imb_sum)
            
        upper_limit = max(b_imb_sum, bt_up)
        if upper_limit >= set_limit:
            imbalance, weighted_count, weighted_sum, weighted_sum_T = ema_tick(imbalance, weighted_count, weighted_sum, weighted_sum_T, upper_limit, alpha, bt_count)
            imb_arr.append(imbalance) # exclude ewma without hitting threshold
            bt_up = 0
        else:
            imb_arr.append(imbalance)    
    return bt_arr, imb_arr, b_t, v_t, d_t

def vol_bar(data, tickers, set_limit, alpha):
    b_t, v_t, d_t = returns(data, tickers)
    vt_arr = []
    imb_arr = []
    weighted_sum_T = 0
    weighted_sum = 0
    weighted_count = 0
    vt_count = 0
    vt_up = 0
    v_imb_sum = 0
    imbalance = 0
    for i, value in enumerate(v_t):
        vt_count += 1
        if value > 0:
            v_imb_sum += value
            vt_up += value
            vt_arr.append(v_imb_sum)
        else:
            v_imb_sum += value
            vt_arr.append(v_imb_sum)
            
        upper_limit = max(v_imb_sum, vt_up)
        if upper_limit >= set_limit:
            imbalance, weighted_count, weighted_sum, weighted_sum_T = ema_tick(imbalance, weighted_count, weighted_sum, weighted_sum_T, upper_limit, alpha, vt_count)
            imb_arr.append(imbalance) # exclude ewma without hitting threshold
            vt_up = 0
        else:
            imb_arr.append(imbalance)    
    return vt_arr, imb_arr, b_t, v_t, d_t

def dollar_bar(data, tickers, set_limit, alpha):
    b_t, v_t, d_t = returns(data, tickers)
    dt_arr = []
    imb_arr = []
    weighted_sum_T = 0
    weighted_sum = 0
    weighted_count = 0
    dt_count = 0
    dt_up = 0
    d_imb_sum = 0
    imbalance = 0
    for i, value in enumerate(d_t):
        dt_count += 1
        if value > 0:
            d_imb_sum += value
            dt_up += value
            dt_arr.append(d_imb_sum)
        else:
            d_imb_sum += value
            dt_arr.append(d_imb_sum)
            
        upper_limit = max(d_imb_sum, dt_up)
        if upper_limit >= set_limit:
            imbalance, weighted_count, weighted_sum, weighted_sum_T = ema_tick(imbalance, weighted_count, weighted_sum, weighted_sum_T, upper_limit, alpha, dt_count)
            imb_arr.append(imbalance) # exclude ewma without hitting threshold
            dt_up = 0
        else:
            imb_arr.append(imbalance)    
    return dt_arr, imb_arr, b_t, v_t, d_t
