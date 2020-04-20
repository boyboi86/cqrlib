"""
This has to be used with order book to track orders flow abnormalities

possible idea for liquidity check VWAP strategy or bid-ask spread PCA

The logic is as per below, but due to high serial correlation (weak statistical property)

Mean-reversion strategies are better off with just standard bars

You may look at the below logic, since it's not easy to implement

"""

import numpy as np
import pandas as pd

#==============================================================================
# calculate log returns for dataframe
# can choose to return pd.series instead
# Can play ard without log return use pct_chg()
#==============================================================================
def returns(data, tickers):
    b_t = []
    _ = data[tickers].pct_change()
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
            imbalance, weighted_count, weighted_sum, weighted_sum_T = ema_tick(imbalance, 
                                                                               weighted_count,
                                                                               weighted_sum,
                                                                               weighted_sum_T,
                                                                               upper_limit,
                                                                               alpha,
                                                                               bt_count)
            imb_arr.append(imbalance) # exclude ewma without hitting threshold
            if upper_limit == bt_up:
                bt_up = 0
            else:
                b_imb_sum = 0
        else:
            imb_arr.append(0.0)    
    return bt_arr, imb_arr, b_t

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
