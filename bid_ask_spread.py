# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:15:09 2020

based on SEV gumbel distribution to measure correlation between spread difference

change in spread difference and last filled price to measure direction price movement

both gumbel/ gauss will try to check if price is trending or not

For now we assume them in terms of percentile as a rolling trend

imbalance spread bars will check if uptick or down tick based on expected values

@author: Wei_X
"""

import numpy as np
import scipy.stats as si
import pandas as pd

p = print

#==============================================================================
# spread divation from last filled and mid range against bid and ask
# try out imbalance charting before using guass or ESV/ gumbel to check directional change
#==============================================================================

def ema_tick(imbalance, weighted_count, weighted_sum, weighted_sum_T, limit, alpha, T_count):
    weighted_sum_T = limit + (1 - alpha) * weighted_sum_T
    weighted_sum = limit / (1.0 * T_count) + (1 - alpha) * weighted_sum
    weighted_count = 1 + (1 - alpha) * weighted_count
    imbalance = weighted_sum_T * weighted_sum/ weighted_count ** 2
    return imbalance, weighted_count, weighted_sum, weighted_sum_T

def spreads_bars(bid, ask, last, set_limit, alpha):
    _bid = bid
    _ask = ask
    _mid = []
    _last = last
    _st = []
    _stup = 0
    _stdn = 0
    _stsum = 0
    _stcount = 0
    imbalance = 0
    weighted_sum_T = 0
    weighted_sum = 0
    weighted_count = 0
    imb_arr = []
    for i, value in enumerate(_bid):
        m = (value + _ask[i])/2
        _mid.append(m)
        diff = _last[i] - _mid[i]
        _st.append(diff)
        _stsum += diff
        _stcount += abs(diff)
        if diff >= 0:
            _stup += diff
        else:
            _stdn.append(diff)
        upper_limit = max(_stsum, _stup)
        if upper_limit >= set_limit:
            imbalance, weighted_count, weighted_sum, weighted_sum_T = ema_tick(imbalance, weighted_count, weighted_sum, weighted_sum_T, upper_limit, alpha, _stcount)
            imb_arr.append(imbalance) # exclude ewma without hitting threshold
            _stup = 0
        else:
            imb_arr.append(imbalance)    
    return _st, imb_arr, _bid, _ask, _mid

#==============================================================================
# spread divation from last filled and mid range against bid and ask
# while guass will assume 68% or 1std between bid and ask
# mid point would be 50% percentile
# any deviation will reflect a probabilistic expected rise or fall based on last filled
# based on rolling tsa to detect possible trend
#==============================================================================
def spread_percentile(high, low, current):
    if high == current:
        return 1
    elif low == current:
        return 0
    else:
        pct = (high - current) / (current + low)
        return pct
    

def spreads_dev(bid, ask, last, set_limit, alpha):
    _bid = bid
    _ask = ask
    _mid = []
    _last = last
    _st = []
    _stsum = 0
    _low = 0
    _high = 0
    spd_pct_arr = []
    for i, value in enumerate(_bid):
        m = (value + _ask[i])/2
        _mid.append(m)
        diff = _last[i] - _mid[i]
        _st.append(diff)
        _stsum += diff
        if diff > _high:
            _high = diff
        elif diff < _low:
            _low = diff
        spd_pct = spread_percentile(_high, _low, diff)
        spd_pct_arr.append(spd_pct)
    return spd_pct, spd_pct_arr
        
