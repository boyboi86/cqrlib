# -*- coding: utf-8 -*-
"""
construct simple data bars
no numba included
"""
import numpy as np
import pandas as pd

def dd_bars(df, dv_column, m):
    t = df[dv_column]
    ts = 0
    idx = []
    for i, x in enumerate(t):
        ts += x
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx

def dd_bar_df(df, dv_column, m):
    idx = dd_bars(df, dv_column, m)
    return df.iloc[idx]


def dd_bar_cum(df, dv_column, m, window):
    df0 = dd_bar_df(df, dv_column, m)
    idx, cum_sum, _up, _dn = [], 0, 0, 0
    df0 = daily_vol(df0['ES=F'], span0 = 100)
    limit = df0.rolling(window).std()
    for i, value in enumerate(df0[1:]):
        cum_sum += value
        _up, _dn = max(0, _up + value), min(0, _dn + value)
        if value > limit[i]:
            _up = 0; idx.append(i)
        elif value < - limit[i]:
            _dn = 0; idx.append(i)
        
    return df0.iloc[idx]