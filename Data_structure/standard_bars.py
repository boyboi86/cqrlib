# -*- coding: utf-8 -*-
"""
construct simple data bars
Can use yahoo data or retrieve single data call from utli folder
"""
import numpy as np
import pandas as pd

def transact_vol(data: pd.DataFrame, pct: float = 0.1):
    bar_size = data.resample('1d').mean() * pct
    return bar_size

def dd_bars(data: pd.DataFrame, column: pd.Series, m: int = None):
    '''
    params: data => dataframe of close series
    params: column => column of data sample; vol, dollar etc
    '''
    if m is None:
        m = transact_vol(data = data)
    
    t, ts, idx = column, 0, []
    for i, x in enumerate(t):
        ts += x
        if ts >= m:
            ts = 0; idx.append(i)
            continue
    return data.iloc[idx]

def dollar_bar(df: pd.DataFrame, m: int):
    if 'DV' in df.columns:
        column = df['DV']
    d_b = dd_bars(df, column, m)
    return d_b

def volume_bar(df: pd.DataFrame, m: int):
    if 'V' in df.columns:
        column = df['V']
    v_t = dd_bars(df, column, m)
    return v_t

def tick_bar(df: pd.DataFrame, m: int):
    if 'Close' in df.columns:
        column = df['Close']
    t_b = dd_bars(df, column, m)
    return t_b
