# -*- coding: utf-8 -*-
"""
construct simple data bars
Can use yahoo data or retrieve single data call from utli folder
"""
import numpy as np
import pandas as pd

def dd_bars(df: pd.DataFrame, column: pd.Series, m: int):
    t, ts, idx = column, 0, []
    for i, x in enumerate(t):
        ts += x
        if ts >= m:
            ts = 0; idx.append(i)
            continue
    return df.iloc[idx]

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

def tick_bar(df0: pd.DataFrame, m: int):
    if 'Close' in df.columns:
        column = df['Close']
    t_b = dd_bars(df, column, m)
    return t_b
