# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:27:10 2020

@author: Wei_X
"""

import pandas as pd
import numpy as np

def daily_vol_est(close: pd.Series, window: int = 20):
    vol = close.pct_change().dropna()
    vol = vol.ema(window).std()
    return vol

def get_parksinson_vol(high: pd.Series, low: pd.Series, window: int = 20):
    ret = np.log(high / low)  # High/Low return
    estimator = 1 / (4 * np.log(2)) * (ret ** 2)
    return np.sqrt(estimator.rolling(window=window).mean())


def get_garman_class_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                         window: int = 20):
    ret = np.log(high / low)  # High/Low return
    close_open_ret = np.log(close / open)  # Close/Open return
    estimator = 0.5 * ret ** 2 - (2 * np.log(2) - 1) * close_open_ret ** 2
    return np.sqrt(estimator.rolling(window=window).mean())


def get_yang_zhang_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                       window: int = 20):
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    open_prev_close_ret = np.log(open / close.shift(1))
    close_prev_open_ret = np.log(close / open.shift(1))

    high_close_ret = np.log(high / close)
    high_open_ret = np.log(high / open)
    low_close_ret = np.log(low / close)
    low_open_ret = np.log(low / open)

    sigma_open_sq = 1 / (window - 1) * (open_prev_close_ret ** 2).rolling(window=window).sum()
    sigma_close_sq = 1 / (window - 1) * (close_prev_open_ret ** 2).rolling(window=window).sum()
    sigma_rs_sq = 1 / (window - 1) * (high_close_ret * high_open_ret + low_close_ret * low_open_ret).rolling(
        window=window).sum()

    return np.sqrt(sigma_open_sq + k * sigma_close_sq + (1 - k) * sigma_rs_sq)