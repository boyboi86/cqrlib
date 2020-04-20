# -*- coding: utf-8 -*-
"""
Cumsum filters for mean reversion
 based on Marco Lopez AFML and filtering strategy research by Fama and Blume[1966]
"""

import numpy as np
import pandas as pd

def cumsum_events(df: pd.Series, limit: float):
    idx, _up, _dn = [], 0, 0
    diff = df.diff()
    for i in diff.index[1:]:
        _up, _dn = max(0, _up + diff.loc[i]), min(0, _dn + diff.loc[i])
        if _up > limit:
            _up = 0; idx.append(i)
        elif _dn < - limit:
            _dn = 0; idx.append(i)
        
    return pd.DatetimeIndex(idx)

def cumsum_filter(df: pd.DataFrame, limit: float):
    if 'Close' in df.columns:
        column = df['Close']
    df0 = cumsum_events(column, limit)
    return df.reindex(df0)