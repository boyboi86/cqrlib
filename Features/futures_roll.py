import pandas as pd
import numpy as np


def roll_gaps(data, dictio={"Instrument":"Ticker", "Open": "Open", "Close":"Close"}, 
              matchEnd = True):
    '''
    matchEnd = True means roll backwards False is roll fowards
    Take note due to contango; cause negative figure
    '''
    roll_dates = data[dictio['Instrument']].drop_duplicates(keep='first').index
    gaps = data[dictio['Close']] * 0.0
    iloc=list(data.index)
    iloc=[iloc.index(i) - 1 for i in roll_dates] #no of days prior to roll
    gaps.loc[roll_dates[1:]] = data[dictio['Open']].loc[roll_dates[1:]] - data[dictio['Close']].iloc[iloc[1:]].values
    gaps = gaps.cumsum()
    if matchEnd: 
        gaps -= gaps.iloc[-1] #roll backwards
    return gaps


def roll_futures(data, key):
    '''
    Basically to set datetime index
    before invoking rolling plus you need VWAP
    '''
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%Sf")
    df = df.set_index("time")
    gaps = roll_gaps(data)
    for fld in ['Close', 'VWAP']: data[fld] -=gaps
    return data