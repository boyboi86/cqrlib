# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:20:06 2020

@author: Wei_X
"""
import pandas as pd
import datetime as dt
import pandas_datareader.data as pdr

def single_sampledata(period: int, ticker: str, date_input: dt.datetime, Today = True):
    '''
    Can choose to set prefered date
    it would retrieve adj close price and volume
    dollar value is merely something for data driven bar
    call only single asset
    if you wish to use unique volatility estimator you might need to change the below code
    period: int
    ticker: str
    date_input: dt.datetime
    '''
    if Today == True:
        date_input = dt.date.today()
    df = pd.DataFrame()
    sampleperiod = 365 * period
    startdate = date_input - dt.timedelta(days = sampleperiod)
    try:
        temp = pdr.get_data_yahoo(ticker, startdate, date_input)
        temp.dropna(inplace = True)
        df[ticker] = temp["Adj Close"]
        df["V"] = temp["Volume"]
        df["DV"] = temp["Adj Close"] * temp["Volume"]
        print("Data retrieved..")
        return df
    except:
        print("Data cannot be retrieved..")