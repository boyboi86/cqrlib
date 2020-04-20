import pandas as pd
import datetime as dt
import pandas_datareader.data as pdr

def single_sampledata(period: int, ticker: str, date_input: dt.datetime, Today = True):
    '''
    Can choose to set prefered date
    for some reason Yahoo adj close is same as close
    period: int
    ticker: str
    date_input: dt.datetime
    '''
    if Today == True or date_input == None:
        date_input = dt.date.today()
    df = pd.DataFrame()
    sampleperiod = 365 * period
    startdate = date_input - dt.timedelta(days = sampleperiod)
    try:
        temp = pdr.get_data_yahoo(ticker, startdate, date_input)
        temp.dropna(inplace = True)
        df['Open'] = temp["Open"]
        df['High'] = temp["High"]
        df['Low'] = temp["Low"]
        df['Close'] = temp["Adj Close"]
        df["V"] = temp["Volume"]
        df["DV"] = temp["Adj Close"] * temp["Volume"]
        print("Data retrieved..")
        return df
    except:
        print("Data cannot be retrieved..")
        