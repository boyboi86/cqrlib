import pandas as pd
import numpy as np

def daily_vol(close: pd.Series, span0: int = 100):
    '''
    AFML page 44. 3.1 snippet
    The original daily volatility
    '''
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days = 1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index = close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index]/ close.loc[df0.array].array - 1
    df0 = df0.ewm(span = span0).std()
    return df0

def get_vol(data: pd.Series, span0: int = 100, period: str = 'days', num_period: int = 1):
    '''
    AFML page 44. 3.1 snippet
    Modify from the original daily volatility
    
    This will retrieve an estimate of volatility based on initial params
    The modification is to track stablility count etc and for other research
    
    param: pd.Series => data use close price
    param: int => num of samples for ewm std
    param: datetime/ string => specify Day, Hour, Minute, Second, Milli, Micro, Nano
    param: int => frequency
    '''
    freq = str(num_period) + period
    df0 = data.index.searchsorted(data.index - pd.Timedelta(freq))
    df0 = df0[df0 > 0]
    df0 = pd.Series(data.index[df0 - 1], index = data.index[data.shape[0] - df0.shape[0]:])
    df0 = data.loc[df0.index]/data.loc[df0.array].array - 1
    df0=df0.ewm(span = span0).std()
    return df0

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