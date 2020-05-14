import pandas as pd
import numpy as np

def vol(data: pd.Series, span0: int = 100, period: str = 'days', num_period: int = 1):
    '''
    AFML page 44. 3.1 snippet
    Modify from the original daily volatility
    
    This will retrieve an estimate of volatility based on initial params
    The modification is to track stablility count etc and for other research
    
    param: pd.Series => data use close price
    param: int => num of samples for ewm std
    param: datetime/ string => specify Day, Hour, Minute, Second, Milli, Micro, Nano
    param: int => frequency, integer only
    '''
    if isinstance(data, (str, int, float)):
        raise ValueError("data must be pandas series, 1d array with datetimeIndex i.e close price series")
    
    if isinstance(span0, (str, list, pd.Series, np.ndarray) or span0 <= 0):
        raise ValueError("span0 must be non-zero positive integer or float i.e 21 or 7.0")
        
    if isinstance(period, (float, int, list, pd.Series, np.ndarray)):
        raise ValueError("period must string i.e 'days', 'mins'")
        
    if isinstance(num_period, (str, list, pd.Series, np.ndarray)) or num_period <= 0:
        raise ValueError("num_period must non-zero positive integer i.e 100, 50")
    else:
        num_period = int(num_period)
    
    freq = str(num_period) + period
    df0 = data.index.searchsorted(data.index - pd.Timedelta(freq))
    df0 = df0[df0 > 0]
    df0 = pd.Series(data.index[df0 - 1], index = data.index[data.shape[0] - df0.shape[0]:])
    df0 = data.loc[df0.index]/data.loc[df0.array].array - 1
    df0=df0.ewm(span = span0).std()
    return df0

def parksinson_vol(high: pd.Series, low: pd.Series, window: int = 20):
    
    if isinstance(high, (str, int, float)):
        raise ValueError("high data must be pandas series, 1d array with datetimeIndex i.e close price series")
    elif high.isnull().values.any():
        raise ValueError("high data contains NaNs or isinf")
    
    if isinstance(low, (str, int, float)):
        raise ValueError("low data must be pandas series, 1d array with datetimeIndex i.e close price series")
    elif low.isnull().values.any():
        raise ValueError("low data contains NaNs or isinf")
        
    if isinstance(window, (str, list, pd.Series, np.ndarray)) or window <= 0:
        raise ValueError("num_period must non-zero positive integer i.e 100, 50")
    else:
        window = int(window)
    
    vol = high/ low
    ret = pd.Series(vol, index=high.index).apply(np.log)  # High/Low return
    estimator = 1 / (4 * np.log(2)) * (ret ** 2)
    return np.sqrt(estimator.rolling(window=window).mean())


def garman_class_vol(open_price: pd.Series, high: pd.Series, low: pd.Series, close_price: pd.Series, window: int = 20):
    
    if isinstance(open_price, (str, int, float)):
        raise ValueError("low data must be pandas series, 1d array with datetimeIndex i.e close price series")
    elif open_price.isnull().values.any():
        raise ValueError("low data contains NaNs or isinf")
    
    if isinstance(high, (str, int, float)):
        raise ValueError("high data must be pandas series, 1d array with datetimeIndex i.e close price series")
    elif high.isnull().values.any():
        raise ValueError("high data contains NaNs or isinf")
    
    if isinstance(low, (str, int, float)):
        raise ValueError("low data must be pandas series, 1d array with datetimeIndex i.e close price series")
    elif low.isnull().values.any():
        raise ValueError("low data contains NaNs or isinf")
        
    if isinstance(close_price, (str, int, float)):
        raise ValueError("low data must be pandas series, 1d array with datetimeIndex i.e close price series")
    elif close_price.isnull().values.any():
        raise ValueError("low data contains NaNs or isinf")
        
    if isinstance(window, (str, list, pd.Series, np.ndarray)) or window <= 0:
        raise ValueError("num_period must non-zero positive integer i.e 100, 50")
    else:
        window = int(window)
    
    hi_lo = high / low
    ret = pd.Series(hi_lo, index = close_price.index).apply(np.log) # High/Low return
    close_open = close_price / open_price
    cl_op_ret = pd.Series(close_open, index = close_price.index).apply(np.log) # Close/Open return
    df0 = pd.DataFrame(index = cl_op_ret.index).assign(ret = ret, 
                                                      cl_op_ret = cl_op_ret)
    for idx in df0.index:
        df0['estimate'].loc[idx] = 0.5 * df0.ret.at[idx] ** 2 - (2 * np.log(2) - 1) * df0.cl_op_ret.at[idx] ** 2
    df0['estimate'] = df0['estimate'].rolling(window=window).mean() 
    return df0['estimate'].apply(np.sqrt)


def yang_zhang_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
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