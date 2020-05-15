import numpy as np
import pandas as pd
import datetime
import warnings
from research.Util.multiprocess import mp_pandas_obj


def _pt_sl_t1(data: pd.Series, events: pd.Series, ptSl: list, molecule):
    '''
    AFML pg 45 snippet 3.2
    
    Code snippet which tells you the logic of how a triple barrier will be formed.
    In the event if price target is touched first before vertical barrier.
    The func is created to be run with python multiprocesses module
    
    params: data => closing price
    params: events => new dataframe with timestamp index, target, metalabel 
    params: ptSl => an array [1,1] which will determine width of horizontal barriers
    params: data => molecule which is part of multiprocess module, to break down jobs and allow parallelisation
    
    '''
    events_ = events.reindex(molecule)
    out = events_[['t1']].copy(deep = True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index = events.index) #Series with index but no value NaNs
    if ptSl[1] > 0:
        sl = - ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index = events.index) #Series with index but no value NaNs
    for loc, t1 in events_['t1'].fillna(data.index[-1]).iteritems():
        df0 = data[loc:t1]
        df0 = (df0/data[loc] - 1) * events_.at[loc, 'side']
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
    return out


def tri_bar(data: pd.Series, events: pd.DatetimeIndex, trgt: pd.Series, min_req: float, 
               num_threads: int = 1, ptSl: list = [1,1], t1: pd.Series = False, side: pd.Series = None):
    '''
    AFML pg 50 snippet 3.6
    This function will return triple barrier data based on params.
    
    There is some amendment to the original snippet provided, to prevent error.
    Added side_ variable which act as boolean counter: 0 = No side prediction provided.(No primary model)
                                                       1 = Side prediction provided. (Primary model)
                                                       
    Some amendment to Pandas multiprocessor object, due to the my machine limitation. 
    Default thread is 1 for all func that requires multiprocessing.
    Pandas multiprocessor Pool contains a min 1 threads, so default is 2 threads.
    
    ptSl is upper and lower limit of spot price. 
    Hence to apply additional multipler effect provide a positive value more than 1
    
    params: data => close price series
    params: events => Datetime series sys_cusum filter always
    params: trgt => target from sample pct change
    params: min_req => minimal requirement (Recommend: transaction cost as a percentage)
    last 4 params: contains default values
                    optional params: num_threads => integer this is for multiprocessing per core
                    optional params: ptSl => list(), [] width of profit taking and stop loss
                    optional params: t1 => pd.DataFrame for vertical_bar()
                    optional params: side => pd.Series() side column must be setup based on primary model
    
    '''
    if isinstance(data, (str, float, int, dict, tuple)):
        raise ValueError('Data must be numpy ndarray or pandas series i.e. close price series')
        
    if isinstance(events, (str, float, int, dict, tuple)):
        raise ValueError('Data must be pandas DatetimeIndex i.e. pd.DatetimeIndex series')
    elif events.shape[0] != data.index.shape[0]:
        warnings.warn('Data and events index shape must be same, reindex data to fit events')
    else:
        isinstance(events, datetime.datetime)
        
    if isinstance(trgt, (str, float, int)):
        raise ValueError('Data must be numpy ndarray or pandas series i.e. sample data percentage change series')
    
    # Optional params test
    if isinstance(num_threads, (str, float, list, dict, tuple, np.ndarray, pd.Series)) or (num_threads < 0):
        raise ValueError('num_threads must be non-zero postive integer i.e. 2')
        
    if isinstance(ptSl, (str, float, int)):
        raise ValueError('Data must be numpy ndarray or list i.e. [1,1]')
    elif ptSl[0] == np.nan or ptSl[1] == np.nan:
        raise ValueError('Data must be numpy 1darray shape(1,2) i.e. [1,1]')
    elif ptSl[0] <= 0 or ptSl[1] <= 0:
        # test case for irrational users
        raise ValueError('Data must be numpy 1darray shape(1,2) with values more than 0 i.e. [1,1]')

    if t1.isnull().values.any() or isinstance(t1, (str, float, int, list, dict, tuple)):
        raise ValueError('t1 must be pd.Series with datetime index, pls use vertical_bar func provided.')
                  
    if side.isnull().values.any() or isinstance(side, (str, float, int, list, dict, tuple)):
        raise ValueError('side must be pd.Series based on primary model prediction.')

    data = pd.DataFrame(index = events).assign(data = data).squeeze()
    trgt = trgt.reindex(events)
    trgt = trgt[trgt > min_req]
    if t1 is False:
        t1 = pd.Series(pd.NaT, index = events)
    if side is None:
        side_, side, ptSl_ = 0, pd.Series(1., index = trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, side, ptSl_ = 1, side.reindex(trgt.index), ptSl[:2]
    events=pd.concat({'t1':t1, 'trgt':trgt, 'side':side}, axis = 1).dropna(subset=['trgt'])
    df0=mp_pandas_obj(func = _pt_sl_t1, 
                      pd_obj=('molecule', events.index), 
                      num_threads = num_threads,
                      data = data,
                      events = events,
                      ptSl = ptSl_)
    events['t1'] = df0.dropna(how = 'all').min(axis = 1) # pd.min ignore NaNs
    if side_ == 0:
        events = events.drop('side', axis = 1)
    return events


# =======================================================

def vert_bar(data: pd.Series, events:pd.DatetimeIndex, period: str = 'days', freq: int = 1):
    '''
    AFML pg 49 modified snippet 3.4
    This is not the original snippet, there is some slight change to period
    
    params: data => close price
    params: period => weeks, days, hours, mins
    params: freq => frequency i.e. 1
    
    Vertical barrier will be events_'t1', which will be part of the final func
    where based on filtered criteria (i.e. sys_csf) will generate a period using vert_bar based on freq and period input
    the column is t1 is to maintain consistance so that we can continue to use in conjunction with other modules i.e. co_events
    
    This func does not include holiday and weekend exclusion.
    Strongly encourage you to ensure your series does not include non-trading days. 
    As well as after trading hours, to prevent OMO.
    
    Otherwise you may end up having non-trading days as an as exit event where it could exceed initial parameter.
    i.e. trigger event on Fri and exit on Monday when vertical barrier only 1 day.
    
    Another note:
    If you are using information driven bars or any series that are not in chronological sequence.
    This func will automatically choose the nearest date time index which may result in more than intended period.
    '''
    if isinstance(period, (pd.Series, np.ndarray, list, int, float)):
        raise ValueError('Period should be string i.e. days')
    elif period != 'days':
        warnings.warn('Recommend using days for simplicity')
    
    if isinstance(freq, (pd.Series, np.ndarray, list, str, float )):
        raise ValueError('Frequency must be in integer, other dtypes not accepted i.e. float, numpy.ndarray')
    elif freq <= 0:
        raise ValueError('Frequency must be in positive integer')
        
    _period = str(freq) + period
    t1 = data.index.searchsorted(events + pd.Timedelta(_period))
    t1 = t1[t1 < data.shape[0]]
    t1 = pd.Series(data.index[t1], index = events[:t1.shape[0]])
    return t1

# =======================================================
# Labeling for side and size [3.5, 3.8]
    
def drop_label(events: pd.Series, min_pct: float = .05):
    # apply weights, drop labels with insufficient examples
    '''
    Drop labels with insufficient example
    During training stage only labels deem somewhat reliable will be used
    
    Normalized outcome as a measure, to maintain shape of data sample.
    Default value is 0.05, any extreme value at [-0.95, 0.95] will be dropped as rare occurance.
    
    params: events => DataFrame from labels func
    params: min_pct => float value as a meansure based on normalization.
    '''
    if isinstance(min_pct,(list, np.ndarray, pd.Series)) or min_pct <= 0:
        raise ValueError('min_pct must be positive float i.e. 0.05')
    elif min_pct > 1:
        raise ValueError('min_pct must be within range(0,1) i.e. 0.05')
        
    if isinstance(events, (float, int, str)):
        raise ValueError('events must be pd.DataFrame, kindly use label func provided')
    
    while True:
        df0=events['bin'].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        try:
            events=events[events['bin']!=df0.argmin()]
            print('dropped label: ', df0.argmin(),df0.min())
        except:
            events=events[events['bin']!=df0.idxmin()]
            print('dropped label: ', df0.idxmin(),df0.min())
    return events

def label(data: pd.Series, events: pd.DataFrame):
    '''
    AFML page 51 snippet 3.7
    Basically this func will return meta-labels or price-labels, which is dependent on 'side' if exist.
    
    If meta-label used it return boolean values (0,1) where 0 means not profitable or vertical barrier was hit first
    and 1 would mean profitable. 
    
    This method will complement a primary model with trading rules, and act as a secondary model.
    
    Note:
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling/ recommended)
    
    params: data => close price which will be used to compare if targets were profitable
    params: events = > DataFrame from triple barrier func
                        -events.index is event's starttime
                        -events['t1'] is event's endtime
                        -events['trgt'] is event's target
                        -events['side'] (optional) implies the algo's position side
    
    '''
    if isinstance(data, (str, float, int)):
        raise ValueError('Data must be numpy ndarray or pandas series i.e. close price series')
    if isinstance(events, (str, float, int, list)):
        raise ValueError('Data must be pd.DataFrame, this function is used after triple barrier function i.e. close price series')
    
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px=data.reindex(px, method='bfill')
    
    out = pd.DataFrame(index = events_.index)
    out['ret'] = px.loc[events_['t1'].values].values/ px.loc[events_.index] - 1
    if 'side' in events_:
        out['ret']*=events_['side'] #meta-labeling
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0
    return out