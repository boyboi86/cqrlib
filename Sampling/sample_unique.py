import pandas as pd
import numpy as np
import datetime
import warnings

from research.Util.multiprocess import mp_pandas_obj, process_jobs_, process_jobs

p = print

# Estimating uniqueness of a label [4.1]
def _num_co_events(dataIndex: pd.DatetimeIndex, t1: pd.Series, molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[modelcule].max() impacts the count.
    
    params: dataIndex => datetime Index based on co_event data input
    params: t1 => vert_bar t1, use in conjunction with tri_bar to generate t1
    params: molecule => multiprocessing
    '''
    #1) find events that span the period [molecule[0],molecule[-1]]
    t1=t1.fillna(dataIndex[-1]) # unclosed events still must impact other weights
    t1=t1[t1>=molecule[0]] # events that end at or after molecule[0]
    t1=t1.loc[:t1[molecule].max()] # events that start at or before t1[molecule].max()
    #2) count events spanning a bar
    iloc=dataIndex.searchsorted(pd.DatetimeIndex([t1.index[0],t1.max()]))
    count=pd.Series(0,index=dataIndex[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.iteritems():
        count.loc[tIn:tOut]+=1.
    return count.loc[molecule[0]:t1[molecule].max()]

# =======================================================
# Estimating the average uniqueness of a label [4.2]
def _mp_sample_TW(t1: pd.DatetimeIndex, num_conc_events: int, molecule):
    '''
    params: t1 => vert_bar end period
    params: num_conc_events => number of cocurrent events generated from _num_co_events,
                               This is an input not a func.
    params: molecule => multiprocess
    '''
    # Derive avg. uniqueness over the events lifespan
    wght=pd.Series(0.0, index = molecule) # NaNs no val index only
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(1./ num_conc_events.loc[tIn:tOut]).mean()
    
    return wght

def co_events(data: pd.Series, events: pd.DataFrame, num_threads: int):
    '''
    params: data => ORIGINAL close price series from imported data src
    params: events => this is used in conjunction to vert_bar function or tri_bar.
                      require pandas series with datetime value to check concurrent samples.
    optional
    params: num_threads => number of process to spawn for multiprocessing, default is 1. Pls do not change.
    
    t1 is the end of period dor vert_bar func, 
    where t0 which is the beginning period based on filter criteria will become vert_bar index.
    '''
    if isinstance(data, pd.Series):   
        if isinstance(data.dtype, (str, dict, tuple, list)):
            raise ValueError('data input must be pandas Series with dtype either float or integer value i.e. close price series')
        if data.isnull().values.any():
            raise ValueError('data series contain isinf, NaNs')
            
    if isinstance(data, pd.DataFrame):
        if isinstance(data.squeeze().dtype, (str, dict, tuple, list)):
            raise ValueError('data input must be pandas Series with dtype either float or integer value i.e. close price series')
        elif not isinstance(data.squeeze(), pd.Series):
            raise ValueError('data input must be pandas Series i.e. close price series')
        else:
            data = data.squeeze()
    else:
        raise ValueError('data input must be pandas Series with dtype either float or integer value i.e. close price series')
        
    if isinstance(events, pd.DataFrame):
        if isinstance(events, (int, str, float, dict, tuple, list)):
            raise ValueError('data input must be pandas DatetimeIndex or datetime values')
        if events.isnull().values.any():
            raise ValueError('events series contain isinf, NaNs, NaTs')
        if events.index.dtype != 'datetime64[ns]' or events['t1'].dtype != 'datetime64[ns]':
            raise ValueError('event["t1"] or devents.index is not datetime value')
    else:
        raise ValueError('data input must be pandas DateFrame please use tri_bar func provided')
    
        
    if isinstance(num_threads, (str, float, dict, tuple, list, pd.Series, np.ndarray)):
        raise ValueError('data input must be integer i.e. 1')

    out = pd.DataFrame()
    df0=mp_pandas_obj(func = _num_co_events, 
                      pd_obj=('molecule', events.index), 
                      num_threads = num_threads,
                      dataIndex = data.index,
                      t1 = events['t1'])
    df0 = df0.loc[~df0.index.duplicated(keep='last')]
    df0 = df0.reindex(df0.index).fillna(0)
    out['tW'] = mp_pandas_obj(func = _mp_sample_TW, 
                              pd_obj = ('molecule', events.index),
                              num_threads = num_threads,
                              t1=events['t1'],
                              num_conc_events=df0)
    return out

# =======================================================
# Sequential Bootstrap [4.5.2]
## Build Indicator Matrix [4.3]

def _idx_matrix(data: pd.DataFrame, molecule):
    '''
    AFML pg 63 snippet 4.3
    Calculates the number of times events sample overlaps each other
    
    logic is still based on initial func stated in AFML pg 63
    
    params: data => close price series with datetime index
    params: events => pandas DataFrame generated from tri_bar func
    
    Attempted to include mp_pandas_obj
    '''
    event_ = data[molecule.index.min(): molecule.max()].index
    
    indM_ = pd.DataFrame(0., index = event_, columns= np.arange(molecule.shape[0]))
    for i,(t0,t1) in enumerate(molecule.iteritems()):
        indM_.loc[t0:t1,i] = 1.
    
    return indM_

def idx_matrix(data: pd.Series, events: pd.DataFrame, num_threads=1):
    '''
    AFML pg 63 snippet 4.3
    Calculates the number of times events sample overlaps each other
    
    logic is still based on initial func stated in AFML pg 63
    
    params: data => close price series with datetime index
    params: events => pandas DataFrame generated from tri_bar func
    
    Attempted to include mp_pandas_obj, only use 1 as num_threads. Still trying to debug for more threads.
    '''
    
    if isinstance(data, (pd.Series, pd.DataFrame)):        
        if data.isnull().values.any():
            raise ValueError('data series contain isinf, NaNs, NaTs')
        elif isinstance(data, (str, list, dict, tuple, np.ndarray, pd.Series)):
            raise ValueError('data series contain non-float or non-integer values')
    else:
        raise ValueError('data input  must be pandas Series with datetime Index')
        
    if isinstance(events, (pd.Series, pd.DataFrame)):        
        if isinstance(events.t1, (str, list, dict, tuple, np.ndarray)):
            raise ValueError('events.t1 value must be date time')
    else:
        raise ValueError('events input must be pandas DataFrame with datetime Index, pls use tri_bar func provided')
        
    data.dropna(inplace=True) #create new index based on t1 events
    #dataIdx = data[events.index.min(): events.t1.max()].index # we only need full dates from events
    #indM_ = pd.DataFrame(0, index = dataIdx, columns=np.arange(events.t1.shape[0])).copy()
    indM = mp_pandas_obj(func = _idx_matrix, 
                              pd_obj = ('molecule', events.t1),
                              num_threads = num_threads,
                              axis = 1, # we are multiprocess based on columns so we need to flip over the axis
                              #indM_ = indM_,
                              data = data)
    
    idxM = indM[indM.sum(axis = 1) != 0]
    return idxM


def old_idx_matrix(data: pd.Series, events: pd.DataFrame):
    '''
    AFML pg 63 snippet 4.3
    Calculates the number of times events sample overlaps each other
    Some simple moification included
    
    logic is still based on initial func stated in AFML pg 63
    
    params: data => close price series with datetime index
    params: events => pandas DataFrame generated from tri_bar func
    
    Attempted to include mp_pandas_obj
    '''
    
    if isinstance(data, (pd.Series, pd.DataFrame)):        
        if data.isnull().values.any():
            raise ValueError('data series contain isinf, NaNs, NaTs')
        elif isinstance(data, (str, list, dict, tuple, np.ndarray, pd.Series)):
            raise ValueError('data series contain non-float or non-integer values')
    else:
        raise ValueError('data input  must be pandas Series with datetime Index')
        
    if isinstance(events, (pd.Series, pd.DataFrame)):        
        if isinstance(events.t1, (str, list, dict, tuple, np.ndarray)):
            raise ValueError('events.t1 value must be date time')
    else:
        raise ValueError('events input must be pandas DataFrame with datetime Index, pls use tri_bar func provided')
        
    warnings.warn("This func is single thread, kindly use the multiprocess version func provided i.e. idx_matrix")
    
    data = data.dropna() #create new index based on t1 events
    dataIdx = data[events.index.min(): events.t1.max()].index # we only need full dates from events
    indM_ = pd.DataFrame(0, index = dataIdx, columns=np.arange(events.t1.shape[0]))
    for i,(t0,t1) in enumerate(events.t1.iteritems()):
        indM_.loc[t0:t1,i] = 1.
    
    idxM = indM_[indM_.sum(axis = 1) != 0]
    return idxM


# =======================================================
# Compute average uniqueness [4.4]
def av_unique(idxM: pd.DataFrame):
    '''
    AFML pg 65 snippet 4.4
    Calculates average uniqueness of sampled data and assigns uniqueness in the form weights
    
    params: data => pandas dataframe input based on Idx_Matrix func which calculates the uniqueness of event samples
    '''
    # Average uniqueness from indicator matrix
    idx_sum = idxM.sum(axis = 1) # concurrency
    u=idxM.div(idx_sum, axis = 0) # uniqueness
    avgU=u[u>0].mean() # avg. uniqueness
    return avgU
# =======================================================
# return sample from sequential bootstrap [4.5]
def seq_bts(idxM: pd.DataFrame, sample_len = None):
    '''
    AFML pg 65 snippet 4.5
    This will make use of both av_unique and idx_matrix function to perform sequential bootstrap method.
    The current func includes both func for ease of use.
    
    
    params: data => index matrix generated by func idx_matrix
    params: events => t1 datetime from triple barrier method func tri_bar
    params: sample_len => if only you want to do sub sample you may wish to reduce or increase to improve on uniqueness
    '''
    

    if isinstance(idxM, (str, list, float, dict, tuple, np.ndarray, pd.Series)):
        raise ValueError('idxM value must be pandas DataFrame, pls use idx_matrix func provided')
    elif idxM.isnull().values.any():
        raise ValueError('idxM value contains NaNs, np.isinf')
            
    if sample_len is not None:
        if isinstance(sample_len, (str, list, float, dict, tuple, np.ndarray, pd.Series)):
            raise ValueError('sLength must be positive non-decimal integer value i.e. 12, 30')
        if sample_len > idxM.shape[1]:
            warnings.warn('sample length exceeds index matrix column, may exceed expected waiting time.')

    
    # Generate a sample via sequential bootstrap
    if sample_len is None:
        sample_len=idxM.shape[1]
    phi=[]
    
    while len(phi) < sample_len:
        avgU=pd.Series(0., dtype=float)
    
        for i in idxM:
            indM_ = idxM[phi + [i]] # reduce indM
            avgU.loc[i] = av_unique(indM_).iloc[-1] #get the last value, if it is very unique it's value will be high
        prob = avgU/ avgU.sum() # draw prob
        p(prob[0])
        phi.append(np.random.choice(idxM.columns, p=prob))
    return phi

#===================================================
    

def rnd_t1(num_obs: int, num_bars: int, max_H: int):
    t1 = pd.Series()
    for i in np.arange(num_obs):
        ix = np.random.randint(0, num_bars)
        val = ix + np.random.randint(1, max_H)
        t1.loc[ix] = val
    return t1.sort_index()

def auxMC(num_obs: int, num_bars: int, max_H: int):
    t1 = rnd_t1(num_obs, num_bars, max_H)
    bar = range(t1.max() + 1)
    indM = idx_matrix(data = bar, events = t1, num_threads = 1)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = av_unique(indM[phi]).mean()
    phi = seq_bts(indM)
    seqU = av_unique(indM[phi]).mean()
    return {'stdU':stdU, 'seqU':seqU}

def MT_MC(num_obs: int = 10, num_bars: int = 100, max_H: int = 5, numIters: int = 1e6, num_threads: int = 24):
    '''
    AFML pg 66 - 68 snippet 4.9
    This is the monte carlo experiment using multiprocessing.
    the idea is to generate a series of random numbers, the logic is same as monte carlos using nested loops
    to generate different paths.
    
    params: num_obs => number of observations similar to number of path
    params: num_bars => number of bars which is also used to mimic timesteps
    params: max_H => maximum number for change in value simialr to standard deviation
    params: num_Iters => number of loops by default 100,000
    params: num_threads => multiprocessing used for cores, processes.
    '''
    jobs = []
    for i in np.arange(int(numIters)):
        job = {'func': auxMC, 
               'num_obs': num_obs, 
               'num_bars': num_bars, 
               'max_H': max_H, 
               'numIters': numIters, 
               'num_threads': num_threads}
        jobs.append(job)
    if num_threads == 1:
        out = process_jobs_(jobs)
    else:
        out = process_jobs(jobs, num_threads = num_threads)
    print(pd.DataFrame(out).describe())
    return pd.DataFrame(out)

'''
# =======================================================
# Determination of sample weight by absolute return attribution [4.10]
def mpSampleW(t1,numCoEvents,close,molecule):
    # Derive sample weight by return attribution
    ret=np.log(close).diff() # log-returns, so that they are additive
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(ret.loc[tIn:tOut]/numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()

# ============================================================
'''
'''
def _wgth_by_rtn(label_endtime, num_conc_events, close_series, molecule):
    """
    Snippet 4.10, page 69, Determination of Sample Weight by Absolute Return Attribution
    Derives sample weights based on concurrency and return. Works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.

    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param num_conc_events: (pd.Series) number of concurrent labels (output from num_concurrent_events function).
    :param close_series: (pd.Series) close prices
    :param molecule: (an array) a set of datetime index values for processing.
    :return: (pd.Series) of sample weights based on number return and concurrency for molecule
    """

    ret = np.log(close_series).diff()  # Log-returns, so that they are additive
    weights = pd.Series(index=molecule)

    for t_in, t_out in label_endtime.loc[weights.index].iteritems():
        # Weights depend on returns and label concurrency
        weights.loc[t_in] = (ret.loc[t_in:t_out] / num_conc_events.loc[t_in:t_out]).sum()
    return weights.abs()


def wght_rtn(triple_barrier_events, close_series, num_threads=5):
    """
    Snippet 4.10(part 2), page 69, Determination of Sample Weight by Absolute Return Attribution
    This function is orchestrator for generating sample weights based on return using mp_pandas_obj.

    :param triple_barrier_events: (data frame) of events from labeling.get_events()
    :param close_series: (pd.Series) close prices
    :param num_threads: (int) the number of threads concurrently used by the function.
    :return: (pd.Series) of sample weights based on number return and concurrency
    """

    null_events = bool(triple_barrier_events.isnull().values.any())
    null_index = bool(triple_barrier_events.index.isnull().any())
    if null_events is False and null_index is False:
        try:
            num_conc_events = mp_pandas_obj(num_concurrent_events, ('molecule', triple_barrier_events.index), num_threads,
                                            close_series_index=close_series.index, label_endtime=triple_barrier_events['t1'])
            num_conc_events = num_conc_events.loc[~num_conc_events.index.duplicated(keep='last')]
            num_conc_events = num_conc_events.reindex(close_series.index).fillna(0)
            weights = mp_pandas_obj(_wgth_by_rtn, ('molecule', triple_barrier_events.index), num_threads,
                                    label_endtime=triple_barrier_events['t1'], num_conc_events=num_conc_events,
                                    close_series=close_series)
            weights *= weights.shape[0] / weights.sum()
            return weights
        except:
            p('NaN values in triple_barrier_events, delete NaNs')


def get_weights_by_time_decay(triple_barrier_events, close_series, num_threads=5, decay=1):
    """
    Snippet 4.11, page 70, Implementation of Time Decay Factors

    :param triple_barrier_events: (data frame) of events from labeling.get_events()
    :param close_series: (pd.Series) close prices
    :param num_threads: (int) the number of threads concurrently used by the function.
    :param decay: (int) decay factor
        - decay = 1 means there is no time decay
        - 0 < decay < 1 means that weights decay linearly over time, but every observation still receives a strictly positive weight, regadless of how old
        - decay = 0 means that weights converge linearly to zero, as they become older
        - decay < 0 means that the oldes portion c of the observations receive zero weight (i.e they are erased from memory)
    :return: (pd.Series) of sample weights based on time decay factors
    """
    null_events = bool(triple_barrier_events.isnull().values.any())
    null_index = bool(triple_barrier_events.index.isnull().any())
    if null_events is False and null_index is False:
        try:
            # Apply piecewise-linear decay to observed uniqueness
            # Newest observation gets weight=1, oldest observation gets weight=decay
            av_uniqueness = av_unique(triple_barrier_events, close_series, num_threads)
            decay_w = av_uniqueness['tW'].sort_index().cumsum()
            if decay >= 0:
                slope = (1 - decay) / decay_w.iloc[-1]
            else:
                slope = 1 / ((decay + 1) * decay_w.iloc[-1])
            const = 1 - slope * decay_w.iloc[-1]
            decay_w = const + slope * decay_w
            decay_w[decay_w < 0] = 0  # Weights can't be negative
            return decay_w
        else:
            p('NaN values in triple_barrier_events, delete NaNs')
'''