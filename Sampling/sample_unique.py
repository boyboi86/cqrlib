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
    this func is permenantly in debug mode.
    otherwise, code will not run. mp_pandas_obj does not allow int as index. Hence transpose not possible.
    
    params: molecule => events.t1 refer to idx_matrix func
    params: data => close price series
    '''

    event_ = data[molecule.index.min(): molecule.max()].index
    
    indM_ = pd.DataFrame(0., index = event_, columns= np.arange(molecule.shape[0]))
    for i,(t0,t1) in enumerate(molecule.iteritems()):
        indM_.loc[t0:t1,i] = 1.
    
    return indM_

def idx_matrix(data: pd.Series, events: pd.DataFrame):
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
            
    if isinstance(data, (str, int, float, list, dict, tuple, np.ndarray)):
        raise ValueError('data input  must be pandas Series with datetime Index')
        
    if isinstance(events, (pd.Series, pd.DataFrame)):        
        if isinstance(events.t1, (str, list, dict, tuple, np.ndarray)):
            raise ValueError('events.t1 value must be date time')
            
    if isinstance(events, (str, int, float, list, dict, tuple, np.ndarray)):
        raise ValueError('events input must be pandas DataFrame with datetime Index, pls use tri_bar func provided')
        
    data.dropna(inplace=True) #create new index based on t1 events
    #dataIdx = data[events.index.min(): events.t1.max()].index # we only need full dates from events
    #indM_ = pd.DataFrame(0, index = dataIdx, columns=np.arange(events.t1.shape[0])).copy()
    indM = mp_pandas_obj(func = _idx_matrix, 
                              pd_obj = ('molecule', events.t1),
                              num_threads = 1,
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
    indM = idx_matrix(data = bar, events = t1, num_threads = 1) #need to change to fit current algo
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
    
    Still trying to fix
    '''
    warnings.warn('This func does not fit into other aglos, do not use it!')
    
    jobs = []
    for i in np.arange(int(numIters)):
        job = {'func': auxMC, 
               'num_obs': num_obs, 
               'num_bars': num_bars, 
               'max_H': max_H}
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

def _wgth_by_rtn(data: pd.DataFrame, events: pd.Series, num_co_events: pd.DataFrame, molecule):
    rtn = np.log(data).diff()  # Log-returns, so that they are additive
    wghts = pd.Series(index=molecule)

    for tIn, tOut in events.loc[wghts.index].iteritems():
        # Weights depend on returns and label concurrency
        wghts.loc[tIn] = (rtn.loc[tIn:tOut] / num_co_events.loc[tIn:tOut]).sum()
    return wghts.abs()


def wght_by_rtn(data: pd.DataFrame, events: pd.DataFrame, num_threads: int = 1):
    '''
    AFML page 69 snippet 4.10 
    weights based on abs returns.
    Those with higher abs return should be given more weights

    param: data => close price series 
    param: events =>  pandas DataFrame using tri_bar func
    param num:_threads => the number of threads concurrently used by the function, depends on len(molecule)
    '''
    if isinstance(data, (str, list, float, dict, tuple, np.ndarray)):
        raise ValueError('idxM input must be pandas Series or DataFrame with single column i.e. close prce series')
    elif data.isnull().values.any():
        raise ValueError('idxM input contains NaNs, np.isinf')
            
    if isinstance(events, (str, list, float, dict, tuple, np.ndarray, pd.Series)):
        raise ValueError('events input must be pandas  DataFrame, pls use tri_bar func provided')
    elif events.isnull().values.any():
        raise ValueError('events input contains NaNs, np.isinf')
        
    if isinstance(num_threads, (str, float, list, float, dict, tuple, np.ndarray, pd.Series)):
        raise ValueError('events input must be positive non-zero integer i.e. 1, 2')

    num_co_events = mp_pandas_obj(_num_co_events, 
                                 ('molecule', events.index), 
                                  num_threads,
                                  dataIndex=data.index, 
                                  events=events['t1'])
    num_co_events = num_co_events.loc[~num_co_events.index.duplicated(keep='last')]
    num_co_events = num_co_events.reindex(data.index).fillna(0)
    wghts = mp_pandas_obj(_wgth_by_rtn, 
                          ('molecule', events.index),
                          num_threads = num_threads,
                          events = events['t1'], 
                          num_co_events = num_co_events,
                          data = data)
    
    wghts *= wghts.shape[0] / wghts.sum()
    return wghts


def wght_by_td(data: pd.Series, events: pd.DataFrame, num_threads: int = 1, td: float = 1.):
    '''
    AFML page 70 Snippet 4.11
    
    weights based on time decay.
    As time passes, older samples will suffer larger discount on their "relevance".

    param: data => close prices series
    param: events => pandas DataFrame from tri_bar func
    param: num_threads => the number of threads concurrently used by the function, depends on len(molecule)
    param: td => decay factor
        - decay = 1 means there is no time decay
        - 0 < decay < 1 means that weights decay linearly over time, but every observation still receives a strictly positive weight, regadless of how old
        - decay = 0 means that weights converge linearly to zero, as they become older
        - decay < 0 means that the oldes portion c of the observations receive zero weight (i.e they are erased from memory)
    '''
    if isinstance(data, (str, list, float, dict, tuple, np.ndarray)):
        raise ValueError('idxM input must be pandas Series or DataFrame with single column i.e. close prce series')
    elif data.isnull().values.any():
        raise ValueError('idxM input contains NaNs, np.isinf')
            
    if isinstance(events, (str, list, float, dict, tuple, np.ndarray, pd.Series)):
        raise ValueError('events input must be pandas  DataFrame, pls use tri_bar func provided')
    elif events.isnull().values.any():
        raise ValueError('events input contains NaNs, np.isinf')
        
    if isinstance(num_threads, (str, float, list, float, dict, tuple, np.ndarray, pd.Series)):
        raise ValueError('events input must be positive non-zero integer i.e. 1, 2')
        
    if isinstance(td, (str, list, float, dict, tuple, np.ndarray, pd.Series)):
        raise ValueError('events input must be positive non-zero float i.e. 0.8, 1.0')
    if isinstance(td, (float, int)) and td < 0:
        raise ValueError('events input must be positive non-zero value i.e. 0.8, 1.0')
    if isinstance(td, (float, int)) and td > 1:
        raise ValueError('events input must be within range(0,1) i.e. 0.8, 1.0')
    if isinstance(td, (int)) and td >= 0:
        td = float(td)
        
    av_un = co_events(data = data, events = events, num_threads = num_threads)
    wghts = av_un['tW'].sort_index().cumsum()
    if td >= 0:
        slope = (1 - td) / wghts.iloc[-1]
    else:
        slope = 1 / ((td + 1) * wghts.iloc[-1])
    const = 1 - slope * wghts.iloc[-1]
    wghts = const + slope * wghts
    wghts[wghts < 0] = 0  # Weights can't be negative
    return wghts
