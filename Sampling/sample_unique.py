'''
Take note:
Everything here as much as possible is based on Pandas and multiprocessing for parallelisation.

In part, I want to keep the code as "pandas" as possible.
As well as a way to force myself to only rely on multiprocessing to perform parallel computering.
Try to see it as a challenge.

However, it is still daunting even if the codes where "optimized". Hence extra caution is advised when using any code.

This is especially true for func with MC_seq_bts, MT_MC which are random sample based on Monte-carlos.
It will take up to 6 hours if you are lucky and up to 7 days if you maximise all parameter input.
That is of course meaning your computer do not sleep/ shutdown/ crash.
'''

import pandas as pd
import numpy as np
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
    #=========================================================================
    #_t1 = pd.DatetimeIndex([t1.index[0],t1.max()])
    #iloc = dataIndex.get_loc(_t1[0]) + 1
    iloc=dataIndex.searchsorted(pd.DatetimeIndex([t1.index[0],t1.max()]))
    #=========================================================================
    count=pd.Series(0,index=dataIndex[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.items():
        count.loc[tIn:tOut]+=1.
    return count.loc[molecule[0]:t1[molecule].max()]

def num_co_events(data: pd.Series, events: pd.DataFrame, num_threads: int):
    '''
    This func only calculate num of concurrent event
    
    params: data => ORIGINAL close price series from imported data src
    params: events => this is used in conjunction to vert_bar function or tri_bar.
                      require pandas series with datetime value to check concurrent samples.
    optional
    params: num_threads => number of process to spawn for multiprocessing, default is 1. Pls do not change.
    
    t1 is the end of period dor vert_bar func, 
    where t0 which is the beginning period based on filter criteria will become vert_bar index.
    '''
        
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

    num_co_event =mp_pandas_obj(func = _num_co_events, 
                      pd_obj=('molecule', events.index), 
                      num_threads = num_threads,
                      dataIndex = data.index,
                      t1 = events['t1'])
    
    num_co_event = num_co_event.loc[~num_co_event.index.duplicated(keep='last')]
    num_co_event = num_co_event.reindex(data.index).fillna(0)
    
    return num_co_event

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
    wght=pd.Series(index = molecule) # NaNs no val index only
    for tIn,tOut in t1.loc[wght.index].items():
        wght.loc[tIn]=(1./ num_conc_events.loc[tIn:tOut]).mean()
    
    return wght

def wght_by_coevents(data: pd.Series, events: pd.DataFrame, num_threads: int):
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
    num_co_event=mp_pandas_obj(func = _num_co_events, 
                      pd_obj=('molecule', events.index), 
                      num_threads = num_threads,
                      dataIndex = data.index,
                      t1 = events['t1'])
    
    num_co_event = num_co_event.loc[~num_co_event.index.duplicated(keep='last')]
    num_co_event = num_co_event.reindex(data.index).fillna(0)
    
    out['tW'] = mp_pandas_obj(func = _mp_sample_TW, 
                              pd_obj = ('molecule', events.index),
                              num_threads = num_threads,
                              t1=events['t1'],
                              num_conc_events=num_co_event)
    return out

# =======================================================
# Sequential Bootstrap [4.5.2]
## Build Indicator Matrix [4.3]

def _mp_idx_matrix(data: pd.DataFrame, molecule):
    '''
    this func is permenantly in debug mode.
    otherwise, code will not run. mp_pandas_obj does not allow int as index. Hence transpose not possible.
    
    params: molecule => events.t1 refer to idx_matrix func
    params: data => close price series
    '''
    event_ = data[molecule.index.min(): molecule.max()].index #in the book they included 1 more date index as max not sure why
    indM_ = pd.DataFrame(0., index = event_, columns= np.arange(molecule.shape[0]))
    for i,(t0,t1) in enumerate(molecule.itertuples()):
        indM_.loc[t0:t1,i] = 1.
    
    return indM_

def mp_idx_matrix(data: pd.Series, events: pd.DataFrame, num_threads: int = 1):
    '''
    Calculates the number of times events sample overlaps each other.
    This is a modified func, based on the initial func AFML pg 63 snippet 4.3
    
    This func is equipped with mulitprocessing.
    Do NOT change number of threads.
    Speed improvement up to 702 times over old func.
    
    logic is still based on initial func stated in AFML pg 63
    
    params: data => close price series with datetime index
    params: events => pandas DataFrame generated from tri_bar func
    
    Attempted to include mp_pandas_obj, only use 1 as num_threads. Still trying to debug for more threads.
    
    %timeit result:
    2.55 s ± 40.5 ms per loop (mean ± std. dev. of 3 runs, 3 loops each)
    2.57 s ± 63.6 ms per loop (mean ± std. dev. of 5 runs, 5 loops each)
    '''
    
        
    data.dropna(inplace=True) #create new index based on t1 events
    indM = mp_pandas_obj(func = _mp_idx_matrix, 
                              pd_obj = ('molecule', events.t1),
                              num_threads = num_threads,
                              axis = 1,
                              data = data)
    
    idxM = indM[indM.sum(axis = 1) != 0]
    return idxM


def idx_matrix(data: pd.Series, events: pd.DataFrame):
    '''
    AFML pg 63 snippet 4.3
    Calculates the number of times events sample overlaps each other
    Some simple moification included
    
    logic is still based on initial func stated in AFML pg 63
    
    params: data => close price series with datetime index
    params: events => pandas DataFrame generated from tri_bar func
    
    Attempted to include mp_pandas_obj
    This version of idx_matrix will take up to 30 minutes when tested against (DataFrame.shape(4025, 1840))
    '''
        
    warnings.warn("Kindly use the multiprocess version func provided i.e. mp_idx_matrix", DeprecationWarning, 2)
    
    data = data.dropna() #create new index based on t1 events
    dataIdx = data[events.index.min(): events.t1.max()].index # we only need full dates from events
    indM_ = pd.DataFrame(0, index = dataIdx, columns=np.arange(events.t1.shape[0]))
    for i,(t0,t1) in enumerate(events.t1.items()):
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
    idx_sum = idxM.sum(axis = 1) # concurrency label
    uniqueness = idxM.div(idx_sum, axis = 0) # uniqueness
    avgU = uniqueness[uniqueness > 0].mean() # avg. uniqueness
    return avgU
# =======================================================
# return sample from sequential bootstrap [4.5]
def _mp_seq_bts(avgU: pd.Series, ind: int, phi: list, idxM: pd.DataFrame, sample_len: int):
    '''
    AFML pg 65 snippet 4.5
    This will make use of both av_unique and idx_matrix function to perform sequential bootstrap method.
    The current func includes both func for ease of use.
    
    
    params: data => index matrix generated by func idx_matrix
    params: events => t1 datetime from triple barrier method func tri_bar
    params: sample_len => if only you want to do sub sample you may wish to reduce or increase to improve on uniqueness
    '''
    indM_ = idxM[phi + [ind]] # reduce indM
    avgU.loc[ind] = av_unique(indM_).iloc[-1] #get the last value, if it is very unique it's value will be high
    return avgU


def mp_seq_bts(idxM: pd.DataFrame, sample_len = None, num_threads = 1, 
               verbose: bool = False, random_state = np.random.RandomState()):
    '''
    AFML pg 65 snippet 4.5
    This will make use of both av_unique and idx_matrix function to perform sequential bootstrap method.
    The current func includes both func for ease of use.
    
    Improvement in speed over normal func(up to 50%). Will review later, refactor into class form and dask
    
    If you try to increase num_thread to more than 1, it will become slower. I will try to fix this at a later stage.
    
    params: data => index matrix generated by func idx_matrix
    params: events => t1 datetime from triple barrier method func tri_bar
    params: sample_len => if only you want to do sub sample you may wish to reduce or increase to improve on uniqueness
    
    
    %timeit results:
    # mp sample len 5 num thread = 1: 34.3 s ± 751 ms per loop (mean ± std. dev. of 3 runs, 3 loops each)
    # mp sample len 8 num thread = 1: 56.8 s ± 361 ms per loop (mean ± std. dev. of 3 runs, 3 loops each)
    # mp sample len 15 num thread = 1: 1min 54s ± 1.47 s per loop (mean ± std. dev. of 3 runs, 3 loops each)
    # mp sample len 15 num thread = 7: wtf, crashed.
    
    '''
    
    # Generate a sample via sequential bootstrap
    if random_state is None: 
        random_state = np.random.RandomState()
        
    if sample_len is None:
        sample_len=idxM.shape[1]
    phi = []
    while len(phi) < sample_len:
        avgU=pd.Series(0., dtype=float)
        for ind in idxM:
            jobs = []
            job = {'func': _mp_seq_bts, 
                   'avgU': avgU, 
                   'idxM': idxM,
                   'phi': phi,
                   'ind': ind,
                   'sample_len': sample_len}
            jobs.append(job) #spawn per loop
            if num_threads == 1:
                avgU_ = process_jobs_(jobs)
            else:
                avgU_ = process_jobs(jobs, num_threads = num_threads)
        avgU = pd.Series(avgU_).sum()
        prob = avgU/ avgU.sum() # as long as the prob[0] is increasing, you are on the right track
        if verbose is not False: 
            p("Probabilities:\n Start: {0:.6f} End: {1:.6f}".format(prob[0], prob[-1])) # Mostly for debug
        phi+=[random_state.choice(idxM.columns, p=prob)]
    return phi

#========================================================================
def seq_bts(idxM: pd.DataFrame, sample_len = None):
    '''
    AFML pg 65 snippet 4.5
    This will make use of both av_unique and idx_matrix function to perform sequential bootstrap method.
    The current func includes both func for ease of use.
    
    In a nutshell, if the current loop aka [i], is not very unique. this effect will bring forward to other rows.
    
    This is especially true is concurrency is prevailing, it will suffer a very low weight since it will continueously be discounted.
    This will again be reflected when we try to have out how unique each events is which in this func's case is the prob.
    
    If the previous choice did suffered from weight reduction but the next pick did not, the next pick will always have a higher probability.
    Since it is 'pure' of concurrency.
    
    Bear in mind the variable "prob" is not an integer but rather a list of float which reflects the logic within AFML example 4.5.3
    Which will tell np.random.choice to assign to labels withi higher probablility.
    
    params: data => index matrix generated by func idx_matrix
    params: events => t1 datetime from triple barrier method func tri_bar
    params: sample_len => if only you want to do sub sample you may wish to reduce or increase to improve on uniqueness
    
    %timeit results:
    # normal sample len 5 : 57.6 s ± 921 ms per loop (mean ± std. dev. of 3 runs, 3 loops each)
    # normal sample len 8 : 1min 48s ± 1.05 s per loop (mean ± std. dev. of 3 runs, 3 loops each)
    # normal sample len 15: 3min 35s ± 5.86 s per loop (mean ± std. dev. of 3 runs, 3 loops each)
    # normal sample len 25: 3min 19s ± 2.58 s per loop (mean ± std. dev. of 3 runs, 3 loops each)
    
    '''
    if isinstance(idxM, (str, list, float, dict, tuple, np.ndarray, pd.Series)):
        raise ValueError('idxM value must be pandas DataFrame, pls use idx_matrix func provided')
    elif idxM.isnull().values.any():
        raise ValueError('idxM value contains NaNs, np.isinf')

    warnings.warn("Kindly use the multiprocess version func provided i.e. mp_seq_bts", DeprecationWarning, 2)
    # Generate a sample via sequential bootstrap
    if sample_len is None:
        sample_len=idxM.shape[1]
    phi=[]
    
    while len(phi) < sample_len:
        avgU=pd.Series(0., dtype=float)
        for i in idxM:
            indM_ = idxM[phi + [i]] # reduce indM
            avgU.loc[i] = av_unique(indM_).iloc[-1] #get the last value, if it is very unique it's value will be high
        prob = avgU/ avgU.sum() # this part lets you know the uniqueness of the entire series based on it's pick, hence assign prob
        phi.append(np.random.choice(idxM.columns, p=prob))
    return phi


#===================================================
    
def _mp_MC_seq_bts(data: pd.DataFrame, events: pd.DataFrame, sample_len: int = None):
    '''
    This is a modified func based on AFML pg 67 snippet 4.9
    
    A similar monte carlos simulation will be used to perform sequential bootstrap
    
    '''
    idxM = mp_idx_matrix(data = data, events = events)
    phi = np.random.choice(idxM.columns, size = idxM.shape[1])
    stdU = av_unique(idxM[phi]).mean()
    phi = mp_seq_bts(idxM = idxM, sample_len = None)
    seqU = av_unique(idxM[phi]).mean()
    p('standard:{0}, sequential:{1}'.format(stdU, seqU))
    return {'stdU':stdU, 'seqU':seqU}

def MC_seq_bts(data: pd.DataFrame, events: pd.DataFrame, sample_len: int = None, num_iterations: int = 100, num_threads: int = 1):
    '''
    This is a modified func based on AFML pg 67 snippet 4.9
    A similar monte carlos simulation will be used to perform sequential bootstrap.
    
    this was the supposed algorithm to be used in decision tree bootstrap. But without HPC (Super computer), it will take days to run.
    
    If you are running on 4 cores like me, this func is not suitable for you. Use multiprocessing bootstrap instead.
    
    params: data => close price series
    params: events => tri_barrier func
    params: sample_len => total num of column if = None was set, this is actually the number of events derived in tri_barrier
    params: num_iterations => num of loops to run
    params: num_threads => multiprocessing
    '''
    jobs = []
    for i in np.arange(int(num_iterations)):
        job = {'func': _mp_MC_seq_bts, 
               'data': data, 
               'events': events, 
               'sample_len': sample_len}
        jobs.append(job)
        if num_threads == 1:
            out_ = process_jobs_(jobs)
        else:
            out_ = process_jobs(jobs, num_threads = num_threads)
    out = pd.DataFrame(out_)
    print(out.describe())
    return out

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
    bar = range(t1.max()) #if you keep +1 error pops up
    indM = mp_idx_matrix(data = bar, events = t1) #need to change to fit current algo
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = av_unique(indM[phi]).mean()
    phi = mp_seq_bts(indM)
    seqU = av_unique(indM[phi]).mean()
    return {'stdU':stdU, 'seqU':seqU}

def MT_MC(num_obs: int = 10, num_bars: int = 100, max_H: int = 5, numIters: int = 1e6, num_threads: int = 24):
    '''
    AFML pg 66 - 68 snippet 4.9
    This is the monte carlo experiment using multiprocessing.
    the idea is to generate a series of random numbers, the logic is same as monte carlos using nested loops
    to generate different paths.
    
    This is an important concept which allows you to understand how to use mp to parallelise your code based using process_jobs
    Try to spend some time understanding the concept.
    
    In short, this algo is trying to build multiple montecarlos with mulitple random paths.
    
    params: num_obs => number of observations similar to number of path i.e. how many series to generate
    params: num_bars => number of bars which is also used to mimic timesteps i.e. how long does the series goes
    params: max_H => maximum number for change in value similar to standard deviation i.e. how much changes to each time step
    params: num_Iters => number of loops by default 100,000 i.e. the number of times to create a monte carlos simulation
    params: num_threads => multiprocessing used for cores, processes.
    
    Still trying to fix
    '''
    warnings.warn('This func does not fit into other aglos, do not use it, it was created for easier understanding for mp.process_jobs!')
    
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


def _wgth_by_rtn(data: pd.DataFrame, events: pd.Series, num_co_events: pd.DataFrame, molecule):
    rtn = np.log(data).diff().fillna(0) # Log-returns, so that they are additive
    wghts = pd.Series(index=molecule) #index is wrong need to get molecule.index.min() to molecule.max()

    for tIn, tOut in events.loc[wghts.index].items():
        # Weights depend on returns and label concurrency
        wghts.loc[tIn] = (rtn.close.loc[tIn:tOut].div(num_co_events.loc[tIn:tOut]).fillna(0)).sum()
    return wghts.abs() #under the assumption that the whole data was generated by tri_bar


def wght_by_rtn(data: pd.DataFrame, events: pd.DataFrame, num_threads: int = 1):
    '''
    AFML page 69 snippet 4.10 
    weights based on abs returns.
    Those with higher abs return should be given less weights according to the book.
    While the algo itself is a func of log price absolute return.
    
    I believe it may be required to drop rare labels, before this step was taken.

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
                                  num_threads = num_threads,
                                  dataIndex = data.index, 
                                  t1 = events['t1'])
    
    out = pd.DataFrame() #create dataframe to hold this, can perform assign at a later stage
    
    out['w'] = mp_pandas_obj(func = _wgth_by_rtn, 
                          pd_obj = ('molecule', events.index),
                          num_threads = num_threads,
                          data = data,
                          events = events['t1'], 
                          num_co_events = num_co_events)
    
    out['w'] *= out.shape[0] / out['w'].sum() #I did not clear weights above 1, therefore you may encounter over weights.
    return out


def wght_by_td(data: pd.DataFrame, events: pd.DataFrame, num_threads: int = 1, td: float = 1.):
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
        
    if isinstance(td, (str, list, dict, tuple, np.ndarray, pd.Series)):
        raise ValueError('events input must be float or integer val i.e. 0.8, 1.0')
    if isinstance(td, (float, int)) and td < -1:
        raise ValueError('events input must be between (-1,1) i.e. 0.8, 1.0')
    if isinstance(td, (float, int)) and td > 1:
        raise ValueError('events input must be within range(0,1) i.e. 0.8, 1.0')
    if isinstance(td, (int)) and td >= 0:
        td = float(td)
        
    av_un = wght_by_coevents(data = data.close, 
                             events = events, 
                             num_threads = num_threads)
    
    wghts = av_un['tW'].sort_index().cumsum()
    if td >= 0:
        slope = (1 - td) / wghts.iloc[-1]
    else:
        slope = 1 / ((td + 1) * wghts.iloc[-1])
    const = 1 - slope * wghts.iloc[-1]
    wghts = const + slope * wghts
    wghts[wghts < 0] = 0  # clear neg weights
    return wghts
