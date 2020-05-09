# -*- coding: utf-8 -*-
"""
Pls import myPandasObj from Util for multi-threading process
"""

import pandas as pd
import numpy as np

from mlfinlab.sampling.concurrent import num_concurrent_events, get_av_uniqueness_from_triple_barrier
from mlfinlab.util.multiprocess import mp_pandas_obj

p = print

# Estimating uniqueness of a label [4.1]
def mpNumCoEvents(closeIdx,t1,molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[modelcule].max() impacts the count.
    '''
    #1) find events that span the period [molecule[0],molecule[-1]]
    t1=t1.fillna(closeIdx[-1]) # unclosed events still must impact other weights
    t1=t1[t1>=molecule[0]] # events that end at or after molecule[0]
    t1=t1.loc[:t1[molecule].max()] # events that start at or before t1[molecule].max()
    #2) count events spanning a bar
    iloc=closeIdx.searchsorted(np.array([t1.index[0],t1.max()]))
    count=pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.iteritems():count.loc[tIn:tOut]+=1.
    return count.loc[molecule[0]:t1[molecule].max()]
# =======================================================
# Estimating the average uniqueness of a label [4.2]
def mpSampleTW(t1,numCoEvents,molecule):
    # Derive avg. uniqueness over the events lifespan
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(1./numCoEvents.loc[tIn:tOut]).mean()
    return wght
# =======================================================
# Sequential Bootstrap [4.5.2]
## Build Indicator Matrix [4.3]
def getIndMatrix(barIx,t1):
    # Get Indicator matrix
    indM=(pd.DataFrame(0,index=barIx,columns=range(t1.shape[0])))
    for i,(t0,t1) in enumerate(t1.iteritems()):indM.loc[t0:t1,i]=1.
    return indM
# =======================================================
# Compute average uniqueness [4.4]
def getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix
    c=indM.sum(axis=1) # concurrency
    u=indM.div(c,axis=0) # uniqueness
    avgU=u[u>0].mean() # avg. uniqueness
    return avgU
# =======================================================
# return sample from sequential bootstrap [4.5]
def seqBootstrap(indM,sLength=None):
    # Generate a sample via sequential bootstrap
    if sLength is None:sLength=indM.shape[1]
    phi=[]
    while len(phi)<sLength:
        avgU=pd.Series()
        for i in indM:
            indM_=indM[phi+[i]] # reduce indM
            avgU.loc[i]=getAvgUniqueness(indM_).iloc[-1]
        prob=avgU/avgU.sum() # draw prob
        phi+=[np.random.choice(indM.columns,p=prob)]
    return phi
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


def _apply_weight_by_return(label_endtime, num_conc_events, close_series, molecule):
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


def get_weights_by_return(triple_barrier_events, close_series, num_threads=5):
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
            weights = mp_pandas_obj(_apply_weight_by_return, ('molecule', triple_barrier_events.index), num_threads,
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
            av_uniqueness = get_av_uniqueness_from_triple_barrier(triple_barrier_events, close_series, num_threads)
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