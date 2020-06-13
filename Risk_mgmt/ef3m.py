import numpy as np
import pandas as pd
import random
from research.Util.multiprocess import process_jobs_, process_jobs

from scipy.stats import norm
from scipy.special import comb
from numba import njit

# This is the mixture’s class
class M2N:
    
    def __init__(self, mts: list, epsilon: float = 1e-3, variant: int = 0):
        self.mts = mts
        self.param = [0 for i in range(5)]
        self.error = sum([mts[i]**2 for i in range(len(mts))])
        self.epsilon = epsilon
        self.variant = variant
        
    def fit(self, mu2: list):
        p1=random.random()
        numIter = 0
        while True:
            numIter += 1
            if self.variant == 0:
                param = self.iter4(mu2, p1) # for the first variant
            else:
                param = self.iter5(mu2, p1) # for the second variant
            #print(param)
            mts = self.get_mts(param)
            #print(mts)
            error=sum([(self.mts[i]-mts[i])**2 for i in np.arange(len(mts))])
            if error<self.error:
                self.param=param
                self.error=error
            if abs(p1 - param[4]) < self.epsilon: return # threshold limitations
            if numIter > 1/ self.epsilon: return # loop limitlations
            p1=param[4]
            mu2=param[1] # for the 5th moment’s convergence
            

#-------------------------------------------
# Derive the mixture’s mts from its param
    def get_mts(self, param: list):
        m1 = param[4] * param[0] + (1 - param[4]) * param[1]

        m2 = param[4] * (param[2]**2 + param[0]**2) + (1-param[4])* (param[3] **2 + param[1]**2)

        m3 = param[4] * (3 * param[2]**2 * param[0] + param[0]**3) + (1 - param[4]) * (3 * param[3]**2 * param[1] + param[1]**3)

        m4 = param[4] * (3 * param[2]**4 + 6 * param[2]**2 * param[0]**2 + param[0]**4)+(1 - param[4]) * \
        (3 * param[3]**4 + 6 * param[3]**2 * param[1]**2 + param[1]**4)

        m5=param[4] * (15 * param[2]**4 * param[0] + 10 * param[2]**2 * param[0]**3 + param[0]**5) + (1 - param[4]) * \
        (15*param[3]**4 * param[1]+10*param[3]**2*param[1]**3+param[1]**5)

        return [m1, m2, m3, m4, m5]
#-------------------------------------------
# Equations for variant 1
    def iter4(self, mu2: float, p1: float):
        mts = self.mts
        
        mu1 = (mts[0] - (1 - p1) * mu2)/p1

        sig_2 = ((mts[2]+2*p1*mu1**3+(p1 - 1)*mu2**3-3*mu1*(mts[1]+mu2**2*(p1-1)))/(3*(1-p1)*(mu2-mu1)))**(.5)

        sig_1 = ((mts[1] - sig_2**2 - mu2**2) / p1 + sig_2**2 + mu2**2 - mu1**2)**(.5)

        p1 = (mts[3] - 3 * sig_2**4 - 6 * sig_2**2 * mu2**2 - mu2**4)/ \
        (3 * (sig_1**4 - sig_2**4)+ 6 * (sig_1**2 * mu1**2 - sig_2**2 * mu2**2) + mu1**4 - mu2**4)
        return [mu1, mu2, sig_1, sig_2, p1]
    


#-------------------------------------------
# Equations for variant 2

    def iter5(self, mu2: float, p1: float):
        mts = self.mts
        mu1 = (mts[0] - (1 - p1) * mu2) / p1
        sig_2 = ((mts[2] + 2 * p1 * mu1**3 + (p1 - 1) * mu2**3 - 3 * mu1 * (mts[1] + mu2**2 * \
        (p1-1))) / (3 * (1 - p1) * (mu2 - mu1)))**(.5)

        sig_1 = ((mts[1] - sig_2**2 - mu2**2) / p1 + sig_2**2 + mu2**2 - mu1**2)**(.5)
        a = (6 * sig_2**4 + (mts[3] - p1 * (3 * sig_1**4 + 6 * sig_1**2 * mu1**2 + mu1**4))/ (1 - p1))**.5
        mu2 = ( a - 3 * sig_2**2)**.5

        a = 15 * sig_1**4 * mu1 + 10 * sig_1**2 * mu1**3 + mu1**5
        b = 15 * sig_2**4 * mu2 + 10 * sig_2**2 * mu2**3 + mu2**5
        p1 = (mts[4] - b)/(a - b)
        return [mu1, mu2, sig_1, sig_2, p1]


"""
@njit
def _iter4(mu2, p1, mts):
        mts = self.mts
        mu1 = (mts[0] - (1 - p1) * mu2)/p1

        sig_2 = ((mts[2] + 2 * p1 * mu1**3 + (p1 - 1) * mu2**3 - 3 * mu1 * (mts[1] + mu2**2 * (p1-1)))/(3 * (1 - p1) * (mu2 - mu1)))**(.5)

        sig_1 = ((mts[1] - sig_2**2 - mu2**2) / p1 + sig_2**2 + mu2**2 - mu1**2)**(.5)

        p1 = (mts[3] - 3 * sig_2**4 - 6 * sig_2**2 * mu2**2 - mu2**4)/ \
        (3 * (sig_1**4 - sig_2**4)+ 6 * (sig_1**2 * mu1**2 - sig_2**2 * mu2**2) + mu1**4 - mu2**4)

        return [mu1, mu2, sig_1, sig_2, p1]
@njit
def _iter5(mu2, p1, mts):
        mts = self.mts
        mu1 = (mts[0] - (1 - p1) * mu2) / p1
        sig_2 = ((mts[2] + 2 * p1 * mu1**3 + (p1 - 1) * mu2**3 - 3 * mu1 * (mts[1] + mu2**2 * \
        (p1-1))) / (3 * (1 - p1) * (mu2 - mu1)))**(.5)

        sig_1 = ((mts[1] - sig_2**2 - mu2**2) / p1 + sig_2**2 + mu2**2 - mu1**2)**(.5)
        a = (6 * sig_2**4 + (mts[3] - p1 * (3 * sig_1**4 + 6 * sig_1**2 * mu1**2 + mu1**4))/ (1 - p1))**.5
        mu2 = ( a - 3 * sig_2**2)**.5

        a = 15 * sig_1**4 * mu1 + 10 * sig_1**2 * mu1**3 + mu1**5
        b = 15 * sig_2**4 * mu2 + 10 * sig_2**2 * mu2**3 + mu2**5
        p1 = (mts[4] - b)/(a - b)
        return [mu1, mu2, sig_1, sig_2, p1]
"""  
#-------------------------------------------
# Compute mts about the mean (or centered) from mts about the origin
def ctr_mts(mts: list, order: int):
    moment_c=0
    for j in range(order+1):
        _comb = comb(order, j)
        if j==order:
            a=1
        else:
            a=mts[order-j-1]
        moment_c += (-1)**j *_comb * mts[0]**j * a
    return moment_c

def _mts_loop(mts: list, epsilon: float, factor: int, variant: int):
    result = []
    std=ctr_mts(mts, 2)**.5
    mu2 = [float(i) * epsilon * factor * std + mts[0] for i in range(1, int(1/ epsilon))]
    m2n = M2N(mts = mts, epsilon = epsilon, variant = variant)
    err_min=m2n.error
    for j in mu2:
        m2n.fit(j)
        if m2n.error < err_min:
            # as long as your error_min is lower than the original
            # it will append so you might have more than 1 param set in a single run
            #print(m2n.param, m2n.error)
            m2n.param.append(m2n.error)
            result.append(m2n.param)
            err_min=m2n.error
    return result

def mts_fit(mts: list,
            epsilon: float = 1e-5,
            factor: int = 5,
            variant: int = 0,
            n_run: int = 10,
            f_params: bool = True,
            num_threads: int = 1):
    
    jobs, out = [], np.array([])
    for i in np.arange(n_run):
        job = {'func': _mts_loop, 
               'mts': mts, 
               'epsilon': epsilon, 
               'factor': factor,
               'variant': variant}
        jobs.append(job)
    if num_threads == 1:
        _out = process_jobs_(jobs)
    else:
        _out = process_jobs(jobs, num_threads = num_threads)
    out = [i for _list in _out for i in _list]
    out = pd.DataFrame(out, columns = ['mu1', 'mu2', 'std_1', 'std_2', 'p1', 'err'])
    if f_params:
        out = out[out.err <= out.err.mean()]
    return out

def get_param(param: pd.DataFrame):
    _p1, _mu1, _mu2  = param.p1.mean(), param.mu1.mean(), param.mu2.mean()
    
    _std_1, _std_2 = param.std_1.mean(), param.std_2.mean()
    return np.array([_mu1, _mu2, _std_1, _std_2, _p1])

def m_dist(data: float, param: list):

    _mu1, _mu2, _std_1, _std_2, _p1  = param
    #print(_mu1, _mu2, _std_1, _std_2, _p1)
    _cdf = _p1 * norm.cdf(data, _mu1, _std_1) + (1 - _p1) * norm.cdf(data, _mu2, _std_2)
    return _cdf

def m_bet_EF3M(data: pd.DataFrame, param: pd.DataFrame):
    m_series = []
    m_param = get_param(param = param)
    m_dist0 = m_dist(data = 0, param = m_param)

    for idx in np.arange(data.shape[0]):
        if data.c_t[idx] >= .0:
            _cdf = (m_dist(data = data.c_t[idx], param = m_param) - m_dist0)/ (1 - m_dist0)
        else:
            _cdf = (m_dist(data = data.c_t[idx], param = m_param) - m_dist0)/ m_dist0
        m_series.append(_cdf)
    data['m'] = m_series
    return data