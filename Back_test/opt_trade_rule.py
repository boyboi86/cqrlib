import numpy as np
import pandas as pd

from random import gauss
from itertools import product

from research.Util.multiprocess import mp_pandas_obj

def _opt_tr(molecule,
            n_run: float = 100,
            max_period: int = 100,
            seed: int = 0):
    
    out = pd.DataFrame(.0, index = molecule.index, columns = ['profit', 'loss', 'mean', 'std', 'sharpe'])
    for idx, forecast,	half_life, sigma, profit_range,	loss_range in molecule.itertuples():
    #can try pandas itertuples for entire series
        phi, output1 = 2 ** (-1/ half_life), []
        for iter_ in range(int(n_run)):
            p, hold_period = seed, 0
            while True:
                p = (1 - phi) * forecast + phi * p + sigma * gauss(0,1)
                cp = p - seed
                hold_period += 1
                if profit_range < cp or - loss_range > cp or hold_period > max_period:
                    output1.append(cp)
                    break
            
        mean, std = np.mean(output1), np.std(output1)
        out.loc[idx, 'mean'], out.loc[idx, 'std'], out.loc[idx, 'sharpe'] = mean, std, mean/std
        out.loc[idx, 'profit'], out.loc[idx, 'loss'] = profit_range, loss_range
    return out

def opt_tr(profit_range: list = np.linspace(.5, 10, 20),
          loss_range: list = np.linspace(.5, 10, 20),
          sigma: list = [1],
          forecast: list = [10,5,0,-5,-10], 
          half_life: list = [5,10,25,50,100],
          n_run: float = 100,
          max_period: int = 100,
          seed: int = 0,
          num_threads: int = 1):
    
    if profit_range is None:
        profit_range = np.linspace(0,10,21)
        
    if loss_range is None:
        loss_range = np.linspace(0,10,21)
        
    _coeff = dict({'forecast': forecast,
                   'half_life': half_life,
                   'sigma':sigma,
                   'profit_range': profit_range,
                   'loss_range':loss_range})
    
    _coeff = list(dict(zip(_coeff, idx )) for idx in product(*_coeff.values()))
    jobs = pd.DataFrame(_coeff)
            
    out = mp_pandas_obj(func=_opt_tr,
            pd_obj = ('molecule', jobs), #we need the index so that during partition they don't go haywire
            num_threads=num_threads, 
            mp_batches=1, 
            lin_mols=True, 
            axis=0,
            seed = seed,
            n_run = n_run,
            max_period = max_period)

    return out