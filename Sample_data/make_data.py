import numpy as np
import pandas as pd
import datetime as dt

from sklearn.datasets import make_classification

p = print

    
def make_classification_data(n_features=40, n_informative=10, n_redundant=10, n_samples=10000, days: int = 1):
    # generate a random dataset for a classification problem
    _today = dt.datetime.today()    
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, random_state=0, shuffle=False)
    df0 = pd.date_range(periods=n_samples, freq=pd.tseries.offsets.BDay(), end=_today)
    X = pd.DataFrame(X, index=df0)
    y = pd.Series(y, index=df0).to_frame('bin')
    df0 = ['I_%s' % i for i in range(n_informative)] + ['R_%s' % i for i in range(n_redundant)]
    df0 += ['N_%s' % i for i in range(n_features - len(df0))]
    X.columns = df0
    y['w'] = 1.0 / y.shape[0]
    y['t1'] = pd.Series(y.index, index=y.index - dt.timedelta(days = days))
    y.at[-1:, 't1'] = _today
    y.t1.fillna(method ='bfill', inplace = True)
    return X, y

def create_price_data(start_price: float = 1000.00, mu: float = .0, var: float = 1.0, n_samples: int = 1000000):
    
    i = np.random.normal(mu, var, n_samples)
    df0 = pd.date_range(periods=n_samples, freq=pd.tseries.offsets.Minute(), end=dt.datetime.today())
    X = pd.Series(i, index=df0, name = "close").to_frame()
    X.iat[0, 'close'] = start_price
    X.cumsum().plot.line()
    return X.cumsum()

def make_randomt1_data(n_samples: int =10000, max_days: float = 5., Bdate: bool = True):
    # generate a random dataset for a classification problem
    if Bdate:
        _freq = pd.tseries.offsets.BDay()
    else:
        _freq = 'D'
    _today = dt.datetime.today()
    df0 = pd.date_range(periods=n_samples, freq=_freq, end=_today)
    rand_days = np.random.uniform(1, max_days, n_samples)
    rand_days = pd.Series([dt.timedelta(days = d) for d in rand_days], index = df0)
    X = pd.Series(df0 + pd.to_timedelta(rand_days, unit='d'), index = df0, name='t1').to_frame()
    X.sort_values(by='t1', inplace=True)
    return X