import numpy as np
import pandas as pd
import scipy.stats as si

#from arch.unitroot import ADF, KPSS

import statsmodels.stats.diagnostic as sm
import statsmodels.api as smi

from statsmodels.tsa.stattools import adfuller, kpss

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

p = print

#==============================================================================
# Model volatility comparison
# Input data array output to test sample period stability
# optimal signal period test include original data along with test samples
#==============================================================================

vol_arr = []

def vol_stability_test(vol_arr):
    _vol = []
    for i in vol_arr:
        _vol.append(np.std(i))
        p("{0} with std {1}").format(vol_arr[0], _vol[0])
    _arr = sorted(zip(vol_arr, _vol), key=lambda x: x[1])
    return _arr

def jb(x,test=True): 
    np.random.seed(12345678)
    if test: 
        p(si.jarque_bera(x)[0])
    return si.jarque_bera(x)[1]

def shapiro(x,test=True): 
    np.random.seed(12345678)
    if test: 
        print(si.shapiro(x)[0])
    return si.shapiro(x)[1]
    
#==============================================================================
# Normality Test
# Against Skewness as well as Kurtosis of series
# If it is not single model, it could be used for goodness of fit comparison
#==============================================================================

def normality(data: pd.Series, test_limit = 0.05):
    '''
    Normality test, to see if series can be considered gauss.
    Do take note, having a gauss distribution is not equal to gauss iid.
    The latter is a joint probability distribution.
    
    Shapiro Wilk and Jarque-Bera will be used
    
    params: data => close price series
    params: test_limit => critical size 0.01, 0.05, 0.1 default: 0.05
    '''
    p("=" * 33)
    count = len(data)
    p(("Total Sample: {0}").format(count))
    if count >= 2000:
        p("\nShapiro Wilk Test recommends at most 2000 sample")
        p("Otherwise test results might not be reliable\n")
    elif count <= 1900:
        p("Jarque-Bera Test recommends at least 2000 sample\n")
        p("Otherwise test results might not be reliable\n")
    else:
        p("Sample Size Suffice for Normality Test\n")
    p("Jarque-Bera Test Results")
    jb_ = jb(data, test=True)
    if jb_ >= test_limit:
        p("Null Hypothesis Accepted: Not Normal Distributed\n")
    else:
        p("Null Hypothesis Rejected: Normal Distributed\n")
    p("Shapiro Wilk Test Results")
    sw_ = shapiro(data, test=True)
    if sw_ <= test_limit:
        p("Null Hypothesis Rejected: Not Normal Distributed\n")
    else:
        p("Null Hypothesis Accepted: Normal Distributed\n")
    p("=" * 33)

#==============================================================================
# Stationarity Test
# Auto-regression effects
# If too strong, might require 2nd order differentiation
# Otherwise EWMA may be considered especially if [a,b] > 1
# If it is not single model, it could be used for goodness of fit comparison
#==============================================================================

def unit_root(data: pd.Series):
    '''
    Unit test is also know as stationary test.
    In this func, there is 2 test to acertain stationarity.
    ADF & KPSS (Usually if ADF can pass, KPSS is usually not a problem)
    
    params: data => close price series
    '''
    adf = adfuller(data, maxlg = 1, regression='c', autolag = None)[1]
    kpss_ = kpss(data, regression='ct', lags='auto')[1]
    if (adf <= 0.05 and kpss >= 0.05):
        p("\nADF & KPSS: Weak evidence for stationary\n")
    elif adf <= 0.05:
        p('\nReject ADF Null hypothesis: Weak evidence that series is stationary')
        p('Reject KPSS Null hypothesis: Series contains unit root\n')
    elif kpss_ >= 0.05:
        p('\nFail to reject ADF Null hypothesis: Series contains unit root')
        p('Fail to reject KPSS Null hypothesis: Weak evidence that series is stationary\n')
    else:
        p('\nSeries has unit root therefore not stationary\n')

#==============================================================================
# Random Test
# Check for heteroscedasticity within data sample
#==============================================================================

def white_random(data: pd.DataFrame):
    '''
    params: data => close price series
    White test is meant to test if series is homoscedastic or heteroscedastic
    null hypothsis: homoscedastic (> 0.05)
    alt hypothsis: heteroscedastic
    
    If homoscedastic would meant series have constant dispersion, ststistically good for mean reverting.
    Test uses OLS non-log. So do take note when using it.
    '''
    data['std1'] = data['close'].std()
    p("Residual: {0}\nExog count: {1}".format(data['std1'][0], data['close'].count()))
    data.dropna(inplace= True)
    X = smi.tools.tools.add_constant(data['close'])
    results = smi.regression.linear_model.OLS(data['std1'], X).fit()
    resid = results.resid
    exog = results.model.exog
    p("White-Test p-Value: {0}".format(sm.het_white(resid, exog)[1]))
    if sm.het_white(resid, exog)[1] > 0.05:
        p("White test outcome at 5% signficance: homoscedastic")
        p("Reject null hypothesis at critical size: 0.05")
    else:
        p("White test outcome at 5% signficance: heteroscedastic")
    return data.drop(columns = ['std1'])

#==============================================================================
# Classification report for ML
#==============================================================================
    
def report_matrix(data: pd.Series):
    '''
    This func is meant to be used with meta labels
    data input should be func label => output
    
    params: data => metalabels from func label i.e. label
    '''

    forecast0 = data['bin'].to_frame(name = 'actual')
    forecast0['pred'] = pd.Series(1, index = forecast0.index)
    
    actual, pred = forecast0['actual'], forecast0['pred']
    
    if (actual.value_counts()[1] < actual.value_counts()[0]) and pred.iloc[1] != 1:
        p("prediction value should be 1\n")
    elif (actual.value_counts()[1] > actual.value_counts()[0]) and pred.iloc[1] != 0:
        p("prediction value should be 0\n")
    
    sep = "=" * 55
    p(" Classification Report\n{0}\n{1}\n"
      .format(sep, 
              classification_report(y_true=actual,
                                    y_pred=pred)))

    p(" Confusion Matrix\n{0}\n{1}\n{2}\n".format(sep,"[[TN, FP]\n [FN, TP]]\n", 
                                                  confusion_matrix(actual, pred)))

    p(" Accuracy Score\n{0}\n{1}\n".format(sep, accuracy_score(actual, pred)))
    
    return forecast0
    
