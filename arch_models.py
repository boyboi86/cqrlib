from bsm_model import IV

# Read SPY CVS file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import ProbPlot
from arch.univariate import ConstantMean, GARCH, Normal
from arch.unitroot import ADF, KPSS

p = print

#IV(option_type, sigma, S, K, r, T, q)
#p(IV('c', 0.3, 3, 3, 6, 30/365, 0.6))

# CBOE benchmark

vix = pd.read_csv("^VIX.csv")
vix.rename(columns={'Adj Close':'CBOE'}, inplace=True)
vix['CBOE'] = vix['CBOE'] * 0.01
pd.to_datetime(vix['Date'])
vix.index = vix['Date']
vix.drop(labels=['Date','Open','Close','High','Low','Volume'], axis=1, inplace=True)

# Conversion of data with index to time using to_datetime()
df = pd.read_csv("SPY.csv")
pd.to_datetime(df['Date'])
df.index = df['Date']
df.drop(labels=['Date','Open','Close','High','Low','Volume'], axis=1, inplace=True)


def unit_test(inputdata):
    adf = ADF(inputdata)
    kpss = KPSS(inputdata)
    adf.trend = 'ct'
    kpss.trend = 'ct'
    if (adf.pvalue <= 0.05 and kpss.pvalue >= 0.05):
        p("************************************************************")
        p("ADF & KPSS: Strong evidence process is stationary")
        p("************************************************************")
    elif adf.pvalue <= 0.05:
        p("************************************************************")
        p(adf.summary())
        p('\nReject Null hypothesis: Process is stationary')
        p("************************************************************\n")
    elif kpss.pvalue >= 0.05:
        p("************************************************************")
        p(kpss.summary())
        p('\nFail to reject Null hypothesis: Process is stationary')
        p("************************************************************\n")
    else:
        p("************************************************************")
        p('Process has unit root therefore not stationary')
        p("************************************************************\n")

# -------------- Draft graphs for any graph --------------
def originalplot(inputdata):
    plt.figure(figsize=(15,5))
    inputdata.plot()

# -------------- Draft graphs for returns, acf, pacf, QQ, PP --------------
def plotgraphfit(returns, lags):
    # ---------- Show original return ----------------------------
    originalplot(returns)
    # ---------- Draft graphs for acf, pacf, QQ, PP --------------
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(15,10))
    plot_acf(returns, lags = lags, ax = ax1)
    plot_pacf(returns, lags = lags, ax = ax2)
    pp=ProbPlot(returns, fit = True)
    ax3.set_title('QQ plot')
    pp.qqplot(line='r', ax = ax3)
    ax4.set_title('PP plot')
    pp.probplot(line='r', ax = ax4)
    
def garch_setup(returns, ts_params = False, summary = False):
    returns = returns * 100
    am = ConstantMean(returns)
    am.volatility = GARCH(1, 0, 1)
    am.distribution = Normal()
    res = am.fit()
    if ts_params == True:
        df['omega'] = res.params['omega']
        df['alpha'] = res.params['alpha[1]']
        df['beta'] = res.params['beta[1]']
    df['forecast_volatility'] = np.sqrt( 0.01 * (res.params['omega'] + res.params['alpha[1]'] * res.resid ** 2 + res.conditional_volatility ** 2 * res.params['beta[1]']))
    if summary == True:
        p('\n', res.summary())

def return_type(inputdata, ln_return = False, summary = False):
    # ---------- either log return or non-log ----------------------------------
    # ---------- Recall Taylor approximation when comparing --------------------
    if ln_return == False:
        df['returns'] = inputdata.pct_change()
    else:
        df['returns'] = np.log(inputdata) - np.log(inputdata.shift(1))
    garch_setup(df['returns'].dropna(), ts_params = True, summary = summary)
        
def count_std(returns, period = 1 , period_type = 'T', summary = False):
    # ---------- 252 trading days as default -----------------------------------
    if period_type.lower() == 't':
        period_day = 252
    # ---------- for longer measure but take note BSM may not accept -----------
    elif (period_type.dtype == np.float64 | period_type.dtype == np.int64):
        period_day = period_type
    # ---------- 365 calendar days ---------------------------------------------
    else:
        period_day = 365
    # ---------- generate standard deviations, annualized volatility -----------
    df['roll_std'] = df['returns'].rolling(period).std()
    df['annual_vol'] = df['roll_std'] * (period_day ** 0.5) 
    # ---------- IV(option_type, sigma, S, K, r, T, q) -------------------------
    df['iv'] = IV('c', df['annual_vol'], returns, returns, 0.03, period / period_day, 0.0175)
    df.dropna(inplace=True)
    
    
    if summary == True:
        p("\n************************************************************")
        p("Period used: {0}           Period length:{1}".format(period, period_day))
        p("Dataset shape:{0}".format(df.shape))
        p("************************************************************\n")
        



def plotgraph():
    plt.figure(figsize=(18,8))
    line1, = plt.plot((df['iv']), color = 'orange') #IV BSM
    line2, = plt.plot((df['forecast_volatility']),color='blue')
    line3, = plt.plot( (vix['CBOE']), color = 'red')
    plt.legend([ line1, line2, line3],['BSM IV', 'GARCH Estimated Volatility', 'CBOE VIX Benchmark'])
    plt.show()


ac = df['Adj Close']

return_type(ac, ln_return = False, summary = True)
count_std(ac, period = 21, period_type = 'T', summary = True)

pc = df['returns']
unit_test(pc)
plotgraph()


