import numpy as np
import scipy.stats as si

    #S: spot price
    #K: strike price
    #T: time to maturity % year; choose 252 trading days or 365 calendar days i.e 30/252 or 30/365, it should be how much time left not total option duration
    #r: interest rate
    #sigma: volatility of underlying asset
    #q: continuous dividend rate

p = print
    # annual dividend yield; use this only if you are unsure
def div_yield(freq, S, div):
    div_yield = np.log((div * freq)/S)
    return div_yield

def err_msg(option_type):
    p('{0} is not valid option type\nchoose c for call and p for put').format(option_type)

def checks(sigma, S, K, r, T, q):
    arr = [sigma, S, K, r, T, q]
    for i in arr:
        if np.isnan(i):
            p('input not number')
        if not i:
            p('input missing\n ')

def d1(sigma, S, K, r, T, q):
    checks(sigma, S, K, r, T, q)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1

def d2(sigma, S, K, r, T, q):
    checks(sigma, S, K, r, T, q)
    d2 = d1(sigma, S, K, r, T, q) - sigma * np.sqrt(T)
    return d2

    #div paying asset using BSM
def bsm_price(option_type, sigma, S, K, r, T, q):

    option_type = option_type.lower()  
    sigma = float(sigma)
    
    #bsm formula for pricing
    _d1 = d1(sigma, S, K, r, T, q)
    _d2 = d2(sigma, S, K, r, T, q)
    
    if option_type == 'c':
        price = S * si.norm.cdf(_d1) - K * np.exp(-r * T) * si.norm.cdf(_d2)
        return price
    elif option_type == 'p':
        price = K * np.exp(-r * T) * si.norm.cdf(-_d2) - S * np.exp((r - q) * T) * si.norm.cdf(-_d1)
        return price
    else:
        err_msg(option_type)

    #bsm formula for delta
def delta(option_type, sigma, S, K, r, T, q):
    option_type = option_type.lower()  
    sigma = float(sigma)
    
    _d1 = d1(sigma, S, K, r, T, q)
    
    if option_type == 'c':
        delta = np.exp(-q*T) * si.norm.cdf(_d1)
        return delta
    elif option_type == 'p':
        delta = np.exp(-q*T) * (si.norm.cdf(_d1) - 1)
        return delta
    else:
        err_msg(option_type)

    #bsm formula for gamma
def gamma(sigma, S, K, r, T, q):
    sigma = float(sigma)
    
    _d1 = d1(sigma, S, K, r, T, q)

    gamma = np.exp(-q * T - 0.5 * _d1 ** 2)/ (S * sigma * np.sqrt(2 * np.pi * T))
    return gamma

    #bsm formula for vega
def vega(sigma, S, K, r, T, q):
    sigma = float(sigma)
    
    _d1 = d1(sigma, S, K, r, T, q)

    vega = 0.01 * S * np.exp(-q * T) * np.sqrt(T) * np.exp(- 0.5 * _d1 ** 2)/ np.sqrt(2 * np.pi)
    return vega

#def IV(option_type, sigma, S, K, r, T, q):
    