import numpy as np
import scipy.stats as si

    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    #q: continuous dividend rate

p = print

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

# div paying asset using BSM
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

#ATM price
print(bsm_price('c', 0.3, 3, 3, 0.032, 30/365, 0.01))
print(bsm_price('p', 0.3, 3, 3, 0.032, 30/365, 0.01))

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
