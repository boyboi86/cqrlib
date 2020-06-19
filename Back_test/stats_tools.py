import numpy as np
from scipy.stats import norm

def _target_sr(p: float = 0.5, freq: int = 52, ptsl: list = [0.02,-0.02], seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    rnd=np.random.binomial(n=1,p=p, size = freq)
    x = [ptsl[0] if i == 1 else ptsl[1] for i in rnd]
    mean = np.mean(x)
    std = np.std(x)
    return (mean, std, mean/std)

def target_sr(p: float = 0.5, freq: int = 52, ptsl: list = [0.02,-0.02], n_run: int = 1000000, seed: int = None):
    mean, std, sr = 0, 0, []
    for n in np.arange(n_run):
        _mean, _std, _sr = _target_sr(p = p, freq = freq, ptsl = ptsl, seed = seed)
        mean += _mean
        std += _std # std is only 0.1, because variance root
        if _sr <= .2: sr.append(_sr)
    mean = mean/n_run # var = 0.01 but std = 0.1
    std = std/n_run
    im_sr = mean/std
    print("Mean: {0:.6f}\nStd: {1:.6f}\nSharpe Ratio: {2:.6f}".format(mean, std, im_sr))
    if len(sr) >0:
        p =len(sr)/n_run
        print("Probability of getting SR < 2.: {0}%".format(100 * p))
        return (mean, std, im_sr, sr)
    else:
        print("All SR >= 2")
        return (mean, std, im_sr)

    
def im_p(freq: int = 52, trgt_sr: int = 2., ptsl: list = [0.02, -0.02]):
    pt, sl = ptsl[0], ptsl[1]
    a = (freq + trgt_sr ** 2) * (pt - sl) ** 2
    b = (2 * freq * sl - trgt_sr ** 2 * (pt-sl)) * (pt-sl)
    c = freq * sl ** 2
    p = (-b+(b ** 2 - 4*a*c) ** .5)/ (2*a)
    print("\nImplied Precision Rate Required: {0:.6f}".format(p))
    return p

def im_freq(p: float = 0.6, trgt_sr: int = 2., ptsl: list = [0.02, -0.02]):
    pt, sl = ptsl[0], ptsl[1]
    freq = (trgt_sr * (pt - sl)) ** 2*p*(1-p)/((pt-sl)*p+sl)**2
    print("\nImplied Frequency Required: {0:.6f}".format(freq))
    return int(freq)

def im_pt(freq: int, trgt_sr: float, p: float, sl: float):
    pt = (sl * freq**(1/2)) / (trgt_sr * (p*(1-p))**(1/2) - p * freq**(1/2)) + sl
    print("\nImplied Profit-taking level: {0:.6f}".format(pt))
    return pt

def im_sl(freq: int, p: float, pt: float, trgt_sr: float):
    sl = (pt * (trgt_sr * (p*(1-p))**(1/2) - p * freq**(1/2))) / \
    (trgt_sr * (p*(1-p))**(1/2) - p * freq**(1/2) + freq**(1/2))
    print("\nImplied Stop-loss limit: {0:.6f}".format(sl))
    return sl

def mix_gauss(mu1: float, mu2: float, sig1: float, sig2: float, p: float, n_obs: int):
    rtn1 = np.random.normal(mu1, sig1, size=int(n_obs * p))
    rtn2 = np.random.normal(mu2,sig2, size=int(n_obs) - rtn1.shape[0])
    rtn = np.append(rtn1,rtn2, axis = 0)
    np.random.shuffle(rtn)
    return rtn

def prob_failure(rtn: float, freq: int, trgt_sr: float):
    pos_rtn, neg_rtn = rtn[rtn>0].mean(), rtn[rtn<=0].mean()
    p = rtn[rtn>0].shape[0]/ float(rtn.shape[0])
    ptsl = [pos_rtn, neg_rtn]
    threshold = im_p(freq = freq, trgt_sr = trgt_sr, ptsl = ptsl)
    risk = norm.cdf(threshold, p, p * (1 - p))
    print("Predicted Precision Pate: {0:.6f}\n".format(p))
    return risk

def strategy_failure(mu1: float, mu2: float, sig1: float, sig2: float, p: float, n_obs: int, freq: int, trgt_sr: float):
    """
    Requires to be replace with EF3M algo to provide real rtn data
    """
    rtn = mix_gauss(mu1 = mu1,
                    mu2 = mu2,
                    sig1 = sig1,
                    sig2 = sig2,
                    p = p,
                    n_obs = n_obs)
    
    _proba_failure = prob_failure(rtn = rtn,
                                  freq = freq,
                                  trgt_sr = trgt_sr)
    
    print("Strategy Failure Probability: {0:.5f}".format(_proba_failure))
    if _proba_failure> 0.05:
        print("Discard Strategy; High risk indicated")
    else:
        print("Accept Strategy; Moderate risk indicated")