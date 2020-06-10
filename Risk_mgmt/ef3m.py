import numpy as np
import pandas as pd
import random

from scipy.stats import norm, moment
from scipy.special import comb
#-------------------------------------------
# Define the problem here
mts=[0.7,2.6,0.4,25,-59.8] # about the origin
epsilon=10**-5
factor=5 # this is the ‘lambda’ referred in the paper
#-------------------------------------------
# This is the mixture’s class
class M2N:
    
    def __init__(self, mts, epsilon: float = 1e-5, variant: int = 0):
        self.mts = mts
        self.param = [0 for i in range(5)]
        self.error = sum([mts[i]**2 for i in range(len(mts))])
        self.epsilon = epsilon
        self.variant = variant
        
    def fit(self, mu2):
        p1 = random.random()
        numIter = 0
        while True:
            numIter += 1
            if self.variant == 0:
                param=iter4(mu2,p1,self.mts) # for the first variant
            else:
                param=iter5(mu2, p1, self.mts) # for the second variant
            mts = get_mts(param)
            error=sum([(self.mts[i]-mts[i])**2 for i in range(len(mts))])
            if error<self.error:
                self.param=param
                self.error=error
            if abs(p1 - param[4]) < epsilon: return #limitations
            if numIter > 1/epsilon: return #limitlations
            p1=param[4]
            mu2=param[1] # for the 5th moment’s convergence
            

#-------------------------------------------
# Derive the mixture’s mts from its param
        def get_mts(param):
            m1 = param[4] * param[0] + (1 - param[4]) * param[1]
            
            m2 = param[4] * (param[2]**2 + param[0]**2) + (1-param[4])* (param[3] **2 + param[1]**2)
            
            m3 = param[4] * (3 * param[2]**2 * param[0] + param[0]**3) + (1-param[4]) * (3 * param[3]**2 * param[1] + param[1]**3)
            
            m4 = param[4] * (3 * param[2]**4 + 6 * param[2]**2 * param[0]**2 + param[0]**4)+(1 - param[4]) * \
            (3 * param[3]**4 + 6 * param[3]**2 * param[1]**2 + param[1]**4)
            
            m5=param[4] * (15 * param[2]**4 * param[0] + 10 * param[2]**2 * param[0]**3 + param[0]**5) + (1 - param[4]) * \
            (15*param[3]**4 * param[1]+10*param[3]**2*param[1]**3+param[1]**5)
            
        return [m1, m2, m3, m4, m5]
#-------------------------------------------
# Equations for variant 1
        def iter4(mu2,p1,mts):
            mu1 = (mts[0]-(1-p1)*mu2)/p1
            
            sig_2 = ((mts[2] + 2 * p1 * mu1**3 + (p1 - 1) * mu2**3 - 3 * mu1 * (mts[1] + mu2**2 * (p1-1)))/(3 * (1 - p1) * (mu2 - mu1)))**(.5)
            
            sig_1 = ((mts[1] - sig_2**2 - mu2**2) / p1 + sig_2**2 + mu2**2 - mu1**2)**(.5)
            
            p1 = (mts[3] - 3 * sig_2**4 - 6 * sig_2**2 * mu2**2 - mu2**4)/ \
            (3 * (sig_1**4 - sig_2**4)+ 6 * (sig_1**2 * mu1**2 - sig_2**2 * mu2**2) + mu1**4 - mu2**4)
            
        return [mu1, mu2, sig_1, sig_2, p1]
#-------------------------------------------
# Equations for variant 2
        def iter5(mu2,p1,mts):
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
# Number of combinations of n over k
        def binomialCoeff(n, k):
            if k<0 or k>n:
                return 0
            if k>n-k:
                k=n-k
            c = 1
            for i in range(k):
                c = c * (n - (k - (i + 1)))
                c = c//(i + 1)
            return c
"""

#-------------------------------------------
# Compute mts about the mean (or centered) from mts about the origin
def centeredMoment(mts,order):
    moment_c=0
    for j in range(order+1):
        _comb = comb(order, j)
        if j==order:
            a=1
        else:
            a=mts[order-j-1]
        moment_c += (-1)**j *_comb * mts[0]**j * a
    return moment_c
#-------------------------------------------
# Main function
        def main():
            stDev=centeredMoment(mts,2)**.5
            mu2=[float(i) * epsilon * factor * stDev + mts[0] for i in range(1, int(1/ epsilon))]
            m2n=M2N(mts, epsilon, variant)
            err_min=m2n.error
            for i in mu2:
                m2n.fit(i, epsilon)
            if m2n.error < err_min:
            print(m2n.param, m2n.error)
            err_min=m2n.error
#-------------------------------------------
# Boilerplate
if __name__=='__main__': main()