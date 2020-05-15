import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
   

# =======================================================================
# list of component across all asset class
# before transaforming it into cov matrix
# time-series based on asset return/ volatility for x period
# =======================================================================

# =======================================================================
# import a list of componenet based on time-series before deriving mean of each factor
# over standard sample period to derive the initiate example matrix below:
#
#           returns sigma div i/r
# asset1    [[x1   x2    x3    x4]
# asset2    [x1   x2    x3    x4]
# asset3    [x1   x2    x3    x4]]
# 
# =======================================================================
'''
asset_arr = np.array([asset1, asset2, asset3])

def asset_risk(asset_arr):
    cov_matrix = np.cov(asset_arr,bias=True)
    sns.heatmap(cov_matrix, annot=True, fmt='g')
    plt.show()
    return cov_matrix
# =======================================================================
# the derived eignvalue/vector will be applied based on inverse
# from there assign risk weight for portfolio hedge based on initial matrix
# =======================================================================       

def pca_weight(cov_matrix, risk_dist = None, risk_tar = 1.0):
    e_val, e_vec = np.linalg.eigh(cov_matrix)
    indices = e_val.argsort()[::-1]
    e_val, e_vec = e_val[indices], e_vec[:,indices]
    if risk_dist is None:
        risk_dist = np.zeros(cov_matrix.shape[0])
        risk_dist[-1] = 1
    loads = risk_tar * (risk_dist/e_val) ** 0.5
    weights = np.dot(e_val, np.reshape(loads, (-1,1)))
    return weights
'''