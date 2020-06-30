import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch


class HRP:
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.cov = data.cov()
        self.corr = np.corrcoef(data, rowvar = 0)
        self.labels = data.columns
        
    def solve(self, method: str = 'single', rescale: bool = False, long_only: bool = True):
        """
        Need to recover pandas dataframe therefore created numpy corref and pandas corr
        1. To avoid clusterwarning for redundant matrix
        2. To ensure we can work with pandas
        """
        methods = ['ward', 'single', 'weighted', 'average', 'centroid']
        if not method in methods:
            raise ValueError("Method not available, default is 'single'.")
        if rescale is False:
            df = self.data
            cov, corr, corr1 = self.cov, self.corr, df.corr()
        else:
            df, cov, corr = self.rescale_cov()
            corr1 = df.corr()
            
        dist = self.corr_dist(corr = corr, long_only = long_only)
        link = sch.linkage(y = dist, method = method, metric='euclidean')
        sort_idx = self.quasi_diag(link = link)
        sort_idx = corr1.index[sort_idx].tolist()    
        hrp = self.rec_bipart(cov = cov, sorted_idx = sort_idx)
        return hrp.sort_index()
    
    def rescale_solve(self, method: str = 'single', n_run: int = 100, long_only: bool = True):
        num_iter = 0
        while num_iter < n_run + 1:
            _hrp = self.solve(method = method, rescale = True, long_only = long_only)
            if num_iter == 0:
                hrp_df = pd.DataFrame(_hrp)
            else:
                hrp_series = pd.Series(_hrp, name = num_iter)
                hrp_df = hrp_df.join(hrp_series)
            num_iter += 1
        hrp_df = pd.DataFrame(hrp_df.mean(axis = 1), columns = ['weights'])
        hrp_df.div(hrp_df.sum()) #rescale to ensure all weights add to 1
        return hrp_df
        
    
    def corr_dist(self, corr: np.ndarray, long_only: bool = True):
        if long_only:
            dist = np.sqrt((1 - corr) / 2.)
        else:
            dist = np.sqrt((1 - np.absolute(corr)))
        return dist  
    
    def quasi_diag(self, link: np.ndarray):
        link = link.astype(int)
        sort_idx = pd.Series([link[-1,0], link[-1,1]])
        n_items = link[-1,3]
        while sort_idx.max() >= n_items:
            sort_idx.index = np.arange(0, sort_idx.shape[0] * 2, 2)
            df0 = sort_idx[sort_idx>=n_items]
            i = df0.index
            j = df0.values - n_items
            sort_idx[i] = link[j,0]
            df0 = pd.Series(link[j,1], index = i+1)
            sort_idx = sort_idx.append(df0)
            sort_idx = sort_idx.sort_index()
            sort_idx.index = np.arange(sort_idx.shape[0])
        return sort_idx.tolist()
    
    def rec_bipart(self, cov: pd.DataFrame, sorted_idx: pd.DataFrame):
        weights = pd.Series(1, index=sorted_idx)
        clustered_alphas = [sorted_idx]
    
        while len(clustered_alphas) > 0:
            clustered_alphas = [cluster[start:end] for cluster in clustered_alphas
                                for start, end in ((0, len(cluster) // 2),
                                                   (len(cluster) // 2, len(cluster)))
                                if len(cluster) > 1]
            for subcluster in range(0, len(clustered_alphas), 2):
                left_cluster = clustered_alphas[subcluster]
                right_cluster = clustered_alphas[subcluster + 1]
    
                left_subcovar = cov[left_cluster].loc[left_cluster]
                inv_diag = 1 / np.diag(left_subcovar.values)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                left_cluster_var = np.dot(parity_w, np.dot(left_subcovar, parity_w))
    
                right_subcovar = cov[right_cluster].loc[right_cluster]
                inv_diag = 1 / np.diag(right_subcovar.values)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                right_cluster_var = np.dot(parity_w, np.dot(right_subcovar, parity_w))
    
                alloc_factor = 1 - left_cluster_var / (left_cluster_var + right_cluster_var)
    
                weights[left_cluster] *= alloc_factor
                weights[right_cluster] *= 1 - alloc_factor
                
        return weights
    
    def rescale_cov(self):
        data = self.data
        cov = self.cov
        labels = self.labels
        num = data.shape[1]
        e_val, e_vec = np.linalg.eig(cov)
        e_matrix = np.identity(n=num) * e_val
        e_matrix = pd.DataFrame(e_matrix, index = labels, columns = labels)
        eps = np.random.uniform(0, 1, size = num)
        e_matrix = (num * eps * e_matrix * (sum(eps) **-1)) #based on formula
        e_df1 = e_vec @ e_matrix @ np.linalg.inv(e_vec) #this is matrix do not use * to multiply
        e_df1 = pd.DataFrame(e_df1)
        e_df1.index = labels
        e_df1.columns = labels
        cov, corr = e_df1.cov(), np.corrcoef(e_df1, rowvar = 0)
        #self.rescale_data, self.rescale_cov, self.rescale_corr = e_df1, e_df1.cov(), np.corrcoef(e_df1, rowvar = 0)
        return e_df1, cov, corr
    
"""
def quasi_diag(link):
    link = link.astype(int)
    sort_idx = pd.Series([link[-1,0], link[-1,1]])
    n_items = link[-1,3]
    while sort_idx.max() >= n_items:
        sort_idx.index = np.arange(0, sort_idx.shape[0] * 2, 2)
        df0 = sort_idx[sort_idx>=n_items]
        i = df0.index
        j = df0.values - n_items
        sort_idx[i] = link[j,0]
        df0 = pd.Series(link[j,1], index = i+1)
        sort_idx = sort_idx.append(df0)
        sort_idx = sort_idx.sort_index()
        sort_idx.index = np.arange(sort_idx.shape[0])
    return sort_idx.tolist()

def rec_bipart(covariances, res_order):
    weights = pd.Series(1, index=res_order)
    clustered_alphas = [res_order]

    while len(clustered_alphas) > 0:
        clustered_alphas = [cluster[start:end] for cluster in clustered_alphas
                            for start, end in ((0, len(cluster) // 2),
                                               (len(cluster) // 2, len(cluster)))
                            if len(cluster) > 1]
        for subcluster in range(0, len(clustered_alphas), 2):
            left_cluster = clustered_alphas[subcluster]
            right_cluster = clustered_alphas[subcluster + 1]

            left_subcovar = covariances[left_cluster].loc[left_cluster]
            inv_diag = 1 / np.diag(left_subcovar.values)
            parity_w = inv_diag * (1 / np.sum(inv_diag))
            left_cluster_var = np.dot(parity_w, np.dot(left_subcovar, parity_w))

            right_subcovar = covariances[right_cluster].loc[right_cluster]
            inv_diag = 1 / np.diag(right_subcovar.values)
            parity_w = inv_diag * (1 / np.sum(inv_diag))
            right_cluster_var = np.dot(parity_w, np.dot(right_subcovar, parity_w))

            alloc_factor = 1 - left_cluster_var / (left_cluster_var + right_cluster_var)

            weights[left_cluster] *= alloc_factor
            weights[right_cluster] *= 1 - alloc_factor
            
    return weights

def corr_dist(corr):
    dist = np.sqrt((1 - corr) / 2.)
    return dist  
    
def HRP(x: pd.DataFrame):
    
    cov, corr = x.cov(), x.corr()
    dist = corr_dist(corr = corr)
    link = sch.linkage(dist, 'single')
    sort_idx = quasi_diag(link = link)
    sort_idx = corr.index[sort_idx].tolist()    
    hrp = rec_bipart(covariances = cov, res_order = sort_idx)
    return hrp.sort_index()
"""