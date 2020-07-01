import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


#===================================================================================
# final algo

class NCO:
    
    def __init__(self, mu0, cov0):
        self.mu0 = mu0,
        self.cov0 = cov0
        self.nObs = cov0.shape[1]
        
    def solve(self,
              n_runs: int = 100, 
              bWidth: float = .25,
              minVarPortf: bool = True, 
              shrink: bool = False):
        
        mu0 = self.mu0[0]
        cov0 = self.cov0
        nObs = self.nObs
        w1, w1_d = self.monteCarlo(mu0 = mu0,
                                   cov0 = cov0,
                                   nObs = nObs,
                                   n_runs = n_runs,
                                   bWidth = bWidth,
                                   minVarPortf = minVarPortf,
                                   shrink = shrink)
        
        w0 = self.optPort_cvo(cov0,None if minVarPortf else mu0)
        w0=np.repeat(w0.T,w1.shape[0],axis=0) # true allocation
        err=(w1-w0).std(axis=0).mean()
        err_d=(w1_d-w0).std(axis=0).mean()
        return w0, w1, w1_d, err, err_d
    
    def monteCarlo(self,
                   mu0: np.ndarray,
                   cov0: np.ndarray,
                   nObs: int,
                   n_runs: int,
                   bWidth: float,
                   minVarPortf: bool,
                   shrink: bool):
        """
        params: mu0 => the original vector of expected outcomes.
        params: cov0 => the original covariance matrix of outcomes.
        params: nObs => the number of observations 𝑇 used to compute mu0 and cov0.
        params: n_runs => the number of simulations run in the Monte Carlo.
        params: bWidth => the bandwidth of the KDE functions used to de-noise the covariance matrix.
        params: minVarPortf => when True, the minimum variance solution is computed. Otherwise, the
                                maximum Sharpe ratio solution is computed.
        params: shrink => when True, the covariance matrix is subjected to the Ledoit-Wolf shrinkage procedure.
        """
        w1=pd.DataFrame(columns=range(cov0.shape[0]), index=range(n_runs),dtype=float)
        w1_d=w1.copy(deep=True)
        for i in range(n_runs):
            mu1,cov1=self.simCovMu(mu0,cov0,nObs,shrink)
            if minVarPortf:mu1=None
            if bWidth>0:cov1 = self.deNoiseCov(cov1,nObs*1./cov1.shape[1],bWidth)
            w1.loc[i] = self.optPort_cvo(cov1,mu1).flatten()
            w1_d.loc[i] = self.optPort_nco(cov1,mu1,int(cov1.shape[0]/2)).flatten()
        return w1, w1_d
    
    
    
    def simCovMu(self, mu0,cov0,nObs,shrink=False):
        """
        Drawing an empirical vector of means and an empirical covariance matrix
        """
        x=np.random.multivariate_normal(mu0.flatten(),cov0,size=nObs)
        mu1=x.mean(axis=0).reshape(-1,1)
        if shrink:cov1=LedoitWolf().fit(x).covariance_
        else:cov1=np.cov(x,rowvar=0)
        return mu1,cov1
    
    def fitKDE(self, obs,bWidth=.25,kernel='gaussian',x=None):
        # Fit kernel to a series of obs, and derive the prob of obs
        # x is the array of values on which the fit KDE will be evaluated
        if len(obs.shape)==1:obs=obs.reshape(-1,1)
        kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
        if x is None:x=np.unique(obs).reshape(-1,1)
        if len(x.shape)==1:x=x.reshape(-1,1)
        logProb=kde.score_samples(x) # log(density)
        pdf=pd.Series(np.exp(logProb),index=x.flatten())
        return pdf
    #------------------------------------------------------------------------------
    def mpPDF(self, var,q,pts):
        # Marcenko-Pastur pdf
        # q=T/N
        eMin,eMax=var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2
        eVal=np.linspace(eMin,eMax,pts)
        pdf=q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
        pdf=pd.Series(pdf.flatten(),index=eVal.flatten())
        return pdf
    #------------------------------------------------------------------------------
    def errPDFs(self, var,eVal,q,bWidth,pts=1000):
        # Fit error
        pdf0=self.mpPDF(var,q,pts) # theoretical pdf
        pdf1=self.fitKDE(eVal,bWidth,x=pdf0.index.values) # empirical pdf
        sse=np.sum((pdf1-pdf0)**2)
        return sse
    #------------------------------------------------------------------------------
    def findMaxEval(self, eVal,q,bWidth):
        # Find max random eVal by fitting Marcenko's dist to the empirical one
        out=minimize(lambda *x: self.errPDFs(*x), .5, args=(eVal,q,bWidth), bounds=((1e-5,1 - 1e-5),))
        if out['success']:var=out['x'][0]
        else:var=1
        eMax=var*(1+(1./q)**.5)**2
        return eMax,var
    
    #==============================================================================
    #------------------------------------------------------------------------------
    def corr2cov(self, corr,std):
        cov=corr*np.outer(std,std)
        return cov
    #------------------------------------------------------------------------------
    def cov2corr(self, cov):
        # Derive the correlation matrix from a covariance matrix
        std=np.sqrt(np.diag(cov))
        corr=cov/np.outer(std,std)
        corr[corr<-1],corr[corr>1]=-1,1 # numerical error
        return corr
    #------------------------------------------------------------------------------
    def getPCA(self, matrix):
        # Get eVal,eVec from a Hermitian matrix
        eVal,eVec=np.linalg.eigh(matrix)
        indices=eVal.argsort()[::-1] # arguments for sorting eVal desc
        eVal,eVec=eVal[indices],eVec[:,indices]
        eVal=np.diagflat(eVal)
        return eVal,eVec
    #------------------------------------------------------------------------------
    def denoisedCorr(self, eVal,eVec,nFacts):
        # Remove noise from corr by fixing random eigenvalues
        eVal_ = np.diag(eVal).copy()
        eVal_[nFacts:] = eVal_[nFacts:].sum()/float(eVal_.shape[0]-nFacts)
        eVal_ = np.diag(eVal_)
        corr1 = np.dot(eVec,eVal_).dot(eVec.T)
        corr1 = self.cov2corr(corr1)
        return corr1
    #------------------------------------------------------------------------------
    def deNoiseCov(self, cov0,q,bWidth):
        corr0 = self.cov2corr(cov0)
        eVal0,eVec0 = self.getPCA(corr0)
        eMax0,var0 = self.findMaxEval(np.diag(eVal0),q,bWidth)
        nFacts0 = eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
        corr1 = self.denoisedCorr(eVal0,eVec0,nFacts0)
        cov1 = self.corr2cov(corr1,np.diag(cov0)**.5)
        return cov1
    
    #===========================================
    #------------------------------------------------------------------------------
    def clusterKMeansBase(self, corr0,maxNumClusters=None,n_init=10):
        dist, silh = ((1-corr0.fillna(0))/2.)**.5, pd.Series(dtype='float64') # distance matrix
        if maxNumClusters is None:maxNumClusters=corr0.shape[0]/2
        for init in range(n_init):
            for i in range(2,maxNumClusters+1): # find optimal num clusters
                kmeans_=KMeans(n_clusters=i,n_init=1) # n_jobs deprecated
                kmeans_=kmeans_.fit(dist)
                silh_=silhouette_samples(dist,kmeans_.labels_)
                stat=(silh_.mean()/silh_.std(),silh.mean()/silh.std())
                if np.isnan(stat[1]) or stat[0]>stat[1]:
                    silh,kmeans=silh_,kmeans_
        newIdx=np.argsort(kmeans.labels_)
        corr1=corr0.iloc[newIdx] # reorder rows
        corr1=corr1.iloc[:,newIdx] # reorder columns
        clstrs={i:corr0.columns[np.where(kmeans.labels_==i)[0]].tolist() for \
                i in np.unique(kmeans.labels_)} # cluster members
        silh=pd.Series(silh,index=dist.index)
        return corr1,clstrs,silh
    #------------------------------------------------------------------------------
    def optPort_cvo(self, cov,mu=None):
        inv=np.linalg.inv(cov)
        ones=np.ones(shape=(inv.shape[0],1))
        if mu is None:mu=ones
        w=np.dot(inv,mu)
        w/=np.dot(ones.T,w)
        return w
    #------------------------------------------------------------------------------
    def optPort_nco(self, cov,mu=None,maxNumClusters=None):
        cov=pd.DataFrame(cov)
        if mu is not None:mu=pd.Series(mu[:,0])
        corr1 = self.cov2corr(cov)
        corr1, clstrs, _ = self.clusterKMeansBase(corr1,maxNumClusters,n_init=10)
        wIntra=pd.DataFrame(0,index=cov.index,columns=clstrs.keys())
        for i in clstrs:
            cov_=cov.loc[clstrs[i],clstrs[i]].values
            mu_=(None if mu is None else mu.loc[clstrs[i]].values.reshape(-1,1))
            wIntra.loc[clstrs[i],i] = self.optPort_cvo(cov_,mu_).flatten()
        cov_=wIntra.T.dot(np.dot(cov,wIntra)) # reduce covariance matrix
        mu_=(None if mu is None else wIntra.T.dot(mu))
        wInter=pd.Series(self.optPort_cvo(cov_,mu_).flatten(),index=cov_.index)
        nco=wIntra.mul(wInter,axis=1).sum(axis=1).values.reshape(-1,1)
        return nco
    
    
