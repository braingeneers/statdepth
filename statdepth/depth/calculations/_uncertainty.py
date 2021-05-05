import pandas as pd 
import numpy as np
from typing import List, Union
from scipy.special import erf, binom
from scipy.integrate import tplquad, quad
import scipy.stats as stats
from scipy.stats import norm
from scipy.special import gamma, gammaincc, factorial
from tqdm import tqdm
from numpy import exp
from scipy.special import hyp2f1, binom
from scipy.stats import poisson


from ._functional import _subsequences
from ._helper import *


__all__ = ['probabilistic_normal_depth', 'probabilistic_poisson_depth']

def normcdf(x, mu, sigma):
    return norm(loc=mu, scale=sigma).cdf(x)

def _normal_containment(z, parameters: list):
    mu_i, sigma_i, mu_j, sigma_j, mu, sigma = parameters
    return (normcdf(z, mu_i, sigma_i)-normcdf(z, mu, sigma)*normcdf(z, mu_j, sigma_j))*\
            norm(loc=mu, scale=sigma).pdf(z)


def _normal_depth(means, stds, curr, f):
    n = len(means)
    cols = list(range(n))
    cols.remove(curr)
    S_nj = 0
    subseq = _subsequences(cols, 2)

    for i in tqdm(range(len(subseq))):
        sequence = subseq[i]
        i, j = sequence
        
        parameters = [
            means[i], stds[i], 
            means[j], stds[j], 
            means[curr], stds[curr]
        ]
        
        integral = quad(lambda x: f(x, parameters), -np.inf, np.inf)[0]
        S_nj += integral
        
    return S_nj / binom(n, 2)

def probabilistic_normal_depth(means, stds, f=_normal_containment):
    if len(means) != len(stds):
        raise ValueError('Error, len(means) must equal len(stds)')

    depths = []
    for k in tqdm(range(len(means))):
        mc = np.delete(means, [k])
        stdsc = np.delete(stds, [k])
        
        depths.append(    
            _normal_depth(means, stds, k, f)
        )
    
    return pd.DataFrame({
        'means': means,
        'stds': stds,
        'depths': depths
    })

def _poisson_containment(lambda_i: float, lambda_f: float, lambda_j: float, lim: int, quiet=False) -> float:
    s = 0
    
    for z in range(1, lim):
        ls = sum(poisson.pmf(k=l, mu=lambda_i) for l in range(z))
        ks = sum(poisson.pmf(k=k, mu=lambda_j) for k in range(z, lim))
        c = ls * ks * poisson.pmf(k=z, mu=lambda_f)
        
        s += c

    return s

def _poisson_depth(df: pd.DataFrame, curr: int, lim: int, to_compute=None):
    n, p = df.shape
    S_nj = 0
    cols = list(df.columns)
    cols.remove(curr)
    
    subseq = _subsequences(cols, 2)
    
    for i in tqdm(range(len(subseq))):
        sequence = subseq[i]
        i, j = sequence
        S_nj += sum(
            _poisson_containment(
                df.loc[k, i], 
                df.loc[k, curr], 
                df.loc[k, j],
                lim) 
            for k in df.index)
        
    return S_nj

def probabilistic_poisson_depth(df: pd.DataFrame, to_compute=None, lim=1000):
    n, p = df.shape
    depths = []
    for f in tqdm(df.columns):
        depths.append(1/binom(n, 2) * _poisson_depth(df, f, lim, to_compute))
    
    return pd.Series(index=df.columns, data=depths)

def _binom_containment(params: list, lim=1000) -> float:
    n_i, p_i, n, p, n_j, p_j = params
    
    s = 0
    for z in range(1, lim):
        t = (1-p)**(n-z)*p**z*(1-p_i)**n_i*(1-p_j)**(n_j-z)*p_j**z*\
             binom(n, z)*binom(n_j, z)*\
             ((1/(1-p_i))**n_i - (1-p_i)**(-1-z)*p_i**(1+z)*binom(n_i, 1+z)*\
             hyp2f1(1,1-n_i+z, 2+z, p_i/(p_i-1)))*hyp2f1(1, -n_j+z, 1+z, p_j/(p_j-1))
        
        if np.isnan(t):
            print(t)
            break
            
    return s