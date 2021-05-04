import pandas as pd 
import numpy as np
from typing import List, Union
from scipy.special import erf, binom
from scipy.integrate import tplquad, quad
import scipy.stats as stats
from scipy.stats import norm
from scipy.special import gamma, gammaincc, factorial
from bigfloat import BigFloat
from tqdm import tqdm
from numpy import exp
from scipy.special import hyp2f1, binom

from ._functional import _subsequences
from ._helper import *


__all__ = ['probabilistic_normal_depth', 'probabilistic_poisson_depth']

def normcdf(x, mu, sigma):
    return norm(loc=mu, scale=sigma).cdf(x)

def f_long_norm(z, parameters: list):
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

def probabilistic_normal_depth(means, stds, f=f_long_norm):
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

def gammainc(a, x):
    return gamma(a) * gammaincc(a, x)

def f_poisson(lambda_i: float, lambda_f: float, lambda_j: float, lim=1000, quiet=False) -> float:
    '''
    Parameters:
    
    lambda_i: 
        mean of f_i (lower function)
    lambda_f: 
        mean of f (function to calculate probabilistic containment)
    lambda_j: 
        mean of f_j (upper function)
    lim=10000: Upper bound of discrete infinite sum
    
    Returns:
    
    float: Probability of containment
    '''
    
    s = 0
    
    for z in range(1, lim):
        num = BigFloat(np.power(lambda_f, z)*(gamma(z)-gammainc(z, lambda_j))*gammainc(1+z, lambda_i))
        denom = BigFloat(factorial(z)*gamma(z)*gamma(1+z))
        
        if num == 0 or denom == 0 and not quiet:
            break
        
        d = BigFloat(num / denom)
        
        s += d
        
    return exp(-lambda_f) * s

def _poisson_depth(means: list, curr: int, to_compute=None):
    n = len(means)
    cols = list(range(n))
    cols.remove(curr)
    S_nj = 0
    subseq = _subsequences(cols, 2)

    for i in tqdm(range(len(subseq))):
        sequence = subseq[i]
        i, j = sequence
        lambda_i, lambda_f, lambda_j = means[i], means[curr], means[j]
        
        S_nj += f_poisson(lambda_i, lambda_f, lambda_j)
        
    return S_nj / binom(n, 2)

def probabilistic_poisson_depth(means: list, to_compute=None):
    depths = []
    for i in tqdm(range(len(means))):
        depths.append(_poisson_depth(means, i, to_compute=to_compute))
    
    return pd.DataFrame({'means': means, 'depths': depths})

def f_binom(params: list, lim=1000) -> float:
    n_i, p_i, n, p, n_j, p_j = params
    
    s = 0
    for z in range(1, lim):
        print(f'z is {z}')
        t = (1-p)**(n-z)*p**z*(1-p_i)**n_i*(1-p_j)**(n_j-z)*p_j**z*\
             binom(n, z)*binom(n_j, z)*\
             ((1/(1-p_i))**n_i - (1-p_i)**(-1-z)*p_i**(1+z)*binom(n_i, 1+z)*\
             hyp2f1(1,1-n_i+z, 2+z, p_i/(p_i-1)))*hyp2f1(1, -n_j+z, 1+z, p_j/(p_j-1))
        
        if np.isnan(t):
            print(t)
            break
            
    return s