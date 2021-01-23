import pandas as pd 
import numpy as np
from typing import List, Union
from scipy.special import erf, binom

from ._functional import _subsequences

from ._helper import *

def _norm_cdf(x: np.array, mu: float, sigma: float):
    """
    Estimate the CDF at x for the normal distribution parametrized by mu and sigma^2
    """
    return 0.5 * (1 + erf(x - mu) / (sigma * np.sqrt(2)))

def _uncertain_depth_univariate(data: pd.DataFrame, curve: Union[str, int], sigma2: pd.DataFrame, J: int=2, relax=False):
    """
    Calculate uncertain depth for the given curve, assuming each entry in our data comes from a normal distribution 
    where the mean is the observed value and the variance is the corresponding entry in sigma2.

    Parameters:
    -----------
    data: pd.DataFrame
        An n x p matrix, where we have p real-valued functions collected at n discrete time intervals
    curve: int or str
        Column (function) to calculate depth for 
    sigma2: pd.DataFrame
        An n x p matrix where each entry is the variance of the distribution at that entry

    Returns:
    ----------
    pd.Series: Depth values for each function (column)
    """
    # Dont use f1 for both x and mu, use curve because that's what we're interested in 

    # if relax:
    #     sym = '*'
    # else:
    #     sym = '+'

    n, p = data.shape
    depth = 0
    sigma = sigma2.pow(.5)

    # Drop our current curve from our data
    if curve in data.columns:   
        data = data.drop(curve, axis=1)

    subseq = _subsequences(data.columns, J)
    if J == 2:
        for seq in subseq:
            d = 1
            f1 = seq[0]
            f2 = seq[1]
            for time in data.index:
                p1 = _norm_cdf(data.loc[time, f1], data.loc[time, curve], sigma.loc[time, f1])
                p2 = _norm_cdf(data.loc[time, f2], data.loc[time, curve], sigma.loc[time, f2])

                if relax:
                    d *= p1 + p2 - 2 * p1 * p2
                else: 
                    d += p1 + p2 - 2 * p1 * p2

            depth += d
    elif J == 3:
        for seq in subseq:
            d = 1
            f1, f_2, f_3 = seq[0], seq[1], seq[2]

            for time in data.index:
                p1 = _norm_cdf(data.loc[time, f1], data.loc[time, curve], sigma.loc[time, f1])
                p2 = _norm_cdf(data.loc[time, f2], data.loc[time, curve], sigma.loc[time, f2])
                p3 = _norm_cdf(data.loc[time, f3], data.loc[time, curve], sigma.loc[time, f3])
                
                if relax:
                    d *= p1 + p2 + p3 - p1 * p2 - p2*p3 - p1*p3
                else:
                    d += p1 + p2 + p3 - p1 * p2 - p2*p3 - p1*p3

            depth += d
    else: # Handle J=4 later, not sure about computation
        pass

    return depth / binom(data.shape[1], J) if relax else depth / binom(data.shape[1], J) * n / p # Because in the nonrelax case we are summing 1/|D| n times

# def _gen_uncertain_depth(data: pd.DataFrame, curve: Union[str, int], sigma2: pd.DataFrame, J: int=2):
#     n, p = data.shape
#     depth = 0
#     sigma = sigma2.pow(.5)

#     # Drop our current curve from our data
#     if curve in data.columns:   
#         data = data.drop(curve, axis=1)

#     subseq = _subsequences(data.columns, J)
#     for seq in subseq:
#         d = 1
#         f1 = seq[0]
#         f2 = seq[1]
#         for time in data.index:
#             p1 = _norm_cdf(data.loc[time, f1], data.loc[time, f1], sigma.loc[time, f1])
#             p2 = _norm_cdf(data.loc[time, f2], data.loc[time, f2], sigma.loc[time, f2])

#             d += p1 + p2 - 2 * p1 * p2
#         depth += d / p

#     return depth / binom(p, J)




def _uncertain_depth(data: pd.DataFrame, sigma2: pd.DataFrame, J: int=2, relax=True):
    """
    Calculate probabilistic depth for each function (column) in the given data. 
    """
    depths = []

    for col in data:
        depths.append(_uncertain_depth_univariate(data=data, curve=col, sigma2=sigma2, J=J))
    return pd.Series(index=data.columns, data=depths)

def _sampleuncertaindepth(data: pd.DataFrame, sigma2: pd.DataFrame, K: int=2, J: int=2, relax=True):
    pass