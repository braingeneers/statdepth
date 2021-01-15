import pandas as pd 
import numpy as np
from typing import List, Union

from scipy.special import erf, binom

from ._depthcalculations import _subsequences

def _norm_cdf(x: np.array, mu: float, sigma2: float):
    """
    Estimate the CDF at x for the normal distribution parametrized by mu and sigma^2
    """
    return 0.5 * (1+erf(x - mu) / (np.sqrt(sigma2) * np.sqrt(2)))

def _strict_uncertain_depth(data: pd.DataFrame, curve: Union[str, int], sigma2: pd.DataFrame, J: int=2):
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

    depth = 0
    # Drop our current curve from our data
    if curve in data.columns:   
        data = data.drop(curve, axis=1)

    subseq = _subsequences(data.columns, J)
    for seq in subseq:
        d = 1
        f1 = seq[0]
        f2 = seq[1]
        for time in data.index:
            d *= _norm_cdf(data.loc[time, f1], data.loc[time, f1], sigma2.loc[time, f1]) + _norm_cdf(data.loc[time, f2], data.loc[time, f2], sigma2.loc[time, f2]) - 2 * _norm_cdf(data.loc[time, f1], data.loc[time, f1], sigma2.loc[time, f1]) * _norm_cdf(data.loc[time, f2], data.loc[time, f2], sigma2.loc[time, f2])
        depth += d

    return depth / binom(data.shape[1], J)

def _gen_uncertain_depth(data: pd.DataFrame, curve: Union[str, int], sigma2: pd.DataFrame, J: int=2):
    pass

def uncertain_depth(data: pd.DataFrame, sigma2: pd.DataFrame, J: int=2, strict=True):
    """
    Calculate probabilistic depth for each function (column) in the given data. 
    """
    depths = []

    for col in data:
        if strict:
            depths.append(_strict_uncertain_depth(data, col, sigma2, J))
        else:
            depths.append(_gen_uncertain_depth(data, col, sigma2, J))

    return pd.Series(index=data.columns, data=depths)