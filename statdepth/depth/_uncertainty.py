import pandas as pd 
import numpy as np
from typing import List, Union

from scipy.special import erf 

from ._depthcalculations import _subsequences

def _norm_cdf(x: np.array, mu: float, sigma2: float):
    """
    Estimate the CDF at x for the normal distribution parametrized by mu and sigma^2
    """
    return 0.5 * (1+erf(x - mu) / (np.sqrt(sigma2) * np.sqrt(2)))

def _strict_uncertain_depth(data: pd.DataFrame, curve: Union[str, int], mu: float, sigma2: float):
    """
    data: pd.DataFrame
        An n x p matrix, where we have p real-valued functions collected at n discrete time intervals
    curve: str or int 
        Column of data for curve we'd like to calculate 
    mu: Mean of distribution

    """

    pass