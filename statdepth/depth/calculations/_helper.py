import pandas as pd 
import numpy as np 
from typing import List, Union, Callable
from itertools import combinations
import numpy as np
import pandas as pd 

import scipy.stats as stats
from scipy.stats import norm
from scipy.special import gamma, gammaincc

# Custom error class for anytime there is going to be some degeneracy with depth calculation (i.e. degenerate simplices)
class DepthDegeneracy(Exception):
    pass

def _subsequences(s: list, l: int) -> list:
    """
    Returns a list of all possible subsequences of length l from the given input list

    Parameters:
    ----------
    s: list
        List to enumerate
    l: int
        Length of subsequences to compute

    Returns:
    ----------
    list: List of subsequences
    """

    return list(set(combinations(s, l)))

def _handle_depth_errors(
    data: List[pd.DataFrame], 
    J: int, 
    containment: Union[Callable, str], 
    relax: bool, 
    deep_check: bool
) -> None:
    """
    Handles errors in band depth methods.

    Parameters:
    ----------
    data: list
        Functions to calculate band depth from
    J: int
        Parameter J for computing band depth
    containment:
        Definition of containment, either a string or bool

    Returns:
    ----------
    None: Nothing is returned, but exceptions are raised if needed
    """
    
    # Type checking for all variables
    if not isinstance(data, list):
        raise ValueError('data must be passed as a list.')

    if not isinstance(J, int):
        raise ValueError('J must be an integer.')

    if not (isinstance(containment, str) or isinstance(containment, Callable)):
        raise ValueError('containment must be of type str or Callable.')

    if not isinstance(deep_check, bool):
        raise ValueError('deep_check must be of type bool.')

    if not isinstance(relax, bool):
        raise ValueError('relax must be of type bool')
    
    # J = 0,1 doesn't make sense
    if J < 2:
        raise ValueError('Parameter J must be greater than or equal to 2.')

    # Make sure a non-empty list is passed
    if len(data) == 0:
        raise ValueError('No data passed.')

    # Make sure J < len(data) in the univariate and multivariate case
    if len(data) == 1 and J >= len(data[0]) or len(data) > 1 and J >= len(data):
        raise ValueError('Parameter J must be less than the number of observations.')
    
    if len(data) > 1 and containment == 'r2':
        raise ValueError('containment argument \'r2\' is invalid for multivariate data. Use one of [\'r2_enum\', \'simplex \'] or a passed containment method. ')

    # If there is not at least d + 2 functions for our d dimensional data, then for each function
    # We won't have d + 1 vertices to construct a simplex, which means every simplex will be at least one dimensional degenerate
    # Therefore we say depth is not well defined and error
    if isinstance(data, list) and len(data) < data[0].shape[1] + 2 and containment == 'simplex':
        raise DepthDegeneracy(f'Error: Need at least {len(data)} functions to form non-degenerate simplices in {data[0].shape + 2} dimensional space. Only have {len(data)}.')

    if deep_check:
        # Check dtypes of all columns over all DataFrames. 
        # Optional because this might be computationally expensive for very large datasets. 
        indices = []
        for df in data:
            indices.append(df.index)
            df = df.infer_objects()
            for col in df:
                if not np.issubdtype(df[col].dtype, np.number):
                    raise ValueError('DataFrame must only contain numeric dtypes.')
        # Check that all indices are the same.
        if not all([all(indices[0] == i) for i in indices]):
            raise ValueError('DataFrames indices must be the same')

def gammainc(a, x):
    return gamma(a) * gammaincc(a, x)

def normcdf(x, mu, sigma):
    return norm(loc=mu, scale=sigma).cdf(x)

