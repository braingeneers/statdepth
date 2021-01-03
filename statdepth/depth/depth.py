import random 

from itertools import combinations

from typing import Callable
from typing import List 
from typing import Union 

import numpy as np
import scipy as sp 
import pandas as pd 

from numba import jit
from scipy.special import comb, binom

from ._containment import _r2_containment
from ._containment import _r2_enum_containment
from ._containment import _simplex_containment
from ._containment import _select_containment
from ._containment import _is_valid_containment

def banddepth(data: List[pd.DataFrame], J=2, containment='r2', relax=False, deep_check=False) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate the band depth for a set of functional curves.

    Band depth is an order statistic for functional data (see `fboxplot`), with
    a higher band depth indicating larger "centrality".  In analog to scalar
    data, the functional curve with highest band depth is called the median
    curve, and the band made up from the first N/2 of N curves is the 50%
    central region.

    Parameters:
    ----------
    data : list of DataFrames
        Functions to calculate band depth from
    J: int (default=2)
        J parameter in the band depth calculation. J=3 can be computationally expensive for large datasets, and also does not have a closed form solution. 
    containment: Callable or string (default='r2')
        Defines what containment means for the dataset. For functions from R-->R, we use the standard ordering on R. For higher dimensional spaces, we implement a simplex method. 
        A full list can be found in the README, as well as instructions on passing a custom definition for containment.  
    relax: bool
        If True, use a strict definition of containment, else use containment defined by the proportion of time the curve is in the band. 
    deep_check: bool (default=False)
        If True, perform a more extensive error checking routine. Optional because it can be computationally expensive for large datasets. 

    Returns:
    ----------
    pd.Series: Depth values for each function.
    """

    # Handle common errors
    _handle_depth_errors(data=data, J=J, containment=containment, relax=relax, deep_check=deep_check)

    # Select containment definition
    cdef = _select_containment(containment=containment)

    # If only one item in the list, it is the real-valued case
    if len(data) == 1:
        band_depths = []
        df = data[0]
        for col in df.columns:
            band_depths.append(_univariate_band_depth(data=df, curve=df[col], relax=relax, containment=cdef, J=J))
        return pd.Series(index=df.columns, data=band_depths)
    else: 
        # Multivariate case
        pass
    
    return None


def samplebanddepth(data: List[pd.DataFrame], K: int, J=2, containment='r2', relax=False, deep_check=False) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate the sample band depth for a set of functional curves.

    This is done by
    1. Splitting the data of n curves into K blocks of size ~n/K
    2. Computing band depth with respect to each block
    3. Returning the average of these

    For K << n, this should approximate the band depth well. 

    Parameters:
    ----------
    data: list of DataFrames
        Functions to calculate band depth from
    K: int 
        Computes band depth by averaging band depth across K blocks of our original curves 
    J: int (default=2)
        J parameter in the band depth calculation. J=3 can be computationally expensive for large datasets, and also does not have a closed form solution. 
    containment: Callable or string (default='r2')
        Defines what containment means for the dataset.  
    relax: bool
        If True, use a strict definition of containment, else use containment defined by the proportion of time the curve is in the band. 
    deep_check: bool (default=False)
        If True, perform a more extensive error checking routine. Optional because it can be computationally expensive for large datasets. 

    Returns:
    ----------
    pd.Series: Depth values for each function.
    """
    samples = []
    depths = []

    # Handle common errros
    _handle_depth_errors(data=data, J=J, containment=containment, relax=relax, deep_check=deep_check)

    cdef = _select_containment(containment=containment)
    
    # Univariate case
    if len(data) == 1:
        df = data[0]
        orig = df.copy()
        ss = df.shape[1] // K
        
        for col in orig.columns:
            depths = []

            for i in range(K):
                t = df.sample(n=ss, axis=1)
                df = df.drop(t.columns, axis=1)
                depths.append(_univariate_band_depth(data=t, curve=orig[col], relax=relax, containment=cdef, J=J))
            
            samples.append(depths)
            df = orig.copy()

        samples = pd.Series(index=df.columns, data=[np.mean(i) for i in samples])

    else:
        # Multivariate case: partition list of DataFrames randomly, compute band depth w.r.t those
        shuffled = data.copy()
        random.shuffle(shuffled)
        ss = len(data) // K 

        for _ in range(K):
            pass
        
    return samples

def _handle_depth_errors(data: List[pd.DataFrame], J: int, containment: Union[Callable, str], relax: bool, deep_check: bool) -> None:
    '''
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
    '''
    
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

    if deep_check:
        # Check dtypes of all columns over all DataFrames. Optional because this might be expensive
        for df in data:
            df = pd.to_numeric(df)
            for col in df:
                if not np.issubdtype(df[col].dtype, np.number):
                    raise ValueError('DataFrame must only contain numeric dtypes.')
    

def _subsequences(s: list, l: int) -> list:
    '''Returns a list of all possible subsequences of length l from the given input list

    Parameters:
    ----------
    s: list
        List to enumerate
    l: int
        Length of subsequences to compute


    Returns:
    ----------
    list: List of subsequences
    '''
    
    return sorted(set([i for i in combinations(s, l)]))


def _univariate_band_depth(data: pd.DataFrame, curve: pd.Series, relax: bool, containment: Callable, J=2) -> float:
    """Calculates each band depth for a given curve in the dataset. Meant for J > 2, as J=2 has a closed solution. This function is wrapped in banddepth()
    
    Parameters:
    ----------
    data: pd.DataFrame
        An n x p matrix where our rows come from R. Each observation should define a curve, in the functional sense. 
    curve: int
        The particular function we would like to calculate band curve for. Given as a column of our original DataFrame, either column name or index. 
    containment: Callable or str
        Function that defines containment for the particular data. For example, in R^2 this would be a discrete subset 
    J=2: int
        Parameter J in band depth. Defaulted to 2. 

    Returns:
    ----------
    float: Depth value for the given function with respect to the data passed 

    """
    
    # Initialize band depth, n (number of curves)
    band_depth = 0
    n = data.shape[1]

    # Define our index to be the columns of our dataset, excluding the last row (for indexing reasons)
    cols = list(data.columns)
    
    # Compute band depth
    for j in range(2, J + 1):
        
        S_nj = 0

        # Get a list of all possible subsequences of samples (cols)
        subseq = _subsequences(cols, j)

        # Iterate over all subsequences
        for sequence in subseq:
            # Grab data for the current subsequence
            subseq_df = data.loc[:, list(sequence)]

            # Check containment for the current subsequence
            S_nj += containment(data=subseq_df, curve=curve, relax=relax)

        band_depth += S_nj / binom(n, j)
    
    return band_depth