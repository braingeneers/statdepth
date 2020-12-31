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

def banddepth(data: List[pd.DataFrame], J=2, containment='r2', relax=False, deep_check=False):
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
    Containment: Callable or string
        Defines what containment means for the dataset. For functions from R-->R, we use the standard ordering on R. For higher dimensional spaces, we implement a simplex method. 
        A full list can be found in the README, as well as instructions on passing a custom definition for containment.  
    
    Returns:
    ----------
    list: Depth values for each row or observation.
    """

    # Handle common errors
    _handle_depth_errors(data=data, J=J, deep_check=deep_check)

    # Select containment definition
    cdef = _select_containment(containment=containment)

    # If only one item in the list, it is the real-valued case
    if len(data) == 1:
        band_depths = []
        df = data[0]
        for col in df.columns:
            band_depths.append(_univariate_band_depth(data=df, curve=col, relax=relax, containment=cdef, J=J))
        return band_depths
    else: 
        # Multivariate case
        pass
    
    return None


def samplebanddepth(data: List[pd.DataFrame], K: int, J=2, containment='r2', relax=False, deep_check=False):
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
    Containment: Callable or string
        Defines what containment means for the dataset.  
    
    Returns:
    ----------
    list: Depth values for each row or observation.
    """
    samples = []
    depths = []

    # Handle common errros
    _handle_depth_errors(data=data, J=J, deep_check=deep_check)

    cdef = _select_containment(containment=containment)
    
    # Univariate case
    if len(data) == 1:
        df = data[0]
        ss = df.shape[0] // K
        
        # Compute band depths with respect to each sample
        for _ in range(K):
            t = df.sample(n=ss)
            df = df.drop(t.index)
            samples.append(banddepth(data=[t], J=J, containment=containment, relax=relax, deep_check=deep_check))
        
        # Average them
        for k in range(df.shape[1]):
            t = [samples[i][k] for i in range(K)]
            depths.append(np.mean(t))

    else:
        # Multivariate case: partition list of DataFrames randomly, compute band depth w.r.t those
        shuffled = data.copy()
        random.shuffle(shuffled)
        ss = len(data) // K 

        for _ in range(K):
            pass
        
    return depths

def _handle_depth_errors(data: List[pd.DataFrame], J: int, deep_check=False) -> None:
    '''
    Handle errors in band depth methods 

    Parameters:
    ----------
    data: list
        Functions to calculate band depth from
    J: int
        Parameter J for computing band depth
    
    Returns:
    ----------
    None: Nothing is returned, but exceptions are raised if needed
    '''

    #TODO: Make deep_check a parameter in higher level functions
    # J = 0,1 doesn't make sense
    if J < 2:
        raise ValueError('Error: Parameter J must be greater than or equal to 2')

    # Make sure a non-empty list is passed
    if len(data) == 0:
        raise ValueError('Error: No data passed')

    # Make sure J < len(data) in the univariate and multivariate case
    if len(data) == 1 and J >= len(data[0]) or len(data) > 1 and J >= len(data):
        raise ValueError('Error: Parameter J must be less than the number of observations')
    
    # Check dtypes of all columns over all DataFrames. Optional because this might be expensive
    if deep_check:
        for df in data:
            df = pd.to_numeric(df)
            for col in df:
                if not np.issubdtype(df[col].dtype, np.number):
                    raise ValueError('Error: DataFrame must only contain numeric dtypes')

def subsequences(s: list, l: int):
    '''Returns a list of all possible subsequences of the given length from the given input list
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


def _univariate_band_depth(data: pd.DataFrame, curve: int, relax: bool, containment: Callable, J=2) -> float:
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
    
    # Grab the data for our curve so numerical slicing is guaranteed to work
    curve_data = data.loc[:, curve]

    # Drop the curve (we don't want it used in defining our band/generalized band -- doesn't make sense)
    data = data.drop(curve, axis=1)

    # Define our index to be the columns of our dataset, excluding the last row (for indexing reasons)
    idx = list(data.columns)
    
    # Compute band depth
    for j in range(2, J + 1):
        
        # Initialize S_n^(j) as defined in the paper
        S_nj = 0

        # Get a list of all possible subsequences of samples (cols)
        subseq = subsequences(idx, j)

        # Get generalized containment for this value of J=j
        for sequence in subseq:
            subseq_df = data.loc[:, list(sequence)]

            S_nj += containment(data=subseq_df, curve=curve_data, relax=relax)

        band_depth += S_nj / binom(n, j)
    
    return band_depth