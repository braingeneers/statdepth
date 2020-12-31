from itertools import combinations
from typing import Callable

import numpy as np
import scipy as sp 
import pandas as pd 
from numba import jit
from scipy.special import comb, binom

# Import all containment methods 
from ._containment import _r2_containment
from ._containment import _r2_enum_containment
from ._containment import _simplex_containment

def banddepth(data: list, J=2, containment='r2', relax=False):
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

    _handle_depth_errors(data=data, J=J)

    # If only one item in the list, it is the real-valued case
    if len(data) == 1:
        band_depths = []
        df = data[0]
        for col in df.columns:
            band_depths.append(_band_depth(data=df, curve=col, relax=relax, containment=containment, J=J))
        return band_depths
    else: 
        pass
    
    return None


def samplebanddepth(data: list, K: int, J=2, containment='r2', relax=False):
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
    _handle_depth_errors(data=data, J=J)

    
    pass

def _handle_depth_errors(data: list, J: int) -> None:
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
    # J = 0,1 doesn't make sense
    if J < 2:
        raise ValueError('Error: Parameter J must be greater than or equal to 2')

    # Make sure a non-empty list is passed
    if len(data) == 0:
        raise ValueError('Error: No data passed')

    # Make sure J < len(data) in the univariate and multivariate case
    if len(data) == 1 and J >= len(data[0]) or len(data) > 1 and J >= len(data):
        raise ValueError('Error: Parameter J must be less than or equal to the number of observations')


def subsequences(s, l):
    '''Returns a list of all possible subsequences of the given length from the given input list
    Parameters:
    ----------
    s: List to enumerate
    l: length of subsequences to find


    Returns:
    ----------
    list: List of subsequences
    '''
    
    return sorted(set([i for i in combinations(s, l)]))


def _band_depth(data: pd.DataFrame, curve: int, relax: bool, containment='r2', J=2) -> float:
    """Calculates each band depth for a given curve in the dataset. Meant for J > 2, as J=2 has a closed solution. This function is wrapped in banddepth()
    
    Parameters:
    ----------
    data: An n x p matrix where our rows come from R. Each observation should define a curve, in the functional sense. 

    curve: The particular function we would like to calculate band curve for. Given as a column of our original DataFrame, either column name or index. 

    containment: function that defines containment for the particular data. For example, in R^2 this would be a discrete subset 

    J=2: Defaulted to 2. 

    Returns:
    ----------
    float: Depth value for the given function with respect to the data passed 

    """
    
    # Initialize band depth, n (number of curves)
    band_depth = 0
    n = data.shape[1]
    
    # Select our containment definition if it is in our pre-defined list
    if containment == 'r2':
        cdef = _r2_containment
    elif containment == 'r2_enum':
        cdef = _r2_enum_containment
    elif containment == 'simplex':
        cdef = _simplex_containment
    else:
        # TODO: Allow user to pass in custom definition of containment
        raise ValueError('Error: Unknown or unspecified definition of containment')

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

            S_nj += cdef(data=subseq_df, curve=curve_data, relax=relax)

        band_depth += S_nj / binom(n, j)
    
    return band_depth