from itertools import combinations
from typing import Callable

import numpy as np
import scipy as sp 
import pandas as pd 
from numba import jit
from scipy.special import comb, binom

from .containment import *

def banddepth(data: list, J=2, containment='r2', method='MBD'):
    """
    Calculate the band depth for a set of functional curves.

    Band depth is an order statistic for functional data (see `fboxplot`), with
    a higher band depth indicating larger "centrality".  In analog to scalar
    data, the functional curve with highest band depth is called the median
    curve, and the band made up from the first N/2 of N curves is the 50%
    central region.

    Parameters
    ----------
    data : list of DataFrames
        Functions to calculate band depth from. Each DataFrame should be an n x p matrix which is a function evaluated at all timepoints. 
    J: int (default=2)
        J parameter in the band depth calculation. J=3 can be computationally expensive for large datasets, and also does not have a closed form solution. 
    Containment: Callable or string
        Defines what containment means for the dataset. For functions from R-->R, we use the standard ordering on R. For higher dimensional spaces, we implement a simplex method. 
        A full list can be found in the README, as well as instructions on passing a custom definition for containment.  
    Returns
    -------
    ndarray
        Depth values for functional curves.
    """

    # Some common error handling 
    if J < 2:
        raise ValueError('Error: Parameter J must be greater than or equal to 2')
    
    if J >= len(data):
        raise ValueError('Error: Parameter J must be less than or equal to the number of observations len(df)')

    # Check if we're dealing with real-valued functions
    if len(data) == 0:
        data = data[0]

    n, p = data.shape
    rv = np.argsort(data, axis=0)
    rmat = np.argsort(rv, axis=0) + 1

    if J == 2:
        # This code is an explicit solution for J=2 and therefore cannot be generalized. 
        # band depth
        def _fbd2():
            down = np.min(rmat, axis=1) - 1
            up = n - np.max(rmat, axis=1)
            return (up * down + n - 1) / comb(n, 2)

        # modified band depth
        def _fmbd():
            down = rmat - 1
            up = n - rmat
            return ((np.sum(up * down, axis=1) / p) + n - 1) / comb(n, 2)

        if method == 'BD2':
            depth = _fbd2()
        elif method == 'MBD':
            depth = _fmbd()
        else:
            raise ValueError("Unknown input value for parameter `method`.")

        return depth
    else:
        band_depths = []
        for row in len(data):
            band_depths.append(_band_depth(data=data, curve=row, containment=containment, J=J))
    
        return band_depths

    return None

def subsequences(s, l):
    '''Returns a list of all possible subsequences of the given length from the given input list
    Parameters:
    -------------
    s: List to enumerate
    l: length of subsequences to find
    '''
    
    return sorted(set([i for i in combinations(s, l)]))


def _band_depth(data: pd.DataFrame, curve: int, containment='r2', J=2) -> float:
    """Calculates each band depth for a given curve in the dataset. Meant for J > 2, as J=2 has a closed solution. This function is wrapped in banddepth()
    
    Parameters:
    ----------------
    data: An n x p matrix where our rows come from R. Each observation should define a curve, in the functional sense. 

    curve: The particular function we would like to calculate band curve for. Given as a row of our original DataFrame. 

    containment: function that defines containment for the particular data. For example, in R^2 this would be a discrete subset 

    J=2: Defaulted to 2. 
    """
    
    # Initialize band depth, n
    band_depth = 0
    n = len(data)
    
    # Select our containment definition if it is in our pre-defined list 
    if containment == 'r2':
        containment = _r2_containment
    elif containment == 'r2_enum':
        containment = _r2_enum_containment
    elif containment == 'simplex':
        containment = _simplex_containment
    else:
        raise ValueError('Error: Unknown or unspecified definition of containment')

    # Reset index of our data 
    data = data.reset_index(drop=True)

    # Grab the data for our curve so numerical slicing is guaranteed to work
    curve_data = data.loc[curve, :]

    # Drop the curve (we don't want it used in defining our band/generalized band -- doesn't make sense)
    data = data.drop(curve)

    # Define our index to be the index of our dataset, excluding the last row (for indexing reasons)
    idx = list(data.index)
    
    # iterate from 2,...,J
    for j in range(2, J + 1):
        
        # Initialize S_n^(j) as defined in the paper
        S_nj = 0

        # Get a list of all possible subsequences of samples (rows)
        subseq = subsequences(idx, j)

        # Get generalized containment for this value of J=j
        for sequence in subseq:
            subseq_df = data.loc[list(sequence), :]
            
            S_nj += containment(data=subseq_df, curve=curve_data)

        band_depth += S_nj / binom(n, j)
    
    return band_depth
