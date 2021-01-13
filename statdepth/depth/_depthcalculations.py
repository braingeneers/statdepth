import random 
import warnings
from itertools import combinations
from typing import Callable, List, Union

import numpy as np
import scipy as sp 
import pandas as pd 

from numba import jit
from scipy.special import comb, binom

from ._containment import _r2_containment, _r2_enum_containment, _simplex_containment, _select_containment, _is_valid_containment, _is_in_simplex

# Custom error class for anytime there is going to be some degeneracy with depth calculation (i.e. k degenerate simplices)
class DepthDegeneracy(Exception):
    pass

def _banddepth(data: List[pd.DataFrame], J=2, containment='r2', relax=False, deep_check=False) -> Union[pd.Series, pd.DataFrame]:
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
    pd.Series, pd.DataFrame: Depth values for each function.
    """

    # Handle common errors
    _handle_depth_errors(data=data, J=J, containment=containment, relax=relax, deep_check=deep_check)

    # Select containment definition
    cdef = _select_containment(containment=containment)

    # If only one item in the list, it is the real-valued case (by assumption)
    if len(data) == 1:

        # In the case of simplex containment in R2 for f: R --> R, the simplices of 2 points degenerate to 
        # intervals, which is equivalent to the standard containment on R2
        if containment == 'simplex':
            warnings.warn('Simplex containment for real-valued functions is degenerate, and equivalent to interval containment. Falling back to interval containment.')
            cdef = _r2_containment

        band_depths = []
        df = data[0]

        # Calculate band depth for each sample (column)
        for col in df.columns:
            band_depths.append(_univariate_band_depth(data=df, curve=col, relax=relax, containment=cdef, J=J))

        # Return a series indexed by our samples
        return pd.Series(index=df.columns, data=band_depths)
    else:
        if containment == 'simplex':
            depths = []

            # Get 'columns' of functions, which are just their indices
            f = [i for i in range(len(data))]

            # Compute band depth for each function (DataFrame)
            for cdf in data:
                cdata = [df for df in data if df is not cdf]
                depths.append(_simplex_depth(data=cdata, curve=cdf, J=J, relax=relax))
                
            return pd.Series(index=f, data=depths)
    
    return None

def _samplebanddepth(data: List[pd.DataFrame], K: int, J=2, containment='r2', relax=False, deep_check=False) -> Union[pd.Series, pd.DataFrame]:
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
        
        # Iterate over curves (columns)
        for col in orig.columns:
            depths = []

            # For each curve, compute sample band depth using K blocks of size ~len(df)/K
            for _ in range(K):
                t = df.sample(n=ss, axis=1)

                df = df.drop(t.columns, axis=1)

                t[col] = orig[col]

                depths.append(_univariate_band_depth(data=t, curve=col, relax=relax, containment=cdef, J=J))
            
            # Collect these estimates
            samples.append(depths)
            df = orig.copy()

        # Average them and return a series
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

    # If there is not at least d + 2 functions for our d dimensional data, then for each function
    # We won't have d + 1 vertices to construct a simplex, which means every simplex will be at least one dimensional degenerate
    # Therefore we say depth is not well defined and error
    if isinstance(data, list) and len(data) < data[0].shape[1] + 2 and containment == _simplex_containment:
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
        
def _subsequences(s: list, l: int) -> list:
    '''
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
    '''
    
    return sorted(set(combinations(s, l)))


def _univariate_band_depth(data: pd.DataFrame, curve: Union[str, int], relax: bool, containment: Callable, J=2) -> float:
    """
    Calculates each band depth for a given curve in the dataset. Meant for J > 2, as J=2 has a closed solution. This function is wrapped in banddepth()
    
    Parameters:
    ----------
    data: pd.DataFrame
        An n x p matrix where our rows come from R. Each observation should define a curve, in the functional sense. 
    curve: int
        The particular function we would like to calculate band curve for. Given as a column of our original DataFrame. 
    relax: bool
        If True, use a strict definition of containment, else use containment defined by the proportion of time the curve is in the band. 
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
    
    # get curve series 
    curvedata = data.loc[:, curve]

    # Drop the curve we're calculating band depth for
    data = data.drop(curve, axis=1)

    # Compute band depth
    for j in range(2, J + 1):
        
        S_nj = 0

        # Get a list of all possible subsequences of samples (cols)
        subseq = _subsequences(list(data.columns), j)

        # Iterate over all subsequences
        for sequence in subseq:
            # Grab data for the current subsequence
            subseq_df = data.loc[:, sequence]

            # Check containment for the data given the current subsequence
            S_nj += containment(data=subseq_df, curve=curvedata, relax=relax)

        band_depth += S_nj / binom(n, j)
    
    return band_depth

def _simplex_depth(data: List[pd.DataFrame], curve: pd.DataFrame, J=2, relax=False):
    """Calculates simplex depth of the given curve with respect to data"""
    l, d = data[0].shape
    n = len(data)
    depth = 0
    
    for j in range(2, J + 1):
        S_nj = 0
        subseq = _subsequences([i for i in range(n)], d + 1)
        
        for seq in subseq:
            cdata = [data[i] for i in seq]
            S_nj += _simplex_containment(data=data, curve=curve, relax=relax)
        
        depth += S_nj / binom(n, d + 1)

    return depth

def _pointwisedepth(data: pd.DataFrame, points: pd.Index=None, J=2, containment='simplex'):
    """Compute pointwise depth for n points in R^p, where data is an nxp matrix of points. If points is not None,
    only compute depth for the given points (should be a subset of data.index)"""
    n, d = data.shape
    depths = []
    to_compute = data.index

    if points is not None:
        to_compute = points

    if containment == 'simplex':
        for time in to_compute:
            S_nj = 0
            
            point = data.loc[time, :]
            
            subseq = _subsequences(list(data.drop(time, axis=0).index), d + 1)

            for seq in subseq:
                S_nj += _is_in_simplex(simplex_points=
                        np.array(data.loc[seq, :]), point=np.array(point))
                
            depths.append(S_nj / binom(n, d + 1))
        
    return pd.Series(index=to_compute, data=depths)

# TODO: This function. Just split data.index into K blocks, compute w.r.t those, average
def _samplepointwisedepth(data: pd.DataFrame, points: pd.Index=None, K=2, J=2, containment='simplex'):
    pass