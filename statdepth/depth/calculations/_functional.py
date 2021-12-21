import random 
import warnings
from itertools import combinations
from typing import Callable, List, Union
from tqdm import tqdm
import numpy as np
import scipy as sp 
import pandas as pd
from scipy.special import comb, binom

from ._containment import _r2_containment, _r2_enum_containment, _simplex_containment, _select_containment, _is_valid_containment, _is_in_simplex
from ._helper import *
from ._helper import _subsequences, _handle_depth_errors

__all__ = ['_functionaldepth', '_samplefunctionaldepth']

def _functionaldepth(
    data: List[pd.DataFrame], 
    to_compute: Union[list, pd.Index]=None,
    J=2, 
    containment='r2', 
    relax=False, 
    deep_check=False,
    quiet=True,
) -> Union[pd.Series, pd.DataFrame]:
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
    to_compute: list of DataFrame, DataFrame, or pd.Index
        Data to compute depths of. If None, compute depth of all samples. 
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
        if containment == 'simplex':
            cdef = _r2_containment

            
        band_depths = []
        df = data[0]
        cols = df.columns

        if to_compute is not None:
            cols = to_compute
        
        # Calculate band depth for each sample (column)
        for col in tqdm(cols, disable=quiet):
            band_depths.append(_univariate_band_depth(data=df, curve=col, relax=relax, containment=cdef, J=J))

        # Return a series indexed by our samples
        depths = pd.Series(index=cols, data=band_depths)
    else: # Multivariate case
        if containment == 'simplex':
            depths = []

            # Get 'index' of functions, which are just their indices in the list
            f = [i for i in range(len(data))]

            if to_compute is not None:
                f = to_compute

            data_to_compute = [data[i] for i in f]
            
            # Compute band depth for each function (DataFrame)
            for cdf in tqdm(data_to_compute, disable=quiet):
                cdata = [df for df in data if df is not cdf]
                depths.append(_simplex_depth(data=cdata, curve=cdf, J=J, relax=relax))
            depths = pd.Series(index=f, data=depths)
    
    return depths

def _samplefunctionaldepth(
    data: List[pd.DataFrame], 
    K: int, 
    to_compute: Union[list, pd.Index]=None, 
    J=2, 
    containment='r2', 
    relax=False, 
    deep_check=False,
    quiet=True
) -> Union[pd.Series, pd.DataFrame]:

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
    to_compute: list of DataFrame, DataFrame, or pd.Index
        Data to compute depths of. If None, compute depth of all samples. 
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
    # This has a much lower asymtotic complexity, but a much much higher space complexity,
    # Since to sample from n functions (one for each curve, without replacement), we require copying the data
    # n - 1 times
    
    samples = []
    depths = []

    # Handle common errros
    _handle_depth_errors(data=data, J=J, containment=containment, relax=relax, deep_check=deep_check)

    cdef = _select_containment(containment=containment)
    
    # Univariate case
    if len(data) == 1:
        df = data[0]
        cols = df.columns

        if to_compute is not None:
            cols = to_compute

        orig = df.loc[:, cols]
        ss = df.shape[1] // K

        if ss == 0:
            raise DepthDegeneracy(f'Block size {K} is too large, not enough functions to sample.')
        
        if containment == 'simplex':
            cdef = _r2_containment

        # Iterate over curves (columns)
        for col in tqdm(orig.columns, disable=quiet):
            depths = []

            # For each curve, compute sample band depth using K blocks of size ~len(df)/K
            for _ in range(K):
                t = df.sample(n=ss, axis=1)
                df = df.drop(t.columns, axis=1)
                t.loc[:, col] = orig.loc[:, col]
                depths.append(_univariate_band_depth(data=t, curve=col, relax=relax, containment=cdef, J=J))
            
            # Collect these estimates
            samples.append(np.mean(depths))
            df = orig.copy()

        # Average them and return a series
        samples = pd.Series(index=df.columns, data=samples)
    else:
        # TODO: Multivariate case: partition list of shuffled DataFrames, compute band depth w.r.t those, average
        shuffled = data.copy()
        random.shuffle(shuffled)
        ss = len(data) // K 

        for _ in tqdm(range(K), disable=quiet):
            pass
        
    return samples
        
def _univariate_band_depth(
    data: pd.DataFrame, 
    curve: Union[str, int], 
    relax: bool, 
    containment: Callable, 
    J=2
) -> float:
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
    """
    Calculates simplex depth of the given curve with respect to the given data

    Parameters:
    ----------
    data: list
        List of n x d matrices, where each matrix is a function observated at n discrete time points with d features
    curve: pd.DataFrame
        The particular function we would like to calculate band curve for
    J=2: int
        Parameter J in band depth. Defaulted to 2. 
    relax: bool
        If True, use a strict definition of containment, else use containment defined by the proportion of time the curve is in the band. 

    Returns:
    ----------
    float: Depth value for the given function with respect to the data passed 

    """
    l, d = data[0].shape
    n = len(data)
    depth = 0
    
    subseq = _subsequences([i for i in range(n)], d + 1)

    for seq in subseq:
        cdata = [data[i] for i in seq]
        depth += _simplex_containment(data=cdata, curve=curve, relax=relax)
    return depth / binom(n, d + 1)
