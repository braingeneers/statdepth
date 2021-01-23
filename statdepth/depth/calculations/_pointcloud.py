import numpy as np
import pandas as pd 
from typing import Union, List
from ._containment import _is_in_simplex
from ._helper import *
from scipy.special import binom 

__all__ = ['_pointwisedepth', '_samplepointwisedepth']

def _pointwisedepth(data: pd.DataFrame, points: Union[list, pd.Index]=None, containment='simplex') -> pd.Series:
    """
    Compute pointwise depth for n points in R^p, where data is an nxp matrix of points. If points is not None,
    only compute depth for the given points (should be a subset of data.index)
    
    Parameters:
    ----------
    data: pd.DataFrame
        n x d DataFrame, where we have n points in d dimensional space.
    points: list, pd.Index
        The particular points (indices) we would like to calculate band curve for. If None, we calculate depth for all points
    containment: str
        Definition of containment

    Returns:
    ----------
    pd.Series: Depth values for the given points with respect to the data. Index of Series are indices of points in the original data, and the values are the depths

    """
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
    elif containment == 'l1':
        return _L1_depth(data=data, points=points)
    else: # Probably will be more in the future 
        pass

    return pd.Series(index=to_compute, data=depths)

def _samplepointwisedepth(data: pd.DataFrame, points: pd.Index=None, K=2, containment='simplex'):
    """
    Compute sample pointwise depth for n points in R^p, where data is an nxp matrix of points. If points is not None,
    only compute depth for the given points (should be a subset of data.index)
    
    Parameters:
    ----------
    data: pd.DataFrame
        n x d DataFrame, where we have n points in d dimensional space.
    points: list, pd.Index
        The particular points (indices) we would like to calculate band curve for. If None, we calculate depth for all points.
    K=2:
        Number of blocks to compute sample depth with. 
    containment: str
        Definition of containment.

    Returns:
    ----------
    pd.Series: Depth values for the given points with respect to the data. Index of Series are indices of points in the original data, and the values are the depths

    """

    # If K=1, don't bother splitting the data. Just return regular depth. 
    if K == 1:
        return _pointwisedepth(data=data, points=points, containment=containment)

    n, d = data.shape 
    to_compute = data.index 
    depths = []
    if points is not None:
        to_compute = points 
    
    # K blocks of points (indices)
    ss = n // K 

    # Compute sample depth of each point, should be containment agnostic
    # Since the computation is being done in _pointwisedepth
    for time in to_compute:
        cd = []
        for _ in range(ss):
            sdata = data.sample(n=ss, axis=0)
            
            # If our current datapoint isnt in the sampled data, just append it since we need to sample it 
            # for _is_in_simplex()
            if not time in sdata.index:
                sdata = sdata.append(data.loc[time, :])
                
            cd.append(_pointwisedepth(data=sdata, points=[time], containment=containment))
        depths.append(np.mean(cd))
        
    return pd.Series(index=to_compute, data=depths)

def _L1_depth(data: pd.DataFrame, points: pd.Index=None):
    """
    Computes L1 data depth of the given points. 
    """
    n, d = data.shape
    depths = []

    to_compute = data.index
    idx = list(data.index)

    if points is not None:
        to_compute = points
    
    for point in to_compute:
        sum_e = 0
        vec = data.loc[point, :]

        # Ugly code, but faster than 
        # Computing data.drop(point).index each time :shrug:
        cidx = idx[:]
        cidx.remove(point)

        for other in cidx:
            sum_e += (data.loc[other, :] - vec) / np.linalg.norm(vec - data.loc[other, :])
            
        depths.append(np.linalg.norm(sum_e) / n)
        
    return pd.Series(index=to_compute, data=1-np.array(depths))

def _sample_L1_depth(data: pd.DataFrame, points: pd.Index=None, K=2):
    """
    Compute l1 depth using sampling
    """
    n, d = data.shape 
    depths = []
    to_compute = data.index 

    if points is not None:
        to_compute = points 

    ss = n // K

    for point in to_compute:
        sample = []
        c = data.copy()

        for _ in range(K):
            t = c.sample(n=ss, axis=0)
            c = c.drop(t.index)
            
            if point not in t.index:
                t = t.append(data.loc[point, :])
            
            sample.append(_L1_depth(data=t, points=[point]))
        depths.append(np.mean(sample))
        
    return pd.Series(index=to_compute, data=depths)
