import numpy as np
import pandas as pd 
from typing import Union, List
from scipy.special import binom 
from scipy.spatial import ConvexHull
from tqdm import tqdm

from ._containment import _is_in_simplex
from ._helper import *
from . import _helper

__all__ = ['_pointwisedepth', '_samplepointwisedepth']

def _pointwisedepth(
    data: pd.DataFrame, 
    to_compute: Union[list, pd.Index]=None, 
    containment='simplex', 
    quiet=True
) -> pd.Series:
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

    if to_compute is None:
        to_compute = data.index 

    if containment == 'simplex':
        for time in tqdm(to_compute, disable=quiet):
            S_nj = 0
            
            point = data.loc[time, :]
            
            subseq = _helper._subsequences(list(data.drop(time, axis=0).index), d + 1)

            for seq in subseq:
                S_nj += _is_in_simplex(simplex_points=
                        np.array(data.loc[seq, :]), point=np.array(point))
                
            depths.append(S_nj / binom(n, d + 1))
    elif containment == 'l1':
        return _L1_depth(data=data, to_compute=to_compute)
    elif containment == 'mahalanobis':
        return _mahalanobis_depth(data=data, to_compute=to_compute)
    elif containment == 'oja':
        return _oja_depth(data=data, to_compute=to_compute)
    else: # Probably will be more in the future 
        raise ValueError(f'{containment} is not a valid containment measure. ')

    return pd.Series(index=to_compute, data=depths)

def _samplepointwisedepth(
    data: pd.DataFrame, 
    to_compute: pd.Index=None, 
    K=2, 
    containment='simplex', 
    quiet=True
) -> pd.Series:
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
        return _pointwisedepth(data=data, to_compute=to_compute, containment=containment)

    n, d = data.shape 
    depths = []

    if to_compute is None:
        to_compute = data.index 

    # K blocks of points (indices)
    ss = n // K 

    # Compute sample depth of each point, should be containment agnostic
    # Since the computation is being done in _pointwisedepth, which will call the appropriate depth measure
    for time in tqdm(to_compute, disable=quiet):
        cd = []
        for _ in tqdm(range(ss), disable=quiet):
            sdata = data.sample(n=ss, axis=0)
            
            # If our current datapoint isnt in the sampled data, just append it since we need to sample it 
            if not time in sdata.index:
                sdata = sdata.append(data.loc[time, :])
                
            cd.append(_pointwisedepth(data=sdata, to_compute=[time], containment=containment))
        depths.append(np.mean(cd))
    
    return pd.Series(index=to_compute, data=depths)

def _L1_depth(data: pd.DataFrame, to_compute: pd.Index=None):
    """
    Computes L1 data depth of the given points. 
    """
    n, d = data.shape
    depths = []
    idx = list(data.index)

    if to_compute is None:
        to_compute = idx
    
    for point in to_compute:
        sum_e = 0
        vec = data.loc[point, :]

        # Ugly code, but faster than 
        # Computing data.drop(point).index each time point
        cidx = idx[:]
        cidx.remove(point)

        for other in cidx:
            sum_e += (data.loc[other, :] - vec) / np.linalg.norm(vec - data.loc[other, :])
            
        depths.append(np.linalg.norm(sum_e) / n)
        
    return pd.Series(index=to_compute, data=1-np.array(depths))

def _mahalanobis_depth(data: pd.DataFrame, to_compute=None):
    """Mahalanobis depth for n points in R^n. Matrix must be square since we need to multiply the inv(covariance(data)).x for each point x."""
    n, p = data.shape

    # I think? We have to multiply and nxn matrix by our point in R^p
    if n != p:
        raise ValueError('Mahalanobis depth requires equal number of dimensions and datapoints.')
        
    mu = data.mean()
    inv_cov = np.linalg.inv(np.cov(data, rowvar=True))
    idx = data.index
    
    if to_compute is not None:
        idx = to_compute
        
    depths = []
    
    for point in idx:
        x = data.loc[point, :]
        S_x = np.dot(inv_cov, x)
        
        depths.append(np.dot((x-mu).T, S_x))
    return pd.Series(index=idx, data=depths)

def _oja_depth(data: pd.DataFrame, to_compute: list=None) -> pd.Series:
    """Oja depth for n multivariate samples in R^p. Data should be an n x p matrix."""
    n, d = data.shape
    idx = data.index
    depths = []
    
    if to_compute is not None:
        idx = to_compute
    
    try:
        vol_conv_p = ConvexHull(data).volume
    except:
        raise DepthDegeneracy('Too many collinear points to compute depth of convex hull spanned by data. Try another depth method or remove collinearities.')
        
    for point in idx:
        ci = list(idx)
        ci.remove(point)
        subseq = _helper._subsequences(ci, d)

        cd = 0
        for seq in subseq:
            try:
                hull = ConvexHull(data.loc[seq, :].append(data.loc[point, :]))
                cd += hull.volume
            except:
                raise DepthDegeneracy(f'Too many collinear points for data sampled at point {point}, will get a low depth score. Continuing...')

        depths.append(cd / vol_conv_p)

    return pd.Series(index=to_compute, data=depths)