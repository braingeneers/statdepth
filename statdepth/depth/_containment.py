from itertools import combinations
from typing import Callable, Union, List
from inspect import signature

import numpy as np
import scipy as sp 
import pandas as pd

from numba import jit
from scipy.special import comb, binom
from scipy.spatial import ConvexHull

def _is_valid_containment(containment: Callable[[pd.DataFrame, Union[pd.Series, pd.DataFrame], bool], float]) -> None: 
    '''Checks if the given function is a valid definition for containment. Used when user passes a custom containment function.
    
    Parameters:
    ----------

    containment: Function (Callable) to check validity of 

    Returns:
    ----------
    Boolean indicating the validity of the passed function
    '''

    # If we got here in _select_containment then the passed string is invalid
    if isinstance(containment, str):
        raise ValueError(f'containment argument \'{containment}\' is invalid. Use one of [\'r2\', \'r2_enum\', \'simplex \'] or a pass a custom containment function.')

    sig = signature(containment)
    params = sig.parameters

    if len(params) != 3:
        raise ValueError('Custom containment method has incorrect number of parameters. Expected 3, recieved {}'.format(len(params))) 

    return containment

def _r2_containment(data: pd.DataFrame, curve: pd.Series, relax: bool) -> float:
    '''Produces \lambda_r with the given input data, using the standard ordering on R as the definition for containment.
    Parameters:
    ----------

    data: list 
        DataFrame of real-valued functions that define our band in R^2 (columns are time intervals, rows are functions)
    curve: pd.Series
        Function to check containment on 
    relax: bool
        If False, we use the strict definition of containment. If True, we consider the proportion of time the curve is in the band

    Returns:
    ----------
    float: If relax=False, then 0 if the function is not contained in the curve, 1 if it is. If relax=True, then we consider the proportion of time the curve is in the band, so we will return a number between 0 and 1. 
    '''

    containment = 0
    
    y_range = []
    
    # Grab the mins/maxs across all rows (functions at each time index)
    mins = data.min(axis=1)
    maxs = data.max(axis=1)
    
    # Generate intervals in R
    intervals = [[i, j] for i, j in zip(mins, maxs)]
    
    # Check if each value in the curve is contained within the band 
    for index, val in enumerate(curve):
        if intervals[index][0] <= val <= intervals[index][1]:
            containment += 1
    
    # If relax=True, then we return the proportion of points in the band, else, Python integer division will round down to 0 unless all points are contained in the band (strict containment)
    return containment / len(curve) if relax else containment // len(curve)


def _r2_enum_containment(data: List[pd.DataFrame], curve: pd.DataFrame, relax: bool) -> float:
    '''Implements the r2_enum definition of containment, where we treat each component in the vector valued function as a real valued function, and calculate containment for each one. If all the components are contained in the curved defined by that componenent, then we say the function is contained.
    
    Parameters:
    ----------

    data: list
        Array of multivariate functions, where each item in array is an observation (columns are features, rows are time indices)
    curve: pd.DataFrame 
        Multivariate function to check containment of 
    relax: bool
        If False, we use the strict definition of containment. If True, we consider the proportion of time the curve is in the band

    Returns:
    ----------
    float: If relax=False, then 0 if the function is not contained in the curve, 1 if it is. If relax=True, then we consider the proportion of time the curve is in the band, so we will return a number between 0 and 1. 
    '''

    # TODO: This entire thing. I have no idea what im doing lol. All wrong so far

    depth = pd.DataFrame()

    # Choose any DataFrame since assumption is that all columns are the same -- dont need to check for this, user can deal with error.
    for col in data[0].columns:
        t = pd.DataFrame()
        for df in data:
            t[col] = data[col]
    

    return depth / len(data) if relax else depth // len(data)


def _simplex_containment(data: List[pd.DataFrame], curve: pd.DataFrame, relax: bool) -> float:
    '''Implements the simplex definition of containment for multivariate functions in R^n
        
    Parameters:
    ----------

    data: list
        Array of multivariate functions, where each item in array is an observation (columns are features, rows are time indices)
    curve: pd.DataFrame 
        Multivariate function to check containment of 
    relax: bool
        If False, we use the strict definition of containment. If True, we consider the proportion of time the curve is in the band

    Returns:
    ----------
    float: If relax=False, then 0 if the function is not contained in the curve, 1 if it is. If relax=True, then we consider the proportion of time the curve is in the band, so we will return a number between 0 and 1. 
    '''

    depth = 0

    for x in data[0].index:
        depth += _is_in_simplex(simplex_points=[curve.loc[x, :] for curve in data], point=curve.loc[x, :])

    return depth / len(data) if relax else depth // len(data)


def _is_in_simplex(simplex_points: pd.DataFrame, point: pd.Series) -> bool:
    '''Checks if the point is in the convex hull formed by simplex_points
    
    Parameters:
    ----------
    
    simplex_points: pd.DataFrame
        List of n-dimensional points (rows) to form the simplex. 
    point: pd.Series
        n-dimensional point to check containment of 

    Returns:
    ----------
    bool: True if point is contained in the simplex, false otherwise
    '''

    # Generate convex hull and grab its vertices. If this errors, the hull is degenerate and we consider the point not contained
    try:
        hull = ConvexHull(simplex_points, incremental=True)
    except:
        return False
    vertices = hull.vertices
    
    # Generate the convex hull with new point
    hull.add_points([point])
    
    # Check if they are the same
    # If they are, then the added point must be contained in the original hull
    # If there are a different number of vertices, clearly the hulls are different
    if len(vertices) != len(hull.vertices):
        return False  
    
    # Otherwise, make sure all the vertices are the same
    return all(vertices == hull.vertices)

def _select_containment(containment: Union[str, Callable]) -> Callable:
    '''Helper function to select definition of containment
    
    Parameters:
    ----------

    containment: Union[str, Callable]
        Containment string or function, used to select a built in containment method or handle a custom containment

    Returns:
    ----------
    Callable: Returns the containment function, or raises an error 

    '''
    
    # Select our containment definition if it is in our pre-defined list
    if containment == 'r2':
        return _r2_containment
    elif containment == 'r2_enum':
        return _r2_enum_containment
    elif containment == 'simplex':
        return _simplex_containment
    else:
        # Otherwise check validity of the containment Callable
        return _is_valid_containment(containment=containment)
