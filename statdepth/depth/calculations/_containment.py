from typing import Callable, Union, List
from inspect import signature
import warnings 

import numpy as np
import scipy as sp 
import pandas as pd

from scipy.special import comb, binom
from scipy.optimize import linprog

from ._helper import *
# This is not a solution I like, but I don't want to spam the user with
# warnings when simplex containment is used, because linprog() is very whiny
np.testing.suppress_warnings()

warnings.filterwarnings("ignore")

def _is_valid_containment(containment: Callable[[pd.DataFrame, Union[pd.Series, pd.DataFrame], bool], float]) -> None: 
    """
    Checks if the given function is a valid definition for containment. Used when user passes a custom containment function.
    
    Parameters:
    ----------

    containment: Function (Callable) to check validity of 

    Returns:
    ----------
    Boolean indicating the validity of the passed function
    """

    # If we got here in _select_containment then the passed string is invalid
    if isinstance(containment, str):
        raise ValueError(f'containment argument \'{containment}\' is invalid. Use one of [\'r2\', \'r2_enum\', \'simplex \'] or a pass a custom containment function.')

    sig = signature(containment)
    params = sig.parameters

    if len(params) != 3:
        raise ValueError('Custom containment method has incorrect number of parameters. Expected 3, recieved {}'.format(len(params))) 

    return containment

def _r2_containment(data: pd.DataFrame, curve: pd.Series, relax: bool) -> float:
    """
    Produces \lambda_r with the given input data, using the standard ordering on R as the definition for containment.
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
    """

    containment = 0
    
    y_range = []
    
    # Grab the mins/maxs across all rows (functions at each time index)
    mins = data.min(axis=1)
    maxs = data.max(axis=1)
    
    # Generate intervals in R over each time index
    intervals = [[i, j] for i, j in zip(mins, maxs)]
    
    # Check if each value in the curve is contained within the band 
    for index, val in enumerate(curve):
        if intervals[index][0] <= val <= intervals[index][1]:
            containment += 1
    
    # If relax=True, then we return the proportion of points in the band, else, Python integer division will round down to 0 unless all points are contained in the band (strict containment)
    return containment / len(curve) if relax else containment // len(curve)


def _r2_enum_containment(data: List[pd.DataFrame], curve: pd.DataFrame, relax: bool) -> float:
    """
    Implements the r2_enum definition of containment, where we treat each component in the vector valued function as a real valued function, 
    and calculate containment for each one. If all the components are contained in the curved defined by that componenent, then we say the function is contained.
    
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
    """

    raise NotImplementedError

def _simplex_containment(data: List[pd.DataFrame], curve: pd.DataFrame, relax: bool) -> float:
    """
    Implements the simplex definition of containment for multivariate functions in R^n
        
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
    """

    n = len(data)
    l, _ = data[0].shape
    
    containment = 0

    # For each time index, check containment 
    for idx in curve.index:
        containment += _is_in_simplex(simplex_points=np.array([df.loc[idx, :] for df in data]), 
                                point=np.array(curve.loc[idx, :]))
    
    # If relaxation, return proportion of containment, else do integer divion so that we 
    # only get 1 if all rows are contained
    return containment / l if relax else containment // l

def _is_in_simplex(simplex_points: pd.DataFrame, point: pd.Series) -> bool:
    """
    Checks if the d dimensional point is in the simplex formed by d + 1 simplex_points (geometric degeneracy allowed)
    
    Parameters:
    ----------
    
    simplex_points: pd.DataFrame
        List of n-dimensional points (rows) to form the simplex. 
    point: pd.Series
        n-dimensional point to check containment of 

    Returns:
    ----------
    bool: True if point is contained in the simplex, False otherwise
    """
    # Check if the vector x (point) can be written as a 
    # convex combination of x_1,...,x_n (simplex_points), i.e.
    # x = a_1x_1+...+a_nx_n such that a_i >=0, a_1+...+a_n = 1.

    # If this is possible, then the point is in the convex hull formed by those points.
    # In this case, the convex hull is formed with d+1 points so it is a simplex. 

    n_points = len(simplex_points)
    n_dim = len(point)
    
    c = np.zeros(n_points)
    A = np.r_[simplex_points.T, np.ones((1, n_points))]
    b = np.r_[point, np.ones(1)]
    
    # This barely ever errors, but when it does the problem is infeasible so let's assume
    # There is no containment
    try:
        lp = linprog(c, A_eq=A, b_eq=b)
    except:
        warnings.warn('Simplex computation failed. Assuming False')
        return False
    
    return lp.success

def _select_containment(containment: Union[str, Callable]) -> Callable:
    """
    Helper function to select definition of containment
    
    Parameters:
    ----------

    containment: Union[str, Callable]
        Containment string or function, used to select a built in containment method or handle a custom containment

    Returns:
    ----------
    Callable: Returns the containment function, or raises an error 

    """
    
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
