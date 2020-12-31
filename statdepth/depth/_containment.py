from itertools import combinations
from typing import Callable

import numpy as np
import scipy as sp 
import pandas as pd 
from numba import jit
from scipy.special import comb, binom

def _is_valid_containment(containment) -> bool: 
    '''Checks if the given function is a valid definition for containment. Used when user passes a custom containment function
    
    Parameters:
    ----------
    containment: Function (Callable) to check validity of 


    Returns:
    ----------
    Boolean indicating the validity of the passed function
    '''

    pass 


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
    
    mins = data.min(axis=1)
    maxs = data.max(axis=1)
    
    intervals = [[i, j] for i, j in zip(mins, maxs)]
    
    # Check if each value in the curve is entirely contained within the band 
    for index, val in enumerate(curve):
        # If any value is not, then break out. This is strict containment!
        if intervals[index][0] <= val <= intervals[index][1]:
            containment += 1
        
    return containment / len(curve) if relax else containment // len(curve)


def _r2_enum_containment(data: list, curve: pd.DataFrame, relax: bool) -> float:
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

    pass


def _simplex_containment(data: list, curve: pd.DataFrame, relax: bool) -> float:
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

    pass


def _select_containment(containment: str) -> Callable:
    '''Helper function to select definition of containment'''
    
    # Select our containment definition if it is in our pre-defined list
    if containment == 'r2':
        return _r2_containment
    elif containment == 'r2_enum':
        return _r2_enum_containment
    elif containment == 'simplex':
        return _simplex_containment
    else:
        # TODO: Allow user to pass in custom definition of containment
        raise ValueError('Error: Unknown or unspecified definition of containment')
