from itertools import combinations
from typing import Callable

import numpy as np
import scipy as sp 
import pandas as pd 
from numba import jit
from scipy.special import comb, binom

def is_valid_containment(containment) -> bool: 
    '''Checks if the given function is a valid definition for containment. Used when user passes a custom containment function
    
    Parameters:
    -------------
    containment: Function (Callable) to check validity of 


    Returns:
    Boolean indicating the validity of the passed function
    '''

    pass 


def _r2_containment(data: pd.DataFrame, curve: pd.Series) -> float:
    '''Produces \lambda_r with the given input data, using the standard ordering on R as the definition for containment.
    Parameters:
    -------------

    data: Array of real-valued functions that define our band in R^2
    curve: Function to check containment on 

    Returns:
    ----------
    0 if the function is not contained in the curve, 1 if it is
    '''

    # Generate discrete band 
    y_range = []
    for time_index in data.columns:
        y_range.append([data[time_index].min(), data[time_index].max()])

    # Check if each value in the curve is entirely contained within the band 
    for index, val in enumerate(curve):
        # If any value is not, then break out. This is strict containment!
        if not (y_range[index][0] <= val <= y_range[index][1]):
            return 0

    return 1


def _r2_enum_containment(data: pd.DataFrame, curve: pd.Series) -> float:
    '''Implements the r2_enum definition of containment, where we treat each component in the vector valued function as a real valued function, and calculate containment for each one. If all the components are contained in the curved defined by that componenent, then we say the function is contained'''

    pass


def _simplex_containment(data: pd.DataFrame, curve: pd.Series) -> float:
    '''Implements the simplex definition of containment for multivariate functions in R^n
        '''

    pass
