import pandas as pd 
import numpy as np
from typing import Union, List

def generate_noisy_univariate(data: Union[list, np.array], n: int=20, columns=None, index=None):
    """
    Generate n univariate functions that are equal to the given data plus some random pertubations. 
    Should be used for testing / understanding other methods in this library.

    Parameters:
    -----------
    data: list or np.array
        1d list of numbers to generate noisy data from. 
    n: (default=20)
        Number of noisy functions to generate.
    columns: (default=None)
        Names of columns. 
    index: (default=None)
        Index to use.

    Returns:    
    ---------
    pd.DataFrame: n x p DataFrame of p real valued functions observed at n discrete time points. (So each column is a function)

    """

    df = pd.DataFrame()

    # Generate perturbed
    for k in range(n):
        r = np.random.rand()
        df[k] = np.multiply(data, r)

    if index is not None:
        df.index = index 
    if columns is not None:
        df.columns = columns
    return df
    
def generate_noisy_pointcloud(n: int=50, d: int=2, columns=None, index=None) -> pd.DataFrame:
    """
    Generate n d-dimensional points from the normal distribution over [0,1]

    Parameters:
    -----------
    n: (default=20)
        Number of points to generate.
    d: (default=2)
        Dimension to draw points from.
    columns: (default=None)
        Names of columns. 
    index: (default=None)
        Index to use.

    """

    df = pd.DataFrame(np.random.rand(n, d))

    if columns is not None:
        df.columns = columns
    
    if index is not None:
        df.index = index
    
    return df