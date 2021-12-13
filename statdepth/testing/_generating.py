import pandas as pd 
import numpy as np
from typing import Union, List

def generate_noisy_univariate(
    data: Union[list, np.array]=None, 
    n: int=20, 
    columns=None, 
    index=None,
    seed=None,
) -> pd.DataFrame:
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

    np.random.seed(seed)

    df = pd.DataFrame()

    # If no starter function is provided, just generate some random one over [0,1]
    if data is None:
        data = np.random.rand(n)

    # Generate perturbed
    for k in range(n):
        r = np.random.rand()
        df[k] = np.multiply(data, r)

    if index is not None:
        df.index = index 
    if columns is not None:
        df.columns = columns

    return df

def generate_noisy_multivariate(
    data: pd.DataFrame=None, 
    num_curves: int=5, 
    n: int=10, 
    d: int=3, 
    columns=None, 
    index=None,
    seed=None,
) -> List[pd.DataFrame]:
    """
    Generate num_curves noisy multivariate functions with d features observed at n time points. 
    Should be used for testing / understanding other methods in this library.

    Parameters:
    -----------
    data: list or np.array
        1d list of numbers to generate noisy data from. 
    num_curves: (default=5)
        Number of multivariate functions to generate.
    n: (default=10)
        Number of timepoints.
    d: (default=3)
        Number of features (columns) our multivariate functions. This is the dimension of the image. 
    columns: (default=None)
        Names of columns. 
    index: (default=None)
        Index to use.

    Returns:    
    ---------
    List[pd.DataFrame]: A list of num_curves multivariate functions (DataFrames)

    """

    np.random.seed(seed)

    fs = []
    
    if data is None:
        data = np.random.rand(n, d)
        
    for _ in range(num_curves):
        r = np.random.rand()
        fs.append(pd.DataFrame(data) * r)

    if index is not None:
        for df in fs:
            df.index = index

    if columns is not None:
        for df in fs:
            df.columns = columns
    
    return fs

def generate_noisy_pointcloud(
    n: int=50, 
    d: int=2, 
    columns=None, 
    index=None,
    seed=None,
) -> pd.DataFrame:
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
    np.random.seed(seed)

    df = pd.DataFrame(np.random.normal(size=[n, d]))

    if columns is not None:
        df.columns = columns
    
    if index is not None:
        df.index = index
    
    return df