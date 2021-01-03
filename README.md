# statdepth: Functional Depth Methods for Multivariate Time Series Data 
This package implements depth calculation methods for multivariate time-series data.

## 1. What is functional depth?
Functional depth, specifically band depth, is a generalization of the median to a set of functions. Given a set of functions (or curves), functional depth answers the question: how can we order our curves with respect to centrality/outlyingness? 

This method implements the theory proposed in the paper *On the Concept of Depth for Functional Data* by authors López-Pintado and Juan Romo. 

## 2. Development

To set up the development environment, run  
```
conda env create --file environment.yml
```

To install locally, run

```
pip install .
```

This code is written in Python, with most methods written in [Numpy](https://numpy.org/). It also uses [numba](https://numba.pydata.org/), a high performance Python compiler. Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN, so this should remove any speed issues Python has. Depending on how this ends up being used, [dask](https://dask.org/) may also be implemented for parallelization. 

## 3. Usage

### i. Data Structure
Your data is indexed by time. Either your data is a set of real valued functions, or your data is a set of multivariate functions. We consider the two cases:

#### ii. Real-valued functions

Dataset example for a set of functions f_i: R --> R:
```Python
>>> df
     f_0  f_1  f_2  f_3  f_4  f_5
x_0    1    2    3  6.0    9    8
x_1    2    4    4  7.0    9    8
x_2    3    5    4  6.5   12   10
x_3    2    6    2  6.0   11   10
x_4    1    2    1  7.0   11    9

```

Each x_i is a timepoint, each column is a function f_i. In this case, we compute band depth using 

```Python 
>>> from statdepth.depth import banddepth

>>> banddepth([df], J=2).sort_values(ascending=False)
f_3    0.400000
f_5    0.266667
f_2    0.200000
f_1    0.200000
f_4    0.000000
f_0    0.000000
dtype: float64
```

(Note: the `sort_values()` is not implemented, but rather `banddepth` returns a `pd.Series` object). If a single item is passed in the list, *it is assumed we are in the univariate case*. This is because there is no way to detect internally where to "split" the DataFrame to isolate each function in the multivariate case. 

#### iii. Multivariate functions

In the case of **multivariate functions**, each DataFrame in a list should be a function where the *columns* are the *features* and the rows are the *time indices*. For example, if we had the three multivariate observations given by
```Python
>>> df1
       size  weight  co_amount
00:00   1.0       2          3
00:50   4.0       5          6
01:25   0.5       1          2
>>> df2
       size  weight  co_amount
00:00     3       2          1
00:50     5       4          3
01:25     1       1          0
>>> df3
       size  weight  co_amount
00:00   1.0       2          3
00:50   9.0      10          1
01:25   0.5       1          2

```

Then to compute band depth, we would use 

```Python
from statdepth.depth import banddepth

banddepth([df1, df2, df3], containment='r2_enum', J=2)
```

## 4. Containment

The methodology implemented here requires a notion of "containment" of a function within a band. In R^2, we can check this pointwise. In higher dimensional space, there are more options. 

Currently, we have the following notions of containment:  
- `'r2'`: Standard pointwise interval containment
- `'r2_enum'`: Treats each component in our vector-valued function as a real-valued function, and then uses the standard pointwise interval definition. (TODO)
- `'simplex'`: A point is contained if it is contained in all simplices defined by the other curves

#### i. Using an alternative definition of containment

To use your own definition of containment, simply pass a containment function into the `containment` parameter of `banddepth`. A containment function must be structured in the following way:

```Python
def containment(
    data: pd.DataFrame, 
    curve: Union[str, int], 
    relax=False
) -> float
```

Where the returned float is a value between 0 and 1. The relaxation parameter is optional, and is used to relax the strict definition of containment into a definition that considers the proportion of time a function is contained in the band (or simplex). 

# 5. Methods

`banddepth(data: List[pd.DataFrame], J=2, containment='r2', relax=False, deep_check=False) -> Union[pd.DataFrame, pd.Series]`:  

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
    pd.Series: Depth values for each function.

`samplebanddepth(data: List[pd.DataFrame], K: int, J=2, containment='r2', relax=False, deep_check=False) -> Union[pd.Series, pd.DataFrame]`

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
