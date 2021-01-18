# statdepth: Functional Depth Methods for Multivariate Time Series Data 
This package implements depth calculation methods for multivariate time-series data.

## 1. What is functional depth?
Functional depth, specifically band depth, is a generalization of the median to a set of functions. Given a set of functions (or curves), band depths allows us to order our curves with respect to their "centrality." The larger the depth, the "deeper" the curve, and the curve with the highest band depth is the "most central" (in a not-so-formal sense 😀).

This method implements the theory proposed in the paper [*On the Concept of Depth for Functional Data*](https://www.researchgate.net/publication/33397608_On_the_Concept_of_Depth_for_Functional_Data) by López-Pintado and Juan Romo. 

## 2. Development

To set up the development environment, run
```
conda env create --file environment.yml
```

To install locally, run

```
pip install .
```

This code is written in Python, with most methods written in [Numpy](https://numpy.org/). It also uses [numba](https://numba.pydata.org/), a high performance Python compiler. Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN, so this should remove any speed issues Python has. 

Depending on how this ends up being used, [dask](https://dask.org/) may also be implemented for parallelization. 

## 3. Usage

### i. Data Structure Assumptions
Your data is indexed by time. Either your data is a set of real valued functions, or your data is a set of multivariate functions, with data collected at a discrete number of time points. We consider the two cases:

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
>>> from statdepth import FunctionalDepth

>>> FunctionalDepth([df], J=2, relax=False).ordered()
f_3    0.400000
f_5    0.266667
f_2    0.200000
f_1    0.200000
f_4    0.000000
f_0    0.000000
dtype: float64
```

If a single item is passed in the list, *it is assumed we are in the univariate case*. This is because there is no way to detect internally where to "split" the DataFrame to isolate each function in the multivariate case. 

#### iii. Multivariate functions

In the case of **multivariate functions**, each DataFrame in a list should be a function where the *columns* are the *features* and the rows are the *time indices*. For example, if we had multivariate observations given by
```Python
>>> df1
       size  co_amount  weight
00:00     0          2       2
00:30     1          0       3
01:00     1          3       2
01:30     3          0       3
02:00     3          3       1
02:30     1          1       0

>>> df2
       size  co_amount  weight
00:00     3          2       3
00:30     0          3       2
01:00     1          3       2
01:30     2          2       0
02:00     0          1       2
02:30     1          3       3

>>> df3
...
```

Then to compute band depth, we would use 

```Python
>>> from statdepth import FunctionalDepth

>>> FunctionalDepth([df1, df2, df3, ... , df6], containment='simplex', J=2, relax=True)
2    0.333333
1    0.333333
5    0.166667
0    0.166667
4    0.000000
3    0.000000
dtype: float64

```

where the index is the index of the DataFrame in the list passed. So in this case `df3` is the deepest multivariate curve. 
## 4. Containment

The methodology implemented here requires a notion of "containment" of a function within a band. In R^2, we can check this pointwise. In higher dimensional space, there are more options. 

Currently, we have the following notions of containment:  
- `'r2'`: Standard pointwise interval containment. 
<!-- - `'r2_enum'`: Treats each component in our vector-valued function as a real-valued function, and then uses the standard pointwise interval definition. (TODO) -->
- `'simplex'`: A d dimensional function is contained by d+1 other functions if each discrete point is contained by the simplex formed by d + 1 other functions at each time index. 

#### i. Using an alternative definition of containment

To use your own definition of containment, simply pass a containment function into the `containment` parameter of `banddepth`. A containment function should be structured similarly to the following (arguments are enforced):

```Python
def containment(
    data: Union[List[pd.DataFrame], pd.DataFrame], 
    curve: Union[pd.DataFrame, pd.Series], 
    relax=False
) -> float:

    contained = 0
    if len(data) == 1:
        # Handle univariate case
    else:
        # Handle multivariate case
    
    return contained
```

`data` should either take in a list of DataFrames (and a DataFrame for `curve`), or a single DataFrame (and a Series for `curve`) in the univariate and multivariate case, respectively. 

The returned float should be a value between 0 and 1 (this is not enforced). 

The relaxation parameter is optional, and is used to relax the strict definition of containment into a definition that considers the proportion of "time" a function is contained.

# 5. Methods
All depth classes implement a `K` paremeter, which computes sampled depth instead of exact depth. This should reduce computational costs for large datasets. 

For a function f, this is done by
1. Splitting the data of n functions into K blocks of size ~n/K
2. Computing band depth of f with respect to each block
3. Returning the average of these

For K << n, this should approximate the band depth well. 


## 5.1: Top level classes
All Depth-y classes subclass `pd.Series`.

```Python
FunctionalDepth(data: List[pd.DataFrame], K=None, J=2, 
    containment='r2', relax=False, deep_check=False)
```

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

    Methods:
    ----------
    ordered(): Return the ordered depths
    deepest(n=1): Return the n deepest curves
    outlying(n=1): Return the n outlying curves

    plot_depths(invert_colors=False, marker=None): Plot the depths, where the colorscale is the depths
    plot_outlying(n=1): Plot all the functions and the n most outlying in a distinct color
    plot_deepest(n=1): Plot all the functions and the n deepest in a distinct color

    sorted(): Alias for ordered()
    get_depths(): Return the depths
    get_data(): Return the original DataFrame 

```Python
def PointwiseDepth(data: pd.DataFrame, points: pd.Index=None, K=None, containment='simplex')
```

    Compute statistical depth for up to n points from a distribution in R^p. If points is not None, only compute depth for the given points (must be a subset of data.index)

    Parameters:
    ----------
    data: pd.DataFrame
        The input data. An n x p matrix where each row is a point in R^p.
    points: pd.Index=None
        Points to compute depth for. If None, compute depth for all points in the DataFrame.
    K: int=None
        Number of blocks to use if using sample band depth
    containment: str='simplex'
        Definition of containment (depth) to use
    
    Methods:
    ----------
    ordered(): Return the ordered depths
    deepest(n=1): Return the n deepest curves
    outlying(n=1): Return the n outlying curves

    plot_depths(invert_colors=False, marker=None): Plot the depths, where the colorscale is the depths
    plot_outlying(n=1): Plot all the functions and the n most outlying in a distinct color
    plot_deepest(n=1): Plot all the functions and the n deepest in a distinct color

    sorted(): Alias for ordered()
    get_depths(): Return the depths
    get_data(): Return the original DataFrame 
