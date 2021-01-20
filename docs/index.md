# statdepth: Depth Calculation Methods 
This package implements depth calculation methods for univariate time series data, multivariate time series data, and pointcloud data.

# 1. What is statistical depth?

Statistical depth allows us to assign a non-negative and bounded real number to each observation in some dataset such that the largest depth value indicates the sample most representitive of the generating distrubution, and the lowest depth value indicates the most outlying sample. 

This allows us to order our observations by their "centrality" and "outylingness." This not only allows us to determine the "best" sample, but is also a non-parametric way to remove outliers. 

# 2. What is functional depth?
Functional depth is the generalization of statistical depth to functions (called curves). The larger the depth, the "deeper" the curve, and the curve with the highest band depth is the "most central", and is typically called the median. In this sense, we assume each function is a single observation.

# 3. General Usage and Examples
For the functional cases, your data is indexed by time. Either your data is a set of real valued functions, or your data is a set of multivariate functions, with data collected at a discrete number of time points. We consider the two cases:

#### i. Real-valued functions

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

#### ii. Multivariate functions

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

#### iii. Pointcloud data
Given a n x p DataFrame for pointcloud data in R^p, each column is a dimension and each row is a point in R^p. In the example below, we have 5 points sampled from a distribution in R^2. 

In this case, to calculate depth of each point with respect to the others, we use 

```Python
>>> from statdepth import PointcloudDepth

>>> df
          0         1
0  0.873179  0.828111
1  0.368512  0.024619
2  0.927522  0.348593
3  0.481917  0.748796
4  0.980515  0.954392

>>> PointwiseDepth(df, containment='l1')
0    0.703605
1    0.239076
2    0.458779
3    0.456768
4    0.258959
dtype: float64
```

# 4. Containment

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

# 5. Classes and Methods
All depth classes implement a `K` parameter, which computes sampled depth instead of exact depth. This should reduce computational costs for large datasets. 

For a function f, this is done by
1. Splitting the data of n functions into K blocks of size ~n/K
2. Computing band depth of f with respect to each block
3. Returning the average of these

For K << n, this should approximate the band depth well. This will have a higher space complexity because the data needs to be copied K times. 

## 5.1: statdepth

#### 5.1.1: FunctionalDepth
Calculate depth of univariate or multivariate functional data. See Examples.

```Python
FunctionalDepth(data: List[pd.DataFrame], K=None, J=2, 
containment='r2', relax=False, deep_check=False)
```

Parameters:  
* data : list of DataFrames 
   * Functions to calculate band depth from  

* J: int (default=2) 
   * J parameter in the band depth calculation. J=3 can be   computationally expensive for large datasets, and also does not have a closed form solution.   
* containment: Callable or string (default='r2')  
   * Defines what containment means for the dataset. For functions from R-->R, we use the standard ordering on R. For higher dimensional spaces, we implement a simplex method.   
   A full list can be found in the README, as well as instructions on passing a custom definition for containment.     
* relax: bool  
   * If True, use a strict definition of containment, else use containment defined by the proportion of time the curve is in the band. 
* deep_check: bool (default=False)
   * If True, perform a more extensive error checking routine. Optional because it can be computationally expensive for large datasets. 

Returns:
   * pd.Series, pd.DataFrame: 
      * Depth values for each function.
      
Analytic methods:
<font size="4">
- `ordered(ascending=False)`: Sort the curves by their band depth 
- `deepest(n=1)`: Return the `n` deepest curves
- `outlying(n=1)`: Return the `n` most outlying curves
- `drop_outlying_data(n=1)`: Return the original data with the `n` most outlying curves dropped
- `get_deep_data(n=1)`: Return the `n` deepest curves from the original data
- `get_depths()`: Return the depths in the order of the original columns
- `get_data()`: Return the original data passed 
- `sorted(ascending=False)`: Alias for ordered()
- `median()`: Alias for `deepest(n=1)`
</font>

Visualizations:
<font size="4">
- `plot_deepest(n=1)`: Plot all the curves with the `n` deepest marked in red
- `plot_outlying(n=1)`: Plot all curves with the `n` outyling marked in red
</font>

#### 5.1.2: PointcloudDepth

Compute pointwise depth for n points in R^p, where data is an nxp matrix of points. If points is not None,
only compute depth for the given points (should be a subset of data.index)

```Python
PointwiseDepth(data: pd.DataFrame, points: pd.Index=None, K=None, containment='simplex')
```

Parameters:
* data: pd.DataFrame
   * n x d DataFrame, where we have n points in d dimensional space.
* points: list, pd.Index
   * The particular points (indices) we would like to calculate band curve for. If None, we calculate depth for all points.
* K=2:
   * Number of blocks to compute sample depth with. 
* containment: str
   * Definition of containment. Should be one of 'l1', 'simplex'

Returns:

* pd.Series: 
   * Depth values for the given points with respect to the data. Index of Series are indices of points in the original data, and the values are the depths

Analytic methods:
<font size="4">
- `ordered(ascending=False)`: Sort the curves by their band depth 
- `deepest(n=1)`: Return the `n` deepest curves
- `outlying(n=1)`: Return the `n` most outlying curves
- `drop_outlying_data(n=1)`: Return the original data with the `n` most outlying curves dropped
- `get_deep_data(n=1)`: Return the `n` deepest curves from the original data
- `get_depths()`: Return the depths in the order of the original columns
- `get_data()`: Return the original data passed 
- `sorted(ascending=False)`: Alias for ordered()
- `median()`: Alias for `deepest(n=1)`
</font>

Visualizations:
<font size="4">
- `plot_depths(invert_colors=False)`: Plot all datapoints and color them by their depth. If dimension of data is greater than 3, plot parallel axis intead.  
- `plot_deepest(n=1)`: Plot all the curves with the `n` deepest marked in red  
- `plot_outlying(n=1)`: Plot all curves with the `n` outyling marked in red  
- `plot_distribution(invert_colors=False)`: Alias for `plot_depths()`  
</font>

### 5.1.3: ProbabilisticDepth

Calculate depth of univariate functional data with respect to variances for each observed datapoint. See Examples.

```Python
ProbabilisticDepth(data: pd.DataFrame, sigma2: pd.DataFrame, K=None, J=2, relax=False)
```

Parameters:  
* data : pd.DataFrame 
   * Functions to calculate band depth from 
* sigma2: pd.DataFrame
   * DataFrame of variances, should be the same size as data
* K=2:
   * Number of blocks to compute sample depth with. 
* J: int (default=2) 
   * J parameter in the band depth calculation. J=3 can be   computationally expensive for large datasets, and also does not have a closed form solution.   
* relax: bool  
   * If True, use a strict definition of containment, else use containment defined by the proportion of time the curve is in the band. 

Returns:
   * pd.Series
      * Depth values for each univariate function.
      
Analytic methods:
<font size="4">
- `ordered(ascending=False)`: Sort the curves by their band depth 
- `deepest(n=1)`: Return the `n` deepest curves
- `outlying(n=1)`: Return the `n` most outlying curves
- `drop_outlying_data(n=1)`: Return the original data with the `n` most outlying curves dropped
- `get_deep_data(n=1)`: Return the `n` deepest curves from the original data
- `get_depths()`: Return the depths in the order of the original columns
- `get_data()`: Return the original data passed 
- `sorted(ascending=False)`: Alias for ordered()
- `median()`: Alias for `deepest(n=1)`
</font>

Visualizations:
<font size="4">
- `plot_deepest(n=1)`: Plot all the curves with the `n` deepest marked in red
- `plot_outlying(n=1)`: Plot all curves with the `n` outyling marked in red
</font>

## 5.2: statdepth.testing 

#### 5.2.1: generate_noisy_univariate
Generate n univariate functions that are equal to the given data plus some random pertubations. 
Should be used for testing / understanding other methods in this library.

```Python
generate_noisy_univariate(data: Union[list, np.array]=None, n: int=20, columns=None, index=None)
```

Parameters:
* data: list or np.array
   * 1d list of numbers to generate noisy data from. If None, sample from normal distribution over [0,1].
* n: (default=20)
   * Number of noisy functions to generate.
* columns: (default=None)
   * Names of columns. 
* index: (default=None)
   * Index to use.

Returns:    
* pd.DataFrame: 
   * n x p DataFrame of p real valued functions observed at n discrete time * points. (So each column is a function)

#### 5.2.2: generate_noisy_multivariate
Generate num_curves noisy multivariate functions with d features observed at n time points. 
Should be used for testing / understanding other methods in this library.

```Python
generate_noisy_multivariate(data: pd.DataFrame=None, num_curves: int=5, n: int=10, d: int=3, columns=None, index=None)
```

Parameters:
* data: list or np.array
   * 1d list of numbers to generate noisy data from. 
* num_curves: (default=5)
   * Number of multivariate functions to generate.
* n: (default=10)
   * Number of timepoints.
* d: (default=3)
   * Number of features (columns) our multivariate functions. This is the dimension of the image. 
* columns: (default=None)
   * Names of columns. 
* index: (default=None)
   * Index to use.

Returns:    
* List[pd.DataFrame]: 
   * A list of num_curves multivariate functions (DataFrames)

#### 5.2.3: generate_noisy_pointcloud

Generate n d-dimensional points from the normal distribution over [0,1]

```Python
generate_noisy_pointcloud(n: int=50, d: int=2, columns=None, index=None)
```

Parameters:
* n: (default=20)
   * Number of points to generate.
* d: (default=2)
   * Dimension to draw points from.
* columns: (default=None)
   * Names of columns. 
* index: (default=None)
   * Index to use.

Returns:
* pd.DataFrame: 
   * Generated noisy data
