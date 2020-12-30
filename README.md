# statdepth: Functional Depth Methods for Multivariate Time Series Data 
This package implements depth calculation methods for multivariate time-series data.

## 1. What is functional depth?
Functional depth, specifically band depth, is a generalization of the median to a set of functions. Given a set of functions (or curves), we'd answer the question: in what sense is a curve the most central? 

This method implements the theory proposed in the paper *On the Concept of Depth for Functional Data* by authors López-Pintado and Juan Romo. 

## 2. Development

To set up the development environment, run  
```
conda env create --file environment.yml
```

This code is written in Python, with most methods written in [Numpy](https://numpy.org/). It also uses [numba](https://numba.pydata.org/), a high performance Python compiler. Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN, so this should remove any speed issues Python has. Depending on how this ends up being used, [dask](https://dask.org/) may also be implemented for parallelization. 

We make the following assumptions about your dataset:

Your data is indexed by time.  This is so our depth calculation can create a depth for each time increment, as well as across all time increments.  

Dataset example for a set of functions f_i: R --> R:
```Python
>>> df = pd.DataFrame({'x_0': [1, 2, 1, 1, 2], 
...                    'x_1': [2, 3, 4, 2, 2], 
...                    'x_2': [1, 3, 1, 3, 2],
...                    'x_3': [1, 2, 1, 1, 2],
...                    'x_4': [2, 3, 4, 2, 1]}, 
...                   index=['f_1', 'f_2', 'f_3', 'f_4', 'f_5'])

>>> df
     x_0  x_1  x_2  x_3  x_4
f_1    1    2    1    1    2
f_2    2    3    3    2    3
f_3    1    4    1    1    4
f_4    1    2    3    1    2
f_5    2    2    2    2    1

```

Each x_i is a timepoint, each row is a function f_i. 

In the case of multivariate functions, each DataFrame in the list should 

## 3. Methods
The methodology implemented here requires a notion of "containment" of a function f within the band defined by other functions. In R^2, we can check this pointwise. In higher dimensional space, there are more options. 

We have the following notions of containment:  
- `'r2'`: Standard pointwise interval containment  
- `'r2_enum'`: Enumerates over samples of our vector valued functions, treating each as a scalar in R, and then uses the standard pointwise interval containment definition.
- `'simplex'`: TODO  


`banddepth(data: list, J=2, containment='r2', method='MBD')`:  

    Calculate the band depth for a set of functional curves.

    Band depth is an order statistic for functional data (see `fboxplot`), with
    a higher band depth indicating larger "centrality".  In analog to scalar
    data, the functional curve with highest band depth is called the median
    curve, and the band made up from the first N/2 of N curves is the 50%
    central region.

    Parameters
    ----------
    data : ndarray
        Functions to calculate band depth from. Each DataFrame should be an n x p matrix which is a function evaluated at all timepoints. 
    J: int (default=2)
        J parameter in the band depth calculation. J=3 can be computationally expensive for large datasets, and also does not have a closed form solution. 
    Containment: Callable or string
        Defines what containment means for the dataset. For functions from R-->R, we use the standard ordering on R. For higher dimensional spaces, we implement a simplex method. 
        A full list can be found in the README, as well as instructions on passing a custom definition for containment.  
    Returns
    -------
    ndarray
        Depth values for functional curves.
