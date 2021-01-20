# statdepth: Depth Calculation Methods 
This package implements depth calculation and visualization methods for univariate time series data, multivariate time series data, and pointcloud data.

This README will now mostly be development information. For information about how to use the package, check out the docs at [https://statdepth.readthedocs.io/en/latest/](https://statdepth.readthedocs.io/en/latest/).

# Development

To set up the development environment as a Conda env, run
```
conda env create --file environment.yml
```

To install locally, run

```
pip install .
```

Or to install directly from this repo,
```
pip install git+https://github.com/braingeneers/functional_depth_methods
```

This code is written in Python, with most methods written in [Numpy](https://numpy.org/). It also uses [numba](https://numba.pydata.org/), a high performance Python compiler. Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN, so this should remove any speed issues Python has. 

Depending on how this ends up being used, [dask](https://dask.org/) may also be implemented for parallelization. 
