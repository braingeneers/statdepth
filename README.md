# statdepth: Depth Calculation Methods 
Read the docs at [https://statdepth.readthedocs.io/en/latest/](https://statdepth.readthedocs.io/en/latest/).

This package implements depth calculation and visualization methods for univariate time series data, multivariate time series data, and pointcloud data. This README will now mostly be development information. To see how to use the package, visit the documentation at the link above.

# Development

To install from `pip`, run
```
pip install statdepth
```

To install locally, run

```
pip install .
```

Or to install directly from this repo,
```
pip install git+https://github.com/braingeneers/functional_depth_methods
```

To set up the development environment as a Conda env, run
```
conda env create --file environment.yml
```

This code is written in Python, with most methods written in [Numpy](https://numpy.org/). It also uses [numba](https://numba.pydata.org/), a high performance Python compiler. Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN, so this should remove any speed issues Python has. 

Depending on how this ends up being used, [dask](https://dask.org/) may also be implemented for parallelization. 
