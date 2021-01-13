import pandas as pd 
from typing import Callable, List, Union, Dict
import plotly.graph_objects as go

from ._depthcalculations import _banddepth, _samplebanddepth, _pointwisedepth, _samplepointwisedepth
from .abstract import AbstractDepth

# Custom error class for anytime there is going to be some degeneracy with depth calculation (i.e. k degenerate simplices)
class DepthDegeneracy(Exception):
    pass

# Private class that wraps the band depth calculation methods with some extra attributes as well
class _FunctionalDepthSeries(AbstractDepth, pd.Series):

    def __init__(self, df: pd.DataFrame, depths: pd.Series):
        super().__init__(data=depths)

        self._orig_data = df
        self._depths = depths
        self._ordered_depths = None

    def ordered(self, ascending=False) -> pd.Series:
        '''Sort the curves by their band depth, from deepest to most outlying'''
        if self._ordered_depths is None:
            self._ordered_depths =  self._depths.sort_values(ascending=ascending)
        return self._ordered_depths

    def deepest(self, n=1) -> pd.Series:
        '''Return the n deepest curves. Equivalently, return the n largest items in the depths Series'''
        if self._ordered_depths is None:
            self._ordered_depths = self._depths.sort_values(ascending=False)
        
        if n == 1:
            return pd.Series(index=[list(self._ordered_depths.index)[0]], data=[self._ordered_depths.values[0]])
        else:
            return pd.Series(index=self._ordered_depths.index[0: n], data=self._ordered_depths.values[0: n])
    
    def outlying(self, n=1) -> pd.Series:
        if self._ordered_depths is None:
            self._ordered_depths = self._depths.sort_values(ascending=False)

        if n == 1:
            return pd.Series(index=[list(self._ordered_depths.index)[-1]], data=[self._ordered_depths.values[-1]])
        else:
            return pd.Series(index=self._ordered_depths.index[-n: ], data=self._ordered_depths.values[-n: ])
    

class _FunctionalDepthMultivariateDataFrame(AbstractDepth, pd.DataFrame):
    # Don't copy over all the functions. Numpy arrays might be large
    # and this will add extra space complexity
    # Also, visualization for multivariate functions isn't extensive enough to justify 
    # storing all of the data
    def __init__(self, names: List[str], depths: pd.DataFrame):
        super().__init__(depths)
        self._names = names
        self._depths = depths

    # Not inheritable from _FunctionalDepthSeries because depths is a DataFrame
    def ordered(self, ascending=False):
        pass

    def deepest(self, n=1):
        pass

    def outlying(self, n=1):
        pass

class _FunctionalDepthUnivariate(_FunctionalDepthSeries):
    def __init__(self, df: pd.DataFrame, depths: pd.Series):
        super().__init__(df=df, depths=depths)

        self._orig_data = df
        self._depths = depths
        self._ordered_depths = None

    def _plot(self, deep_or_outlying: pd.Series) -> None:
        cols = self._orig_data.columns
        x = self._orig_data.index

        # We use deep_or_outlying.index to get the columns because 
        # deep_or_outlying is a Series indexed by the original columns
        data=[go.Scatter(x=x, y=self._orig_data.loc[:, y], mode='lines+markers', marker_color='Blue') for y in cols]
        data.extend([go.Scatter(x=x, y=self._orig_data.loc[:, y], mode='lines+markers', marker_color='Red') for y in deep_or_outlying.index])

        fig = go.Figure(data=data)
        fig.update_layout(showlegend=False)

        fig.show()

    def plot_deepest(self, n=1) -> None:
        '''Plots all the data in blue and marks the n deepest in red'''
        self._plot(deep_or_outlying=self.deepest(n=n))

    def plot_outlying(self, n=1) -> None:
        '''Plots all the data in blue and marks the n most outlying curves in red'''
        self._plot(deep_or_outlying=self.outlying(n=n))

class _PointwiseDepth(_FunctionalDepthSeries):
    '''Pointwise depth calculation for Multivariate data. Calculates depth of each point with respect to the sample in R^n.'''

    def __init__(self, df: pd.DataFrame, depths: pd.Series):
        super().__init__(df=df, depths=depths)

        self._orig_data = df
        self._depths = depths
        self._ordered_depths = None

    def _plot_parallel_axis(self, df: pd.DataFrame) -> None:
        pass 

    def _plot(self, deep_or_outlying: pd.Series) -> None:
        n = self._orig_data.columns
        if n > 3:
            self._plot_parallel_axis(df=self._orig_data)
        elif n == 3:
            pass
        elif n == 2: 
            pass
        else: # n = 1
            pass

    def plot_deepest(self, n=1) -> None:
        pass

    def plot_outlying(self, n=1) -> None:
        pass

def PointwiseDepth(data: pd.DataFrame, K=None, J=2, containment='simplex', relax=False, deep_check=False) -> _PointwiseDepth:
    if K is not None:
        depth = _samplepointwisedepth(data=data, K=K, J=J, containment=containment, relax=relax, deep_check=False)
    else:
        depth = _pointwisedepth(data=data, J=J, containment=containment)
    
    return _PointwiseDepth(data=data, depths=depth)

def FunctionalDepth(data: List[pd.DataFrame], K=None, J=2, 
containment='r2', relax=False, deep_check=False) -> Union[_FunctionalDepthSeries, _FunctionalDepthMultivariateDataFrame]:   

    # If there is not at least d + 2 functions for our d dimensional data, then for each function
    # We won't have d + 1 vertices to construct a simplex, which means every simplex will be at least one dimensional degenerate
    # Therefore we say depth is not well defined and error
    if isinstance(data, list) and len(data) < data[0].shape[1] + 2:
        raise DepthDegeneracy(f'Error: Need at least {len(data)} functions to form non-degenerate simplices in {data[0].shape + 2} dimensional space. Only have {len(data)}')

    if K is not None:
        depth = _samplebanddepth(data=data, K=K, J=J, containment=containment, relax=relax, deep_check=deep_check)
    else:
        depth = _banddepth(data=data, J=J, containment=containment, relax=relax, deep_check=deep_check)

    if isinstance(depth, pd.DataFrame):
        return _FunctionalDepthMultivariateDataFrame(names=keys, depths=depth)
    else:
        return _FunctionalDepthUnivariate(df=data[0], depths=depth)
