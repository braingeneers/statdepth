import pandas as pd 
from typing import Callable, List, Union, Dict
import plotly.graph_objects as go

from ._depthcalculations import _banddepth, _samplebanddepth, _pointwisedepth, _samplepointwisedepth
from .abstract import AbstractDepth

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
    
    def plot_deepest(self, n=1) -> None:
        '''Plots all the data in blue and marks the n deepest in red'''
        s = self.deepest(n=n)
        cols = self._orig_data.columns
        x= self._orig_data.index

        data=[go.Scatter(x=x, y=self._orig_data[y], mode='lines+markers', marker_color='Blue') for y in cols]
        data.extend([go.Scatter(x=x, y=self._orig_data[y], mode='lines+markers', marker_color='Red') for y in s.index])

        fig = go.Figure(data=data)
        fig.update_layout(showlegend=False)

        fig.show()

class _FunctionalDepthDataFrame(AbstractDepth, pd.DataFrame):
    def __init__(self, names: List[str], depths: pd.DataFrame):
        super().__init__(depths)
        self._names = names
        self._depths = depths

    def ordered(self, ascending=False):
        pass

    def deepest(self, n=1):
        pass

    def outlying(self, n=1):
        pass

class _PointwiseDepth(AbstractDepth, pd.Series):
    '''Pointwise depth calculation for Multivariate data. Calculates depth of each point with respect to the sample in R^n.'''

    def __init__(self, data: pd.DataFrame, depths: pd.Series):

        self._orig_data = data
        self._depths = depths
        self._ordered_depths = None

    def ordered(self, ascending=False):
        pass

    def deepest(self, n=1):
        pass

    def outlying(self, n=1):
        pass

def PointwiseDepth(data: pd.DataFrame, K=None, J=2, containment='simplex', relax=False, deep_check=False) -> _PointwiseDepth:
    if K is not None:
        depth = _samplepointwisedepth(data=data, K=K, J=J, containment=containment, relax=relax, deep_check=False)
    else:
        depth = _pointwisedepth(data=data, J=J, containment=containment)
    
    return _PointwiseDepth(data=data, depths=depth)

def FunctionalDepth(data: List[pd.DataFrame], K=None, J=2, 
containment='r2', relax=False, deep_check=False) -> Union[_FunctionalDepthSeries, _FunctionalDepthDataFrame]:    
    keys = []
    
    if isinstance(data, dict):
        keys.extend(data.keys())
        data = data.values()

    if K is not None:
        depth = _samplebanddepth(data=data, K=K, J=J, containment=containment, relax=relax, deep_check=deep_check)
    else:
        depth = _banddepth(data=data, J=J, containment=containment, relax=relax, deep_check=deep_check)

    if isinstance(depth, pd.DataFrame):
        return _FunctionalDepthDataFrame(names=keys, depths=depth)
    else:
        return _FunctionalDepthSeries(df=data[0], depths=depth)

# def PointWiseDepth(data: pd.DataFrame, K=None, J=2, containment='r2', relax=False, deep_check=False) -> _PointwiseDepth:
#     return _PointwiseDepth(None, None)