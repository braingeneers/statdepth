import pandas as pd 
from typing import Callable, List, Union, Dict
import plotly.graph_objects as go

from ._depthcalculations import _banddepth, _samplebanddepth

class _BandDepthSeries(pd.Series):
    def __init__(self, df, depths):
        super().__init__(data=depths)

        self._orig_data = df
        self._depths = depths
        self._ordered_depths = None

    def ordered(self, ascending=False):
        if self._ordered_depths is None:
            self._ordered_depths =  self._depths.sort_values(ascending=ascending)
        return self._ordered_depths

    def sorted(self, ascending=False):
        return self.ordered(ascending=ascending)

    def deepest(self, n=1):
        if self._ordered_depths is None:
            self._ordered_depths = self._depths.sort_values(ascending=False)
        
        if n == 1:
            return pd.Series(index=[list(self._ordered_depths.index)[0]], data=[self._ordered_depths.values[0]])
        else:
            return pd.Series(index=self._ordered_depths.index[0: n], data=self._ordered_depths.values[0: n])
    
    def outlying(self, n=1):
        if self._ordered_depths is None:
            self._ordered_depths = self._depths.sort_values(ascending=False)

        if n == 1:
            return pd.Series(index=[list(self._ordered_depths.index)[-1]], data=[self._ordered_depths.values[-1]])
        else:
            return pd.Series(index=self._ordered_depths.index[-n: ], data=self._ordered_depths.values[-n: ])

    def median(self):
        return self.deepest(n=1)
    
    def plot_deepest(self, n=1):
        s = self.deepest(n=n)
        cols = self._orig_data.columns
        x= self._orig_data.index

        data=[go.Scatter(x=x, y=self._orig_data[y], mode='lines+markers', marker_color='Blue') for y in cols]
        data.extend([go.Scatter(x=x, y=self._orig_data[y], mode='lines+markers', marker_color='Red') for y in s.index])

        fig = go.Figure(data=data)
        fig.update_layout(showlegend=False)

        fig.show()

class _BandDepthDataFrame(pd.DataFrame):
    def __init__(self, depths):
        super().__init__(data=depths)
        self._depths = depths

def BandDepth(data: Union[List[pd.DataFrame], Dict[str, pd.DataFrame]], K=None, J=2, containment='r2', relax=False, deep_check=False):
    if K is not None:
        depth = _samplebanddepth(data=data, K=K, J=J, containment=containment, relax=relax, deep_check=deep_check)
    else:
        depth = _banddepth(data=data, J=J, containment=containment, relax=relax, deep_check=deep_check)

    if isinstance(depth, pd.DataFrame):
        return _BandDepthDataFrame(depth)
    else:
        return _BandDepthSeries(df=data[0], depths=depth)
