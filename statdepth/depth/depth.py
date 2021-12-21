# Copyright, Regents of the University of California, 2021, all rights reserved
import pandas as pd 
from typing import Callable, List, Union, Dict
import plotly.graph_objects as go

from .calculations._helper import *
from .calculations._functional import _functionaldepth, _samplefunctionaldepth
from .calculations._pointcloud import _pointwisedepth, _samplepointwisedepth
from .abstract import AbstractDepth

__all__ = ['FunctionalDepth', 'PointcloudDepth', 'ProbabilisticDepth']

# Private class that wraps the depth calculation methods with some extra attributes as well
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

    # Aliases for some abstract methods above
    def sorted(self, ascending=False):
        return self.ordered(ascending=ascending)

    def median(self):
        return self.deepest(n=1)

    def get_depths(self):
        return self._depths

    def get_data(self):
        return self._orig_data

    def depths(self):
        return self.get_depths()

class _FunctionalDepthMultivariateDataFrame(AbstractDepth, pd.DataFrame):
    # Don't copy over all the data. Numpy arrays might be large and this will add extra space complexity
    # Also, visualization for multivariate functions isn't extensive enough to justify 
    # storing all of the data
    def __init__(self, depths: pd.DataFrame):
        super().__init__(depths)
        self._depths = depths

    # Not inheritable from _FunctionalDepthSeries because depths is a DataFrame
    def ordered(self, ascending=False):
        pass

    def deepest(self, n=1):
        pass

    def outlying(self, n=1):
        pass
    
# Just extends the FunctionalDepthSeries class with extra plotting capabilities,
# Since in this case our data are real-valued functions
class _FunctionalDepthUnivariate(_FunctionalDepthSeries):
    def __init__(self, df: pd.DataFrame, depths: pd.Series):
        super().__init__(df=df, depths=depths)

    def _plot(
        self, 
        deep_or_outlying: pd.Series, 
        title: str, 
        xaxis_title: str,
        yaxis_title: str,
        return_plot: bool,
        showlegend: bool,
    ) -> None:
        cols = self._orig_data.columns
        x = self._orig_data.index

        # We use deep_or_outlying.index to get the columns because 
        # deep_or_outlying is a Series indexed by the original columns
        
        data = [go.Scatter(
            x=x, 
            y=self._orig_data.loc[:, y], 
            mode='lines', 
            name=y,
            line=dict(color='#6ea8ff', width=.5)) for y in cols.difference(deep_or_outlying.index)
        ]

        data.extend([go.Scatter(
                x=x, 
                y=self._orig_data.loc[:, y], 
                mode='lines',
                name=y,
                line=dict(color='Red', width=1)) for y in deep_or_outlying.index
            ]
        )

        fig = go.Figure(
            data=data, 
            layout=go.Layout(
                title=dict(text=title, y=0.9, x=0.5, xanchor='center', yanchor='top'),
                xaxis=dict(title=xaxis_title),
                yaxis=dict(title=yaxis_title),
            )
        )

        data = []

        fig.update_layout(showlegend=showlegend)
        
        if return_plot:
            return fig
        else:
            fig.show()

    def plot_deepest(
        self, n=1, 
        title=None, 
        xaxis_title = None,
        yaxis_title = None,
        return_plot=False,
        showlegend=False
    ) -> None:
        '''Plots all the data in blue and marks the n deepest in red'''
        return self._plot(
            deep_or_outlying=self.deepest(n=n), 
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            return_plot=return_plot,
            showlegend=showlegend,
        )

    def plot_outlying(
        self, n=1, 
        title=None, 
        xaxis_title = None,
        yaxis_title = None,
        return_plot=False,
        showlegend=False,
    ) -> None:
        '''Plots all the data in blue and marks the n most outlying curves in red'''
        return self._plot(
            deep_or_outlying=self.outlying(n=n), 
            title=title, 
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            return_plot=return_plot,
            showlegend=showlegend
        )

    # Have to redefine these because in the univariate case our samples are column based
    def drop_outlying_data(self, n=1) -> pd.DataFrame:
        return self._orig_data.drop(self.outlying(n=n).index, axis=1)
    
    def get_deep_data(self, n=1) -> pd.DataFrame:
        return self._orig_data.loc[:, self.deepest(n=n).index]

    def get_outlying_data(self, n=1) -> pd.DataFrame:
        return self._orig_data.loc[:, self.outlying(n=n).index]

class _PointwiseDepth(_FunctionalDepthSeries):
    def __init__(self, df: pd.DataFrame, depths: pd.Series):
        '''Pointwise depth calculation for Multivariate data. Calculates depth of each point with respect to the sample in R^n.'''
        super().__init__(df=df, depths=depths)

    # This method should be removed eventually
    def plot_depths(
        self, 
        invert_colors=False, 
        marker=None,
        return_plot=False,
        title='',
        xaxis_title=None, 
        yaxis_title=None,
     ) -> None:
        d = self._depths
        cols = self._orig_data.columns
        n = len(self._orig_data.columns)

        if invert_colors:
            d = 1 - d

        if marker is None:
            marker = dict(color=d, colorscale='viridis', size=7)

        if n > 3:
            self._plot_parallel_axis()
        elif n == 3:
            data=[
                go.Scatter3d(
                    x=self._orig_data.loc[:, cols[0]], 
                    y=self._orig_data.loc[:, cols[1]], 
                    z=self._orig_data.loc[:, cols[2]], 
                    mode='markers', 
                    marker=marker
                )
            ]

        elif n == 2:
            data=[
                go.Scatter(
                    x=self._orig_data.loc[:, cols[0]], 
                    y=self._orig_data.loc[:, cols[1]], 
                    mode='markers',
                    marker=marker
                )
            ]
        else: #n==1
            raise ValueError(f'Error: Dimensionality of data must be >=2. Value found is {n}')
        
        fig = go.Figure(
            data=data,
            layout=go.Layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
        )
        fig.update_layout(showlegend=False)

        return fig if return_plot else fig.show()
        
    def _plot_parallel_axis(self) -> None:
        pass 

    def _plot(
        self, 
        deep_or_outlying: pd.Series,
        return_plot: bool,
        title: str,
        xaxis_title: str, 
        yaxis_title: str,
    ) -> None:
        n = len(self._orig_data.columns)
        cols = self._orig_data.columns
        select = self._orig_data.loc[deep_or_outlying.index, :]
        
        if n > 3:
            self._plot_parallel_axis()
        elif n == 3:
            data=[
                go.Scatter3d(
                    x=self._orig_data[cols[0]], 
                    y=self._orig_data[cols[1]], 
                    z=self._orig_data[cols[2]], 
                    mode='markers', 
                    marker_color='blue', 
                    name=''
                ),
                go.Scatter3d(
                    x=select[cols[0]], 
                    y=select[cols[1]], 
                    z=select[cols[2]], 
                    mode='markers', 
                    marker_color='red', name=''
                )
            ]
            
        elif n == 2: 
            data=[
                go.Scatter(
                    x=self._orig_data[cols[0]], 
                    y=self._orig_data[cols[1]], 
                    mode='markers', 
                    marker_color='blue', 
                    name=''
                ),
                go.Scatter(
                    x=select[cols[0]], 
                    y=select[cols[1]], 
                    mode='markers', 
                    marker_color='red', 
                    name=''
                )
            ]
        else: # n = 1, plot number line maybe
            raise ValueError(f'Error: Dimensionality of data must be >=2. Value found is {n}')
        
        fig = go.Figure(
            data=data, 
            layout=go.Layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
        )
        fig.update_layout(showlegend=False)

        return fig if return_plot else fig.show()

    def plot_deepest(self, n=1,
        return_plot=False,
        title='',
        xaxis_title=None, 
        yaxis_title=None,
    ) -> None:
        return self._plot(
            deep_or_outlying=self.deepest(n=n),
            return_plot=return_plot,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
        )

    def plot_outlying(self, n=1,
        return_plot=False,
        title='',
        xaxis_title=None, 
        yaxis_title=None,
    ) -> None:
        return self._plot(
            deep_or_outlying=self.outlying(n=n),
            return_plot=return_plot,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
        )
     
    def drop_outlying_data(self, n=1) -> pd.DataFrame:
        return self._orig_data.drop(self.outlying(n=n).index, axis=0)
    
    def get_deep_data(self, n=1) -> pd.DataFrame:
        return self._orig_data.loc[self.deepest(n=n).index, :]

    def plot_distribution(self, invert_colors=False, marker=None) -> None:
        self.plot_depths(invert_colors, marker)

# Wraps the PointwiseDepth class in a function, because we need to compute depths before we pass down to the class
def PointcloudDepth(
    data: pd.DataFrame, 
    to_compute: pd.Index=None, 
    K=None, 
    containment='simplex', 
    quiet=True,
) -> _PointwiseDepth:
    if K is not None:
        depth = _samplepointwisedepth(data=data, to_compute=to_compute, K=K, containment=containment)
    else:
        depth = _pointwisedepth(data=data, to_compute=to_compute, containment=containment)
    
    return _PointwiseDepth(df=data, depths=depth)

# Wraps FunctionalDepth classes in a function, because we need to compute depths before we pass down to the class
def FunctionalDepth(
    data: List[pd.DataFrame], 
    to_compute=None, 
    K=None, 
    J=2, 
    containment='r2', 
    relax=False, 
    deep_check=False, 
    quiet=True
) -> Union[_FunctionalDepthSeries, _FunctionalDepthUnivariate, _FunctionalDepthMultivariateDataFrame]:   

    # Compute band depth completely or sample band depth
    if K is not None:
        depth = _samplefunctionaldepth(
            data=data,
            to_compute=to_compute, 
            K=K, 
            J=J, 
            containment=containment, 
            relax=relax, 
            deep_check=deep_check,
            quiet=quiet
        )
    else:
        depth = _functionaldepth(
            data=data, 
            to_compute=to_compute, 
            J=J, 
            containment=containment, 
            relax=relax, 
            deep_check=deep_check,
            quiet=quiet
        )

    # Return the appropriate class
    if isinstance(depth, pd.DataFrame): 
        return _FunctionalDepthMultivariateDataFrame(depths=depth)
    elif len(data) == 1: # Univariate case (by assumption)
        return _FunctionalDepthUnivariate(df=data[0], depths=depth)
    else: # Multivariate case
        return _FunctionalDepthSeries(df=data[0], depths=depth)