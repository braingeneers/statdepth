import pandas as pd 
from typing import Callable, List, Union

from ._depthcalculations import _banddepth, _samplebanddepth

class _BandDepthSeries(pd.Series):

    def __init__(self, depths):
        super().__init__(data=depths)

        self._depths = depths
        self._ordered_depths_ = None
        self._deepest_curve_ = None

    def ordered(self, ascending=False):
        if self._ordered_depths_ is None:
            self._ordered_depths_ =  self._depths.sort_values(ascending=ascending)
        return self._ordered_depths_

    def deepest(self):
        if self._ordered_depths_ is not None or (self._ordered_depths_ is None and self._deepest_curve_ is None):
            sorted_depths = self._depths.sort_values(ascending=False)
            self._deepest_curve_ = pd.Series(index=[sorted_depths.index[0]], data=[sorted_depths[0]])
        return self._deepest_curve_
    
class _BandDepthDataFrame(pd.DataFrame):
    def __init__(self):
        pass

def BandDepth(data: List[pd.DataFrame], K=None, J=2, containment='r2', relax=False, deep_check=False):
    if K is not None:
        data = _samplebanddepth(data=data, K=K, J=J, containment=containment, relax=relax, deep_check=deep_check)
    else:
        data = _banddepth(data=data, J=J, containment=containment, relax=relax, deep_check=deep_check)

    if isinstance(data, pd.DataFrame):
        return _BandDepthDataFrame(data)
    else:
        return _BandDepthSeries(data)
