import pandas as pd 
import numpy as np 
from typing import List, Union

from ..depth.depth import FunctionalDepth, PointcloudDepth

def FunctionalHomogeneity(F: List[pd.DataFrame], G: List[pd.DataFrame], K=None, J=2, containment='r2', hom_method='p1', relax=False, deep_check=False):
    _handle_errors(F, G, hom_method) 

    G_depths = FunctionalDepth(data=G, K=K, J=J, containment=containment, relax=relax, deep_check=deep_check)
    
    # Univariate case
    if len(F) == 1:
        F = F[0]
        G = G[0]
        G_deepest = G_depths.get_deep_data(n=1)
        F.loc[:, 'g_deepest'] = G_deepest

        G_deep_in_F = FunctionalDepth(F, ['g_deepest'], K, J, containment, relax, deep_check).depths()['g_deepest']

        if hom_method == 'p1':
            return G_deep_in_F
        elif hom_method == 'p2':
            F_depths = FunctionalDepth(F, K, J, containment, relax, deep_check)
            return np.abs(G_deep_in_F - F_depths.median()[0])
    else:
        G_deepest = G[G_depths.median()[0]] # Get deepest DataFrame

        


def PointcloudHomogeneity(F: pd.DataFrame, G: pd.DataFrame, method='p1'):
    pass


def _handle_errors(F: List[pd.DataFrame], G: List[pd.DataFrame], method='p1'):
    if len(F) != len(G):
        raise ValueError('F and G must have data of the same length')
    
    if len(F) == 1: # Univariate error handling
        if F[0].shape[0] != G[0].shape[0]:
            raise ValueError('Univariate data must have same number of time indices to check containment.')
    else:
        pass

