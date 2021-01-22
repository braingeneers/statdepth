import pandas as pd 
import numpy as np 
from typing import List, Union

from ..depth.depth import FunctionalDepth, PointcloudDepth

def FunctionalHomogeneity(F: List[pd.DataFrame], G: List[pd.DataFrame], K=None, J=2, containment='r2', method='p1', relax=False, deep_check=False):
    _handle_errors(F, G, method) 

    G_depths = FunctionalDepth(data=G, K=K, J=J, containment=containment, relax=relax, deep_check=deep_check)

    # Univariate case
    if len(F) == 1:
        F = F[0]
        G = G[0]

        # Get deepest function in G
        G_deepest = G_depths.get_deep_data(n=1)

        # Append this to F and calculate it's depth with respect to the other samples in F
        F.loc[:, 'g_deepest'] = G_deepest
        G_deep_in_F = FunctionalDepth([F]).ordered().loc['g_deepest']
        F = F.drop('g_deepest', axis=1)

        if method == 'p1':
            return G_deep_in_F
        elif method == 'p2':
            F_depths = FunctionalDepth([F])
            return np.abs(G_deep_in_F - F_depths.median().values[0])
        elif method == 'p3':
            t = []
            print(f'G columns are {G.columns}')
            # Get the deepest curve of G in F
            for col in G.columns:
                F.loc[:, col] = G.loc[:, col]
                t.append(FunctionalDepth([F], to_compute=[col]).values[0])
                F = F.drop(col, axis=1)

            # Sort depths of G in F
            depths_G_in_F = pd.Series(index=list(G.columns), data=t).sort_values(ascending=False)

            return depths_G_in_F.iloc[0]
        elif method == 'p4':
            pass
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

