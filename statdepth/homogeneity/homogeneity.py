import pandas as pd 
import numpy as np 
from typing import List, Union

from ..depth.depth import FunctionalDepth, PointcloudDepth
from ..depth.abstract import AbstractDepth

# Helper class 
class FunctionalHomogeneity:
    def __init__(self, F, G, method='p1', K=None, J=2, containment='r2', relax=False, deep_check=False, quiet=False):
        self._orig_F = F 
        self._orig_G = G 
        self._hom = _functionalhomogeneity(
            F=F,
            G=G,
            K=K,
            J=J,
            containment=containment,
            method=method,
            relax=relax,
            deep_check=deep_check,
            quiet=quiet
        )
    
    def __str__(self):
        return self._hom 
    
    def homogeneity(self):
        return self._hom 

    def __str__(self):
        return str(self.homogeneity())
    
    def __repr__(self):
        return str(self.homogeneity())

class PointcloudHomogeneity:
    def __init__(self, F, G, method='p1', K=None, J=None, containment='simplex', relax=False, deep_check=False):
        self._orig_F = F
        self._orig_G = G
        
        self._F_depths, self._G_depths, self._hom  = _pointcloudhomogeneity(
            F=F,
            G=G,
            K=K,
            containment=containment,
            method=method,
        )

    def F_depths(self):
        return self._F_depths
    
    def G_depths(self):
        return self._G_depths

    def homogeneity(self):
        return self._hom 
    
    def __str__(self):
        return str(self.homogeneity())
    
    def __repr__(self):
        return str(self.homogeneity())    

def _functionalhomogeneity(
    F: List[pd.DataFrame], 
    G: List[pd.DataFrame], 
    K=None, 
    J=2, 
    containment='r2', 
    method='p1', 
    relax=False,
    deep_check=False,
    quiet=False
):
    _handle_errors(F, G, method) 

    # Compute depths of G, needed in either case
    G_depths = FunctionalDepth(
        data=G,
        K=K,
        J=J,
        containment=containment,
        relax=relax,
        deep_check=deep_check,
        quiet=quiet,
    )
    
    # Univariate case
    if len(F) == 1:
        F = F[0]
        G = G[0]
        
        # Get deepest function in G
        G_deepest = G_depths.get_deep_data(n=1)

        if 'g_deepest' in F.columns:
            F = F.drop('g_deepest', axis=1)

        # Append this to F and calculate it's depth with respect to the other samples in F
        F.loc[:, 'g_deepest'] = G_deepest
        
        G_deep_in_F = FunctionalDepth(
            [F],
            to_compute=['g_deepest'],
            K=K,
            J=J,
            containment=containment,
            relax=relax,
            deep_check=deep_check,
            quiet=quiet,
        )
        
        F = F.drop('g_deepest', axis=1)

        if method == 'p1':
            return G_deep_in_F 
        elif method == 'p2':
            F_depths = FunctionalDepth([F], K=K, J=J, containment=containment, relax=relax, deep_check=deep_check, quiet=quiet)
            return np.abs(G_deep_in_F - F_depths.median().iloc[0])
        elif method == 'p3':
            t = []

            # Get the deepest curve of G in F
            for col in G.columns:
                F.loc[:, col] = G.loc[:, col]
                t.append(FunctionalDepth([F], to_compute=[col],  K=K, J=J, containment=containment, relax=relax, deep_check=deep_check).loc[col])
                F = F.drop(col, axis=1)

            # Sort depths of G in F
            depths_G_in_F = pd.Series(index=list(G.columns), data=t).sort_values(ascending=False)

            return depths_G_in_F.iloc[0] / G_depths.median().iloc[0]
        elif method == 'p4':
            raise NotImplementedError()
        else:
            raise ValueError(f'{method} is not a valid depth method for the given data. Use one of [\'p1\', \'p2\', \'p3\', \'p4\']')
    else:
        G_deepest = G[G_depths.index[0]] # Get deepest dataframe in G

        F.append(G_deepest)
        G_deep_in_F = FunctionalDepth(F, to_compute=[len(F) - 1], K=K, J=J, containment=containment, relax=relax, deep_check=deep_check).ordered().iloc[0]
        F.pop(-1) # Remove G after we added it 

        if method == 'p1':
            return G_deep_in_F / G_depths.median().iloc[0]
        elif method == 'p2':
            F_depths = FunctionalDepth(F, K, J, containment, relax, deep_check)
            return 1 - np.abs(G_deep_in_F - F_depths.median().iloc[0])
        elif method == 'p3':
            pass
        else:
            raise ValueError(f'{method} is not a valid depth method for the given data. Use one of [\'p1\', \'p2\', \'p3\', \'p4\']')

def _pointcloudhomogeneity(
    F: pd.DataFrame, 
    G: pd.DataFrame, 
    K=None, 
    containment='simplex', 
    method='p1', 
):
    _handle_errors(F, G, method) 

    G_depths = PointcloudDepth(data=G, K=K, containment=containment)
    F_depths = PointcloudDepth(data=F, K=K, containment=containment)
    hom = 0

    # Get deepest function in G
    G_deepest = G_depths.get_deep_data(n=1)
    G_deepest.index = ['g_deepest']

    # Append this to F and calculate it's depth with respect to the other samples in F
    F = F.append(G_deepest)
    G_deep_in_F = PointcloudDepth(F, to_compute=['g_deepest'], K=K, containment=containment).ordered().loc['g_deepest']
    F = F.drop('g_deepest', axis=0)

    if method == 'p1':
        hom =  G_deep_in_F / F_depths.median().iloc[0] # Normalized value
    elif method == 'p2':
        hom = 1 - np.abs(G_deep_in_F - F_depths.median().iloc[0])
    elif method == 'p3':
        t = []

        # Get the deepest curve of G in F
        for point in G.index:
            F.loc[point, :] = G.loc[point, :]
            t.append(PointcloudDepth(F, to_compute=[point],  K=K, containment=containment).loc[point])
            F = F.drop(point, axis=0)

        # Sort depths of G in F
        depths_G_in_F = pd.Series(index=list(G.index), data=t).sort_values(ascending=False)

        hom = depths_G_in_F.iloc[0] / G_depths.median().iloc[0]
    elif method == 'p4':
        t1 = np.abs(_pointcloudhomogeneity(F, G, K, containment, 'p3') - _pointcloudhomogeneity(F, F, K, containment, 'p1'))
        t2 = np.abs(_pointcloudhomogeneity(F, G, K, containment, 'p3') - _pointcloudhomogeneity(G, G, K, containment, 'p1'))
        hom = t1 * t2
    else:
        raise ValueError(f'{method} is not a valid depth method for the given data. Use one of [\'p1\', \'p2\', \'p3\', \'p4\']')
    
    return F_depths, G_depths, hom

def _handle_errors(F: List[pd.DataFrame], G: List[pd.DataFrame], method='p1'):
    if len(F) != len(G):
        raise ValueError('F and G must have data of the same length')
    
    if len(F) == 1: # Univariate error handling
        if F[0].shape[0] != G[0].shape[0]:
            raise ValueError('Univariate data must have same number of time indices to check containment.')
    else:
        pass


def P1_homogeneity(
    F: pd.DataFrame, 
    G: pd.DataFrame,
    K=None, 
    J=2, 
    containment='r2', 
    relax=False,
    quiet=False
) -> float:
    '''
    Calculates the P1 homogeneity coefficient between the samples F and G.    
    Closer to 1 means the two experiments were likely generated by the same process.
    
    Parameters:
    ----------
    F: pd.DataFrame: An m x n DataFrame where each n functions are sampled at m points
    G: pd.DataFrame: An m x n DataFrame where each n functions are sampled at m points

    Returns:
    ----------
    float: P1 homogeneity coefficient
    '''
    
    # Find the deepest function in G
    G_depth = FunctionalDepth(
        data=[G],
        K=K,
        J=J,
        containment=containment,
        relax=relax, 
        quiet=quiet,
    )
    
    G_deepest = G_depth.get_deep_data()

    F.loc[:, 'G_deepest'] = G_deepest
    
    G_deep_in_F = FunctionalDepth(
        [F],
        to_compute=['G_deepest'],
        K=K,
        J=J,
        containment=containment,
        relax=relax,
        quiet=quiet
        
    )
    
    return G_deep_in_F.iloc[0]

def P2_homogeneity(
    F: pd.DataFrame, 
    G: pd.DataFrame,
    K=None, 
    J=2, 
    containment='r2', 
    relax=False,
    quiet=False
) -> float:
    '''
    Calculates the P2 homogeneity coefficient between the samples F and G.
    Closer to 0 means the two experiments likely are generated by the same process.
    
    Parameters:
    ----------
    F: pd.DataFrame: An m x n DataFrame where each n functions are sampled at m points
    G: pd.DataFrame: An m x n DataFrame where each n functions are sampled at m points

    
    Returns:
    ----------
    float: P2 homogeneity coefficient
    '''

    P1_F_G = P1_homogeneity(
        F=F,
        G=G,
        K=K,
        J=J,
        containment=containment,
        relax=relax,
        quiet=quiet
    )

    P1_F_F = FunctionalDepth(
        data=[F],
        K=K,
        J=J,
        containment=containment,
        relax=relax,
        quiet=quiet
    ).deepest().iloc[0]
    
    return np.abs(P1_F_G - P1_F_F)