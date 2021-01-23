import pandas as pd 
import numpy as np 
from typing import List, Union

from ..depth.depth import FunctionalDepth, PointcloudDepth

def FunctionalHomogeneity(
    F: List[pd.DataFrame], 
    G: List[pd.DataFrame], 
    K=None, 
    J=2, 
    containment='r2', 
    method='p1', 
    relax=False, 
    deep_check=False
):
    _handle_errors(F, G, method) 

    G_depths = FunctionalDepth(data=G, K=K, J=J, containment=containment, relax=relax, deep_check=deep_check)
    
    # Univariate case
    if len(F) == 1:
        F = F[0]
        G = G[0]
        
        # Get deepest function in G
        G_deepest = G_depths.get_deep_data(n=1)

        # Shitty hacky fix, will change this. Sorry if someone is reading this in the future and I didn't lol
        if 'g_deepest' in F.columns:
            raise ValueError('Cannot have column named g_deepest')

        # Append this to F and calculate it's depth with respect to the other samples in F
        F.loc[:, 'g_deepest'] = G_deepest
        G_deep_in_F = FunctionalDepth([F], to_compute=['g_deepest'], K=K, J=J, containment=containment, relax=relax, deep_check=deep_check).ordered().loc['g_deepest']
        F = F.drop('g_deepest', axis=1)

        if method == 'p1':
            return G_deep_in_F / G_depths.median().iloc[0]
        elif method == 'p2':
            F_depths = FunctionalDepth([F], K=K, J=J, containment=containment, relax=relax, deep_check=deep_check)
            return 1 - np.abs(G_deep_in_F - F_depths.median().iloc[0])
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
            t1 = np.abs(self.PointcloudHomogeneity(F, G, K, J, containment, 'p3', relax, deepcheck) - self.PointcloudHomogeneity(F, F, K, J, containment, 'p1', relax, deepcheck))
            t2 = np.abs(self.PointcloudHomogeneity(F, G, K, J, containment, 'p3', relax, deepcheck) - self.PointcloudHomogeneity(G, G, K, J, containment, 'p1', relax, deepcheck))
            return t1 * t2
        else:
            raise ValueError(f'{method} is not a valid depth method for the given data. Use one of [\'p1\', \'p2\', \'p3\', \'p4\']')
    else:
        G_deepest = G[G_depths.index[0]] # Get deepest dataframe in G

        F.append(G_deepest)
        G_deep_in_F = FunctionalDepth(F, to_compute=[len(F) - 1], K, J, containment, relax, deep_check).ordered().iloc[0]
        F.pop(-1) # Remove G after we added it 

        if method == 'p1':
            return G_deep_in_F / G_depths.median().iloc[0]
        elif method == 'p2':
            F_depths = FunctionalDepth(F, K, J, containment, relax, deep_check)
            reurn 1 - np.abs(G_deep_in_F - F_depths.median().iloc[0])
        elif method == 'p3':
            pass
        else:
            raise ValueError(f'{method} is not a valid depth method for the given data. Use one of [\'p1\', \'p2\', \'p3\', \'p4\']')



def PointcloudHomogeneity(
    F: List[pd.DataFrame], 
    G: List[pd.DataFrame], 
    K=None, 
    J=2, 
    containment='r2', 
    method='p1', 
    relax=False, 
    deep_check=False
):
    _handle_errors(F, G, method) 

    G_depths = PointcloudDepth(data=G, K=K, J=J, containment=containment, relax=relax, deep_check=deep_check)
    
    # Get deepest function in G
    G_deepest = G_depths.get_deep_data(n=1)

    # Append this to F and calculate it's depth with respect to the other samples in F
    F.loc['g_deepest', :] = G_deepest
    G_deep_in_F = PointcloudDepth(F, to_compute=['g_deepest'], K=K, J=J, containment=containment, relax=relax, deep_check=deep_check).ordered().loc['g_deepest']
    F = F.drop('g_deepest', axis=0)

    if method == 'p1':
        return G_deep_in_F / G_depths.median().iloc[0]
    elif method == 'p2':
        F_depths = PointcloudDepth(F, K=K, J=J, containment=containment, relax=relax, deep_check=deep_check)
        return 1 - np.abs(G_deep_in_F - F_depths.median().iloc[0])
    elif method == 'p3':
        t = []

        # Get the deepest curve of G in F
        for point in G.index:
            F.loc[point, :] = G.loc[point, :]
            t.append(PointcloudDepth(F, to_compute=[point],  K=K, J=J, containment=containment, relax=relax, deep_check=deep_check).loc[point])
            F = F.drop(point, axis=0)

        # Sort depths of G in F
        depths_G_in_F = pd.Series(index=list(G.index), data=t).sort_values(ascending=False)

        return depths_G_in_F.iloc[0] / G_depths.median().iloc[0]
    elif method == 'p4':
        t1 = np.abs(self.PointcloudHomogeneity(F, G, K, J, containment, 'p3', relax, deepcheck) - self.PointcloudHomogeneity(F, F, K, J, containment, 'p1', relax, deepcheck))
        t2 = np.abs(self.PointcloudHomogeneity(F, G, K, J, containment, 'p3', relax, deepcheck) - self.PointcloudHomogeneity(G, G, K, J, containment, 'p1', relax, deepcheck))
        return t1 * t2
    else:
        raise ValueError(f'{method} is not a valid depth method for the given data. Use one of [\'p1\', \'p2\', \'p3\', \'p4\']')


def _handle_errors(F: List[pd.DataFrame], G: List[pd.DataFrame], method='p1'):
    if len(F) != len(G):
        raise ValueError('F and G must have data of the same length')
    
    if len(F) == 1: # Univariate error handling
        if F[0].shape[0] != G[0].shape[0]:
            raise ValueError('Univariate data must have same number of time indices to check containment.')
    else:
        pass

