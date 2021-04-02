import statdepth
import numpy as np
import pandas as pd 

from statdepth import FunctionalDepth
from statdepth.testing import generate_noisy_univariate
from tqdm import tqdm


class ClusterSet:
    def __init__(self, data=list()):
        self.clusters = data
    
    def add_cluster(self, data):
        self.clusters.append(data)
    
    def get_cluster(self, position):
        return self.clusters[position].copy()
    
    def clusters(self):
        return self.clusters
    
    def set_cluster(self, idx, data):
        self.clusters[idx] = data
        
    def __len__(self):
        return len(self.clusters)
    
    def __iter__(self):
        return iter(self.clusters)
    
    def display(self, cluster=None):
        if cluster is not None:
            display(self.clusters[cluster])
        for k, c in enumerate(self.clusters):
            print(f'cluster {k}')
            print(c)
    
    def move_function(from_cluster, to_cluster):
        pass
    
    def copy(self):
        return [df.copy() for df in self.clusters]
        
def init_clusterset(data: pd.DataFrame, k: int):
    n = data.shape[1]
    cset = ClusterSet()
    
    for i in range(k):
        sample = data.sample(n=n // k, axis=1)
        data = data.drop(sample.columns, axis=1)
        cset.add_cluster(sample)
        
    return cset

def cluster(
    data, 
    k, 
    tol=1/(10**6), 
    max_iter=25,
    quiet=False
) -> ClusterSet:
    cset = init_clusterset(data, k)

    for l in tqdm(range(max_iter), disable=quiet):
        for j, cluster in enumerate(cset):
            # print(f'NOW DOING CLUSTER {j}')
            # print(f'CLUSTER {j} HAS COLUMNS {cluster.columns}')
            to_move = {}
            for f in cluster.columns:
                depths = []
                for i in range(k):
                    cdf = cset.get_cluster(i).copy()
                    if f not in cdf.columns:
                        cdf.loc[:, f] = cluster.loc[:, f]
                    d = float(FunctionalDepth([cdf], to_compute=[f], K=1))
                    depths.append(d)
                # Get index of largest depth from depths to reassign the current curve
                ldepth = depths.index(max(depths))
                # print(f'ldepth for {f} is {ldepth}')
                # Move the function to the cluster in which it is deepest, if that cluster is not the current one 
                if ldepth != j:
                    # d = cset.get_cluster(ldepth).copy()
                    # d.loc[:, f] = cluster.loc[:, f]
                    # cset.set_cluster(ldepth, d)
                    to_move[ldepth] = f

            # print(f'To move is {to_move}')
            # print(cluster)
            for ldepth, f in to_move.items():
                # print(f'ldepth is {ldepth} and f is {f}')
                # print(f'ldepth is {ldepth} and f is {f}')
                # print('got here')
                # print(cluster)
                d = cset.get_cluster(ldepth)
                d[f] = cluster.loc[:, f]
                cset.set_cluster(ldepth, d)

            # print(f'Therefore, cluster {j} becomes \n {cluster.drop(to_move.values(), axis=1)}')
            # Remove from the current cluster if the best cluster isn't the current cluster
            cset.set_cluster(j, cluster.drop(to_move.values(), axis=1))
        # cset.display()
                
    if l == max_iter:
        print('Warning: max iterations reached, convergence is not guaranteed')
        
    return cset

def gen_tight_clusters(start, columns):
    multipliers = [0.8, 0.9, 1, 1.1, 1.2]
    df = pd.DataFrame()
    for m, col in zip(multipliers, columns):
        df[col] = start * m

    return df

# l1 = np.array([1,1.5,2,1,1.3,1])
# l2 = np.array([3,2,5,7,5,6])
# l3 = np.array([12, 10, 9, 8, 9, 12])

# df1 = gen_tight_clusters(l1, columns=list(range(0, 5)))
# df2 = gen_tight_clusters(2*l2, columns=list(range(5, 10)))
# df3 = gen_tight_clusters(3*l3, columns=list(range(10, 15)))
# l1 = np.array([1,1.5,2,1,1.3,1])
l1 = np.ones(6)
l2 = np.array([3,2,5,7,5,6])
l3 = np.array([12, 10, 9, 8, 9, 12])

df1 = gen_tight_clusters(3*l1, columns=list(range(0, 5)))
df2 = gen_tight_clusters(l2, columns=list(range(5, 10)))
df3 = gen_tight_clusters(l3, columns=list(range(10, 15)))



df = df1.join(df2).join(df3)

s = cluster(df, 3)

s.display()