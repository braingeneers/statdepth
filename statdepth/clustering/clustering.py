import numpy as np 
import pandas as pd 

from ..depth.depth import *

class ClusterSet:
    def __init__(self, data=list()):
        self.clusters = data
    
    def add_cluster(self, data):
        self.clusters.append(data)
    
    def get_cluster(self, position):
        return self.clusters[position]
    
    def clusters(self):
        return self.clusters
    
    def set_cluster(self, idx, data):
        self.clusters[idx] = data
        
    def __len__(self):
        return len(self.clusters)
    
    def __iter__(self):
        for df in self.clusters:
            yield df 
    
    def display(self, cluster=None):
        if cluster is not None:
            display(self.clusters[cluster])
        for c in self.clusters:
            display(c)
    
def init_clusterset(data: pd.DataFrame, k: int):
    n = data.shape[1]
    cset = ClusterSet()
    
    for i in range(k):
        sample = data.sample(n=n // k, axis=1, replace=False)
        cset.add_cluster(sample)
        
    return cset
