import pandas as pd 
import numpy as np

from statdepth import FunctionalDepth, PointcloudDepth
from statdepth.testing import generate_noisy_univariate, generate_noisy_multivariate, generate_noisy_pointcloud 

df = generate_noisy_univariate()
bd = FunctionalDepth([df], K=2)

print(f'univariate median is {bd.median()}')

data = generate_noisy_multivariate(num_curves=8)
bd = FunctionalDepth(data, containment='simplex', quiet=False)

print(f'multivariate depth is {bd.ordered()}')

df = generate_noisy_pointcloud()
bd = PointcloudDepth(df, K=10, quiet=False)
print(f'pointcloud median is {bd.median()}')

