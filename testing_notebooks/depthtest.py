import pandas as pd 
import numpy as np

from statdepth import FunctionalDepth, PointwiseDepth


# data = [pd.DataFrame(np.random.randint(0,30,size=(30, 3))) for _ in range(7)]
df = pd.DataFrame(np.random.rand(15, 6))

bd = FunctionalDepth([df], J=2, relax=True, containment='r2')
print(bd.ordered())
print(bd.median())

# cd = PointwiseDepth(data=df)
# print(cd)