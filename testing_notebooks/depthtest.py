import pandas as pd 
import numpy as np

from statdepth import FunctionalDepth, PointwiseDepth


cols = ['size', 'co_amount', 'weight']
index = ['00:00', '00:30', '01:00', '01:30', '02:00', '02:30']
df1 = pd.DataFrame(np.random.randint(0,6,size=(30, 2)), )

# data = [df1, df2, df3, df4, df5, df6, df7, df8, df9]
bd = PointwiseDepth(data=df1, K=5)

print(bd.ordered())

# cd = PointwiseDepth(data=df)
# print(cd)
