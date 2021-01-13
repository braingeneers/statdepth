import pandas as pd 
import numpy as np

from statdepth import FunctionalDepth, PointwiseDepth


cols = ['size', 'co_amount', 'weight']
index = ['00:00', '00:30', '01:00', '01:30', '02:00', '02:30']
df1 = pd.DataFrame(np.random.randint(0,2,size=(6, 3)), columns=cols, index=index)
# df2 = pd.DataFrame(np.random.randint(0,2,size=(6, 3)), columns=cols, index=index)
# df3 = pd.DataFrame(np.random.randint(0,2,size=(6, 3)), columns=cols, index=index)
# df4 = pd.DataFrame(np.random.randint(0,2,size=(6, 3)), columns=cols, index=index)
# df5 = pd.DataFrame(np.random.randint(0,2,size=(6, 3)), columns=cols, index=index)
# df6 = pd.DataFrame(np.random.randint(0,2,size=(6, 3)), columns=cols, index=index)
# df7 = pd.DataFrame(np.random.randint(0,2,size=(6, 3)), columns=cols, index=index)
# df8 = pd.DataFrame(np.random.randint(0,2,size=(6, 3)), columns=cols, index=index)
# df9 = pd.DataFrame(np.random.randint(0,2,size=(6, 3)), columns=cols, index=index)

# data = [df1, df2, df3, df4, df5, df6, df7, df8, df9]
bd = FunctionalDepth([df1], J=2, relax=True, containment='simplex')

print(bd.ordered())

# cd = PointwiseDepth(data=df)
# print(cd)
