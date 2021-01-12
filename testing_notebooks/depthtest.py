import pandas as pd 
import numpy as np

from statdepth import FunctionalDepth, PointwiseDepth


cols = ['size', 'co_amount', 'weight']
index = ['00:00', '00:30', '01:00', '01:30', '02:00', '02:30']
df1 = pd.DataFrame(np.random.randint(0,4,size=(6, 3)), columns=cols, index=index)
df2 = pd.DataFrame(np.random.randint(0,4,size=(6, 3)), columns=cols, index=index)
df3 = pd.DataFrame(np.random.randint(0,4,size=(6, 3)), columns=cols, index=index)
df4 = pd.DataFrame(np.random.randint(0,4,size=(6, 3)), columns=cols, index=index)
df5 = pd.DataFrame(np.random.randint(0,4,size=(6, 3)), columns=cols, index=index)
df6 = pd.DataFrame(np.random.randint(0,4,size=(6, 3)), columns=cols, index=index)

print(df1)
print(df2)
print(df3)

data = [df1, df2, df3, df4, df5, df6]
bd = FunctionalDepth(data, J=2, relax=True, containment='simplex')

print(bd.ordered())

# cd = PointwiseDepth(data=df)
# print(cd)