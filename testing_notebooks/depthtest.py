import pandas as pd 
import numpy as np

from statdepth import FunctionalDepth


data = [pd.DataFrame(np.random.randint(0,30,size=(30, 3))) for _ in range(7)]

bd = FunctionalDepth(data, J=2, relax=True, containment='simplex')
print(bd.ordered())
print(bd.median())
