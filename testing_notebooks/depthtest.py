import pandas as pd 
import numpy as np

from statdepth import FunctionalDepth

df = pd.DataFrame(np.random.rand(5, 10), columns=list('ZXCVBNMLKJ'))

bd = FunctionalDepth([df], J=2, relax=True)
print(bd.ordered())
print(bd.median())
