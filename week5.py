
#%%
import numpy as np

A = [[7, 1],
     [-3, 0]]
Ainv = np.linalg.inv(A)

T = [[2, 7],
     [0, -1]]

Res = Ainv@T@A

print(Res)
