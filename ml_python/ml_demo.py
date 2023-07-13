#%%

import numpy as np

a = np.random.randn(2,3)
a
a.T

b = np.random.randn(3,2)
b

a@b

#%%
x = np.random.rand(4,4)
np.linalg.det(x)
