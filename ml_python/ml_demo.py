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

a = np.array([4,5,6])
b = np.array([7,8,9])

np.inner(a,b)
np.outer(a,b)
a*b
proj_b_a = np.inner(a,b)/np.inner(b,b)*b
print(proj_b_a)

#%%
import scipy

a = np.random.randn(3,4)
p,l,u = scipy.linalg.lu(a)
q,r = np.linalg.qr(a)

print(p)
print(l) #lower triangular matrix
print(u) #upper triangular matrix
print(q) #orthogonal matrix
print(r)

#%%
x = np.diagflat([[1,2],[3,4]])
L = np.linalg.cholesky(x)
print(L)

ei = np.random.randn(4,4)
w,v = np.linalg.eig(ei)

print(w) #eigenvalues
print(v) #eigenvectors

a = np.random.randn(3,4)
pi = np.linalg.pinv(a)
print(pi) #psuedo inverse. for non-square matrix

#%%
svd = np.random.rand(4,5)
u,s,vh = np.linalg.svd(svd) #Singular Value Decomposition. I have no idea what this is tbh.

print(u)
print(s)
print(vh)

a = np.array([4,5,6])
L3 = np.linalg.norm(a, ord=3)
x = np.random.rand(2,3)
fro = np.linalg.norm(x,ord='fro') #Frobenius norm..
print(L3)
print(fro)

#%%
x = np.random.rand(2,3)
y = np.random.rand(5,3)

ein = np.einsum('ij,kj -> ik',x,y)

print(x)
print(y)
print(ein)



#%%
x = np.random.rand(2,3)
tp = np.einsum('ij -> ji', x) # transpose
su = np.einsum('ij ->', x) #sum
csum = np.einsum('ij -> i', x) #column sum
rsum = np.einsum('ij -> j', x) #row sum

print(tp)
print(su)
print(csum)
print(rsum)

x = np.random.rand(2,3)
y = np.random.rand(5,3)
mm = np.einsum('ij,kj -> ik', x,y) #matrix multiplication

print(mm)

a = np.array([4,5,6])
b = np.array([7,8,9])

ip = np.einsum('i,i ->', a,b) #inner product
op = np.einsum('i,j -> ij', a,b) #outer product
hp = np.einsum('i,i -> i', a,b) #hadamard product

print(ip)
print(op)
print(hp)

y = np.random.rand(5,3)
di = np.einsum('ij -> j', y) #diagonal
tr = np.einsum('ij ->', y) #trace

print(di)
print(tr)

#%%
help(np.einsum)






