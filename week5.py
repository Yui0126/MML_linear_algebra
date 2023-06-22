
#%%

import numpy as np
import numpy.linalg as la

np.set_printoptions(suppress=True)

def generate_internet(n) :
    c = np.full([n,n], np.arange(n))
    c = (abs(np.random.standard_cauchy([n,n])/2) > (np.abs(c - c.T) + 1)) + 0
    c = (c+1e-10) / np.sum((c+1e-10), axis=0)
    return c

L = np.array([[0,   1/2, 1/3, 0, 0,   0 ],
              [1/3, 0,   0,   0, 1/2, 0 ],
              [1/3, 1/2, 0,   1, 0,   1/2 ],
              [1/3, 0,   1/3, 0, 1/2, 1/2 ],
              [0,   0,   0,   0, 0,   0 ],
              [0,   0,   1/3, 0, 0,   0 ]])

eVals, eVecs = la.eig(L) # Gets the eigenvalues and vectors
order = np.absolute(eVals).argsort()[::-1] # Orders them by their eigenvalues
eVals = eVals[order]
eVecs = eVecs[:,order]

# r = eVecs[:, 0] # Sets r to be the principal eigenvector
# 100 * np.real(r / np.sum(r)) # Make this eigenvector sum to one, then multiply by 100 Procrastinating Pats

# r = 100 * np.ones(6) / 6 # Sets up this vector (6 entries of 1/6 × 100 each)
# # r # Shows it's value

# for i in np.arange(100) : # Repeat 100 times
#     r = L @ r
# r

# r = 100 * np.ones(6) / 6 # Sets up this vector (6 entries of 1/6 × 100 each)
# lastR = r
# r = L @ r
# i = 0
# while la.norm(lastR - r) > 0.01 :
#     lastR = r
#     r = L @ r
#     i += 1
# print(str(i) + " iterations to convergence.")
# r

L2 = np.array([[0,   1/2, 1/3, 0, 0,   0, 0 ],
               [1/3, 0,   0,   0, 1/2, 0, 0 ],
               [1/3, 1/2, 0,   1, 0,   1/3, 0 ],
               [1/3, 0,   1/3, 0, 1/2, 1/3, 0 ],
               [0,   0,   0,   0, 0,   0, 0 ],
               [0,   0,   1/3, 0, 0,   0, 0 ],
               [0,   0,   0,   0, 0,   1/3, 1 ]])

# r = 100 * np.ones(7) / 7 # Sets up this vector (6 entries of 1/6 × 100 each)
# lastR = r
# r = L2 @ r
# i = 0
# while la.norm(lastR - r) > 0.01 :
#     lastR = r
#     r = L2 @ r
#     i += 1
# print(str(i) + " iterations to convergence.")
# r

d = 0.5 # Feel free to play with this parameter after running the code once.
M = d * L2 + (1-d)/7 * np.ones([7, 7]) # np.ones() is the J matrix, with ones for each entry.

r = 100 * np.ones(7) / 7 # Sets up this vector (6 entries of 1/6 × 100 each)
lastR = r
r = M @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = M @ r
    i += 1
print(str(i) + " iterations to convergence.")
r



#%%

# PACKAGE
# Here are the imports again, just in case you need them.
# There is no need to edit or submit this cell.
import numpy as np
import numpy.linalg as la

np.set_printoptions(suppress=True)

def generate_internet(n) :
    c = np.full([n,n], np.arange(n))
    c = (abs(np.random.standard_cauchy([n,n])/2) > (np.abs(c - c.T) + 1)) + 0
    c = (c+1e-10) / np.sum((c+1e-10), axis=0)
    return c

# GRADED FUNCTION
# Complete this function to provide the PageRank for an arbitrarily sized internet.
# I.e. the principal eigenvector of the damped system, using the power iteration method.
# (Normalisation doesn't matter here)
# The functions inputs are the linkMatrix, and d the damping parameter - as defined in this worksheet.
# (The damping parameter, d, will be set by the function - no need to set this yourself.)

M = generate_internet(5)


def pageRank(linkMatrix, d) :
    n = linkMatrix.shape[0] # returns size/dimention
    M = d * linkMatrix + (1-d)/n * np.ones([n, n])
    r = 100 * np.ones(n) / n
    lastR = r
    r = M @ r
    i = 0
    while la.norm(lastR - r) > 0.01 :
        lastR = r
        r = M @ r
        i += 1
    print(str(i) + " iterations to convergence.")
    return r

pageRank(M, 1)



#%%

M = np.array([[4,-5,6],
              [7,-8,6],
              [3/2,-1/2,-2]])

# vals, vecs = np.linalg.eig(M)
# vecs

L = np.array([[3/2, -1],
              [-1/2, 1/2]])

ls = L@L
ls

# vals, vecs = np.linalg.eig(L)
# vals = np.absolute(vals)
# vals
