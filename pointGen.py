from numba import cuda
from Frep import union
import numpy as np
import userInput as u
try: TPB = u.TPB 
except: TPB = 8

@cuda.reduce
def sum_reduce(a, b):
    return a + b

@cuda.jit
def genRandPointsKernel(d_u, d_r, d_v, threshold):
    i,j, k = cuda.grid(3)
    m,n,p = d_u.shape
    if i < m and j < n and k < p:
        if d_u[i,j,k] < 0 and d_r[i,j,k] < threshold/abs(d_u[i,j,k]):
            #Uses a simple function for the thresholding calculation, 
            #threshold/abs(d_u[i,j,k]) can be replaced with any desired function.
            d_v[i,j,k] = 0

def genRandPoints(u,threshold):
    #u = Voxel model of boundary object.
    #threshold = normalized value to determine how likely it is for each voxel to have a point placed in it.
    #Outputs a matrix with random points within the boundaries of object u.  The random points are set to 0 while the rest of the matrix is ones.
    x,y,z = u.shape
    threshold=threshold/max(x,y,z) 
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    d_r = cuda.to_device(np.random.rand(x,y,z))
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(np.ones(u.shape)) #Generates a matrix for us to plot the points in 
    gridDims = (x+TPBX-1)//TPBX, (y+TPBY-1)//TPBY, (z+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    genRandPointsKernel[gridDims, blockDims](d_u, d_r, d_v, threshold)
    print(str(int((x*y*z-sum_reduce(cuda.to_device(d_v.copy_to_host().flatten())))+0.5))+" Points") #Prints how many random points were generated.
    return d_v.copy_to_host()

def explode(u):
    #u = points, negative = internal
    m,n,p = u.shape
    x = u.min(0)
    y = u.min(1)
    z = u.min(2)
    x_stretch = np.ones(u.shape)
    y_stretch = np.ones(u.shape)
    z_stretch = np.ones(u.shape)
    for i in range(m):
        x_stretch[i,:,:] = x
    for j in range(n):
        y_stretch[:,j,:] = y
    for k in range(p):
        z_stretch[:,:,k] = z
    return union(union(x_stretch,y_stretch),z_stretch)-np.ones(u.shape)/2