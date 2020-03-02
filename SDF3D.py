from numba import cuda
import math
import numpy as np
import userInput as u
try: TPB = u.TPB 
except: TPB = 8

@cuda.jit(device = True)
def norm(i,j,k,m,n,p,order):
    return (abs((i-m)**order)+abs((j-n)**order)+abs((k-p)**order))**(1/order)

@cuda.jit(device = True)
def distance(i,j,k,m,n,p):
    return math.sqrt((i-m)*(i-m)+(j-n)*(j-n)+(k-p)*(k-p))

@cuda.jit
def JFKernel(d_pr,d_pw,stepSize):
    i,j,k = cuda.grid(3)
    dims = d_pr.shape
    m,n,p,d = d_pr[i,j,k]
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    for index in range(27):
        checkPos = (i+((index//9)%3-1)*stepSize,
                    j+((index//3)%3-1)*stepSize,
                    k+(index%3-1)*stepSize)
        if checkPos[0]<dims[0] and checkPos[1]<dims[1] and checkPos[2]<dims[2] and min(checkPos)>0:
            m1,n1,p1,d1 = d_pr[checkPos]
            d1 = distance(i,j,k,m1,n1,p1)
            if d1<d:
                m,n,p,d = m1,n1,p1,d1
    d_pw[i,j,k,:] = m,n,p,np.float32(d)
    
    
@cuda.jit
def JFKernelNorm(d_pr,d_pw,stepSize,order):
    i,j,k = cuda.grid(3)
    dims = d_pr.shape
    m,n,p,d = d_pr[i,j,k]
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    for index in range(27):
        checkPos = (i+((index//9)%3-1)*stepSize,
                    j+((index//3)%3-1)*stepSize,
                    k+(index%3-1)*stepSize)
        if checkPos[0]<dims[0] and checkPos[1]<dims[1] and checkPos[2]<dims[2] and min(checkPos)>0:
            m1,n1,p1,d1 = d_pr[checkPos]
            d1 = norm(i,j,k,m1,n1,p1,order)
            if d1<d:
                m,n,p,d = m1,n1,p1,d1
    d_pw[i,j,k,:] = m,n,p,np.float32(d)
    
@cuda.jit
def JFSetupKernel(d_u,d_p):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    if d_u[i,j,k]<=0.0:
        d_p[i,j,k,:]=float(i),float(j),float(k),0.0

def jumpFlood(u,norm):
    #u = a voxel model where the negative values indicate that the voxel is 
    #inside the object, positive is outside, and 0 is on the surface.
    #Output formatted as follows: 
    #u[i,j,k,d]=(i coord of Nearest Seed (NS), j coord of NS, k coord of NS, distance to NS)
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    d_r = cuda.to_device(1000*np.ones([dims[0],dims[1],dims[2],4],np.float32))
    d_w = cuda.to_device(1000*np.ones([dims[0],dims[1],dims[2],4],np.float32))
    d_u = cuda.to_device(u)
    JFSetupKernel[gridSize, blockSize](d_u,d_r)
    n = int(round(np.log2(max(dims)-1)+0.5))
    if norm==2.0:
        for count in range(n):
            stepSize = 2**(n-count-1)
            JFKernel[gridSize, blockSize](d_r,d_w,stepSize)
            d_r,d_w = d_w,d_r
        for count in range(2):
            stepSize = 2-count
            JFKernel[gridSize, blockSize](d_r,d_w,stepSize)
            d_r,d_w = d_w,d_r
    else:
        for count in range(n):
            stepSize = 2**(n-count-1)
            JFKernelNorm[gridSize, blockSize](d_r,d_w,stepSize,norm)
            d_r,d_w = d_w,d_r
        for count in range(2):
            stepSize = 2-count
            JFKernelNorm[gridSize, blockSize](d_r,d_w,stepSize,norm)
            d_r,d_w = d_w,d_r
    return d_r.copy_to_host()

@cuda.jit
def toSDF(JFpos,JFneg,d_u):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    dp = JFpos[i,j,k,3]
    dn = JFneg[i,j,k,3]
    if dp>0:
        d_u[i,j,k]=dp
    else:
        d_u[i,j,k]=-dn

def SDF3D(u,norm=2.0):
    #u = a voxel model where the negative values indicate that the voxel is 
    #inside the object, positive is outside, and 0 is on the surface.
    #Outputs a new voxel model where the same sign rules apply, but the value 
    #of the cell indicates how far away that cell is from the nearest surface.
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    d_p = cuda.to_device(jumpFlood(u,norm))
    d_n = cuda.to_device(jumpFlood(-u,norm))
    d_u = cuda.to_device(u)
    toSDF[gridSize, blockSize](d_p,d_n,d_u)
    return d_u.copy_to_host()

@cuda.jit
def simplifyKernel(d_u,d_v):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i >= dims[0] or j >= dims[1] or k >= dims[2]:
        return
    p = 0
    n = 0
    if d_u[i,j,k]<=0:
        for index in range(27):
            checkPos = (i+((index//9)%3-1),j+((index//3)%3-1),k+(index%3-1))
            if index!=13 and d_u[checkPos]>0:
                p=1
            if index!=13 and d_u[checkPos]<0:
                n=1
            if p+n==2:
                d_v[i,j,k]=0
                return
        d_v[i,j,k]=-.01
    else:
        d_v[i,j,k]=.01

def simplify(u):
    d_u = cuda.to_device(u)
    dims = u.shape
    d_v = cuda.device_array(dims,dtype=np.float32)
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    simplifyKernel[gridSize, blockSize](d_u,d_v)
    return d_v.copy_to_host()

@cuda.jit
def xHeightKernel(d_u,i):
    j,k = cuda.grid(2)
    m,n,p = d_u.shape
    if j < n and k < p and d_u[i,j,k]<=0:
            d_u[i,j,k]=min(-1,d_u[i+1,j,k]-1)

def xHeight(u):
    #u = voxelized model, negative = internal
    #Assumes X is the vertical axis, sets each solid voxel value to the above value minus 1.
    m, n, p = u.shape
    TPBY, TPBZ = TPB, TPB
    gridDims = (n+TPBY-1)//TPBY, (n+TPBZ-1)//TPBZ
    blockDims = TPBY, TPBZ
    i = m-2
    d_u = cuda.to_device(simplify(u))
    while i>0:
        xHeightKernel[gridDims, blockDims](d_u,i)
        i -= 1
    return d_u.copy_to_host()