from numba import cuda
import numpy as np

TPB = 8

@cuda.jit(device = True)
def distance(i,j,k,m,n,p,L):
    return (abs((i-m)**L)+abs((j-n)**L)+abs((k-p)**L))**(1/L)

@cuda.jit
def JFKernel(d_pr,d_pw,stepSize,L):
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
            d1 = distance(i,j,k,m1,n1,p1,L)
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
        return

def jumpFlood(u,order=2.0):
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    d_r = cuda.to_device(1000*np.ones([dims[0],dims[1],dims[2],4],np.float32))
    d_w = cuda.to_device(1000*np.ones([dims[0],dims[1],dims[2],4],np.float32))
    d_u = cuda.to_device(u)
    JFSetupKernel[gridSize, blockSize](d_u,d_r)
    n = int(round(np.log2(max(dims)-1)+0.5))
    for count in range(n):
        stepSize = 2**(n-count-1)
        JFKernel[gridSize, blockSize](d_r,d_w,stepSize,order)
        d_r,d_w = d_w,d_r
    for count in range(2):
        stepSize = 2-count
        JFKernel[gridSize, blockSize](d_r,d_w,stepSize,order)
        d_r,d_w = d_w,d_r
    return d_r

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

def SDF3D(u):
    #u = a voxel model where the negative values indicate that the voxel is 
    #inside the object, positive is outside, and 0 is on the surface.
    #Outputs a new voxel model where the same sign rules apply, but the value 
    #of the cell indicates how far away that cell is from the nearest surface.
    #Formatted u[i,j,k]=(Nearest Seed i, NSj, NSk, Distance to Seed)
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    d_p = cuda.to_device(jumpFlood(u))
    d_n = cuda.to_device(jumpFlood(-u))
    d_u = cuda.to_device(u)
    toSDF[gridSize, blockSize](d_p,d_n,d_u)
    return d_u.copy_to_host()

@cuda.jit(device = True)
def CFD(n2,n1,p1,p2,h):
    #return (n2/12-2*n1/3+2*p1/3-p2/12)/h
    return (p1-n1)/(2*h)

@cuda.jit
def grad3DKernel(d_u,d_v,maxVal):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    h1 = 1 #step size
    h2 = 2**(1/2)
    h3 = 3**(1/2)
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    dx = CFD(d_u[i-2,j,k],d_u[i-1,j,k],d_u[i+1,j,k],d_u[i+2,j,k],h1)**2
    dy = CFD(d_u[i,j-2,k],d_u[i,j-1,k],d_u[i,j+1,k],d_u[i,j+2,k],h1)**2
    dz = CFD(d_u[i,j,k-2],d_u[i,j,k-1],d_u[i,j,k+1],d_u[i,j,k+2],h1)**2
    dPxPy = CFD(d_u[i-2,j-2,k],d_u[i-1,j-1,k],d_u[i+1,j+1,k],d_u[i+2,j+2,k],h2)**2
    dNxPy = CFD(d_u[i-2,j+2,k],d_u[i-1,j+1,k],d_u[i+1,j-1,k],d_u[i+2,j-2,k],h2)**2
    dPyPz = CFD(d_u[i,j-2,k-2],d_u[i,j-1,k-1],d_u[i,j+1,k+1],d_u[i,j+2,k+2],h2)**2
    dNyPz = CFD(d_u[i,j-2,k+2],d_u[i,j-1,k+1],d_u[i,j+1,k-1],d_u[i,j+2,k-2],h2)**2
    dPxPz = CFD(d_u[i-2,j,k-2],d_u[i-1,j,k-1],d_u[i+1,j,k+1],d_u[i+2,j,k+2],h2)**2
    dNxPz = CFD(d_u[i-2,j,k+2],d_u[i-1,j,k+1],d_u[i+1,j,k-1],d_u[i+2,j,k-2],h2)**2
    dPPP = CFD(d_u[i-2,j-2,k-2],d_u[i-1,j-1,k-1],d_u[i+1,j+1,k+1],d_u[i+2,j+2,k+2],h3)**2
    dNPP = CFD(d_u[i+2,j-2,k-2],d_u[i+1,j-1,k-1],d_u[i-1,j+1,k+1],d_u[i-2,j+2,k+2],h3)**2
    dPNP = CFD(d_u[i-2,j+2,k-2],d_u[i-1,j+1,k-1],d_u[i+1,j-1,k+1],d_u[i+2,j-2,k+2],h3)**2
    dPPN = CFD(d_u[i-2,j-2,k+2],d_u[i-1,j-1,k+1],d_u[i+1,j+1,k-1],d_u[i+2,j+2,k-2],h3)**2
    d2,d3,d4,d5,d6,d7,d8 = 1,1,1,1,1,1,1
    d1 = min(maxVal,(dx+dy+dz))
    d2 = min(maxVal,(dPxPy+dNxPy+dz))
    d3 = min(maxVal,(dx+dPyPz+dNyPz))
    d4 = min(maxVal,(dPxPz+dy+dNxPz))
    d5 = min(maxVal,(dPPP+dPPN+dNxPy))
    d6 = min(maxVal,(dPPP+dNPP+dNyPz))
    d7 = min(maxVal,(dNPP+dPNP+dPxPy))
    d8 = min(maxVal,(dPNP+dPPN+dPyPz))
    grad = min(d1,d2,d3,d4,d5,d6,d7,d8)**(1/2) #Lowest gradient
    if maxVal>0:
        d_v[i,j,k]=min(grad,maxVal)
    else:
        d_v[i,j,k]=grad
    
def grad3D(u,maxVal=0):
    #u = SDF of our desired voxel model
    #maxVal = Upper cap on the gradient.  If any gradient values fall above 
    #this value, it is set to the maxVal.  This is for visualization purposes.
    #Outputs the magnitude of the gradient vector at each cell in the voxel model.
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(np.ones(np.shape(u)))
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    grad3DKernel[gridSize, blockSize](d_u,d_v,maxVal)
    return d_v.copy_to_host()

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