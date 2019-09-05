from numba import cuda
import numpy as np
import math
TPB = 8

@cuda.jit
def boolKernel(d_u,d_v):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i >= dims[0] or j >= dims[1] or k >= dims[2]:
        return
    d_u[i,j,k] = min(d_u[i,j,k],d_v[i,j,k])
    
def union(u,v):
    #u,v = voxel models that you want to union
    #Outputs the union of the models (fills in the resulting matrix such that 
    #if a cell is negative in either u or v, it is negative in the output).
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    boolKernel[gridSize, blockSize](d_u,d_v)
    return d_u.copy_to_host()

def intersection(u,v):
    #u,v = voxel models that you want to intersect
    #Outputs the intersection of the models (fills in the resulting matrix such
    #that if a cell is positive in either u or v, it is positive in the output).
    d_u = cuda.to_device(-1*u)
    d_v = cuda.to_device(-1*v)
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    boolKernel[gridSize, blockSize](d_u,d_v)
    return -1*d_u.copy_to_host()

def subtract(u,v):
    #u = cutting tool model (Model that's removed)
    #v = base model
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(-1*v)
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    boolKernel[gridSize, blockSize](d_u,d_v)
    return -1*d_u.copy_to_host()

@cuda.jit
def projectionKernel(d_u,X):
    j,k = cuda.grid(2)
    m,n,p = d_u.shape
    if j < n and k < p:
        if d_u[X+1,j,k]<=0:
            d_u[X,j,k]=-1    

def projection(u):
    #u = voxelized model, negative = internalago
    #Assumes X is the vertical axis, projects entire part to lowest Z value.
    TPBY, TPBZ = TPB, TPB
    m, n, p = u.shape
    minX = -1
    i = 0
    while minX<0:
        if np.amin(u[i,:,:])<0:
            minX = i
        else:
            i += 1
    X = m-1
    d_u = cuda.to_device(u)
    gridDims = (n+TPBY-1)//TPBY, (p+TPBZ-1)//TPBZ
    blockDims = TPBY, TPBZ
    while X>minX:
        X -= 1
        projectionKernel[gridDims, blockDims](d_u,X)
    return d_u.copy_to_host()

@cuda.jit
def translateKernel(d_u,d_v,x,y,z):
    i,j,k = cuda.grid(3)
    m,n,p = d_u.shape
    if i >= m or j >= n or k >= p:
        return
    d_v[i,j,k] = d_u[(i-x)%m,(j-y)%n,(k-z)%p]
    
def translate(u,x,y,z):
    #u = voxel model to translate
    #x,y,z = translation vector, integers
    #moves the model according to the translation vector.
    d_u = cuda.to_device(u)
    d_v = cuda.device_array(shape = u.shape, dtype = np.float32)
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    translateKernel[gridSize, blockSize](d_u,d_v,x,y,z)
    return d_v.copy_to_host()

def thicken(u,weight):
    #u = voxel model to thicken, assumes SDF
    #origShape = outer bounds of model
    #weight = how much we're thickening the object
    return u - np.ones(u.shape)*weight
    
def shell(uSDF,sT):
    #u = voxel model to shell, assumes SDF
    #sT = thickness of the shell
    return intersection(uSDF,-uSDF-np.ones(uSDF.shape)*sT)

@cuda.jit
def heartKernel(d_u, d_x, d_y, d_z,cx,cy,cz):
    i,j, k = cuda.grid(3)
    m,n,p = d_u.shape
    if i < m and j < n and k < p:
        x = d_x[i]-cx
        y = d_y[j]-cy
        z = d_z[k]-cz
        d_u[i,j,k] = (x**2+9*(y**2)/4+z**2-1)**3-(x**2)*(z**3)-9*(y**2)*(z**3)/80
        
def heart(x,y,z,cx,cy,cz):
    #x,y,z = coordinate domain that we want the shape to live in.
    #Outputs a 3D matrix with negative values showing the inside of our shape, 
    #positive values showing the outside, and 0s to show the surfaces.
    #cx,cy,cz specify the center of the heart shape.
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    m = x.shape[0]
    n = y.shape[0]
    p = z.shape[0]
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_z = cuda.to_device(z)
    d_u = cuda.device_array(shape = [m, n, p], dtype = np.float32)
    gridDims = (m+TPBX-1)//TPBX, (n+TPBY-1)//TPBY, (n+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    heartKernel[gridDims, blockDims](d_u, d_x, d_y, d_z,cx,cy,cz)
    return d_u.copy_to_host()

@cuda.jit
def rectKernel(d_u, d_x0, d_y0, d_z0, xl, yl, zl, origin):
    i,j, k = cuda.grid(3)
    m,n,p = d_u.shape
    if i < m and j < n and k < p:
        sx = abs(d_x0[i]-origin[0]) - xl/2
        sy = abs(d_y0[j]-origin[1]) - yl/2
        sz = abs(d_z0[k]-origin[2]) - zl/2
        d_u[i,j,k]=max(sx,sy,sz)
        
def rect(x0,y0,z0,xl,yl,zl,origin = [0,0,0]):
    #x0,y0,z0 = x,y,z coordinate domain that we want the shape to live in.
    #Outputs a 3D matrix with negative values showing the inside of our shape, 
    #positive values showing the outside, and 0s to show the surfaces.
    #xl,yl,zl are the sidelengths of the rectangular prism.
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    m = x0.shape[0]
    n = y0.shape[0]
    p = z0.shape[0]
    d_x0 = cuda.to_device(x0)
    d_y0 = cuda.to_device(y0)
    d_z0 = cuda.to_device(z0)
    d_origin = cuda.to_device(origin)
    d_u = cuda.device_array(shape = [m, n, p], dtype = np.float32)
    gridDims = (m+TPBX-1)//TPBX, (n+TPBY-1)//TPBY, (n+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    rectKernel[gridDims, blockDims](d_u, d_x0, d_y0, d_z0, xl, yl, zl, d_origin)
    return d_u.copy_to_host()

@cuda.jit
def sphereKernel(d_u, d_x0, d_y0, d_z0, rad):
    i,j, k = cuda.grid(3)
    m,n,p = d_u.shape
    if i < m and j < n and k < p:
        d_u[i,j, k] = math.sqrt(d_x0[i]**2+d_y0[j]**2+d_z0[k]**2)-rad
        
def sphere(x0,y0,z0,rad):
    #x0,y0,z0 = x,y,z coordinate domain that we want the shape to live in.
    #Outputs a 3D matrix with negative values showing the inside of our shape, 
    #positive values showing the outside, and 0s to show the surfaces.
    #rad gives the radius of the sphere.
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    m = x0.shape[0]
    n = y0.shape[0]
    p = z0.shape[0]
    d_x0 = cuda.to_device(x0)
    d_y0 = cuda.to_device(y0)
    d_z0 = cuda.to_device(z0)
    d_u = cuda.device_array(shape = [m, n, p], dtype = np.float32)
    gridDims = (m+TPBX-1)//TPBX, (n+TPBY-1)//TPBY, (n+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    sphereKernel[gridDims, blockDims](d_u, d_x0, d_y0, d_z0, rad)
    return d_u.copy_to_host()

@cuda.jit
def cylinderYKernel(d_u, d_x0, d_y0, d_z0, start, stop, rad):
    i,j,k = cuda.grid(3)
    m,n,p = d_u.shape
    if i < m and j < n and k < p:
        height = (d_y0[j]-start)*(d_y0[j]-stop)
        width = math.sqrt(d_x0[i]**2+d_z0[k]**2)-rad
        d_u[i,j,k] = max(height,width)

def cylinderY(x0,y0,z0,start,stop,rad):
    #x0,y0,z0 are the x,y,z coordinate domain that we want the shape to live in.
    #Outputs a 3D matrix with negative values showing the inside of our shape, 
    #positive values showing the outside, and 0s to show the surfaces.
    #start and stop give the start and end Y coordinates.
    #rad gives the radius of the cylinder.
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    m = x0.shape[0]
    n = y0.shape[0]
    p = z0.shape[0]
    d_x0 = cuda.to_device(x0)
    d_y0 = cuda.to_device(y0)
    d_z0 = cuda.to_device(z0)
    d_u = cuda.device_array(shape = [m, n, p], dtype = np.float32)
    gridDims = (m+TPBX-1)//TPBX, (n+TPBY-1)//TPBY, (n+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    cylinderYKernel[gridDims, blockDims](d_u, d_x0, d_y0, d_z0, start, stop, rad)
    return d_u.copy_to_host()

@cuda.jit
def cylinderXKernel(d_u, d_x0, d_y0, d_z0, start, stop, rad):
    i,j,k = cuda.grid(3)
    m,n,p = d_u.shape
    if i < m and j < n and k < p:
        height = (d_x0[i]-start)*(d_x0[i]-stop)
        width = math.sqrt(d_y0[j]**2+d_z0[k]**2)-rad
        d_u[i,j,k] = max(height,width)

def cylinderX(x0,y0,z0,start,stop,rad):
    #x0,y0,z0 are the x,y,z coordinate domain that we want the shape to live in.
    #Outputs a 3D matrix with negative values showing the inside of our shape, 
    #positive values showing the outside, and 0s to show the surfaces.
    #start and stop give the start and end x coordinates.
    #rad gives the radius of the cylinder.
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    m = x0.shape[0]
    n = y0.shape[0]
    p = z0.shape[0]
    d_x0 = cuda.to_device(x0)
    d_y0 = cuda.to_device(y0)
    d_z0 = cuda.to_device(z0)
    d_u = cuda.device_array(shape = [m, n, p], dtype = np.float32)
    gridDims = (m+TPBX-1)//TPBX, (n+TPBY-1)//TPBY, (n+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    cylinderXKernel[gridDims, blockDims](d_u, d_x0, d_y0, d_z0, start, stop, rad)
    return d_u.copy_to_host()