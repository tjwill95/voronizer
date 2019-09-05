import numpy as np
from numba import cuda
from PIL import Image
import os #Just used to set up file directory

TPB = 8

def generateImageStack(model,modelColor,support,supportColor,sliceLocations,background=[0,0,0],name = ""):
    #support = 3D voxel representation of support
    #supportColor = [R,G,B] color for support
    #model = 3D voxel representation of model
    #modelColor = [R,G,B] color for model
    #sliceLocations = x indices for slice to be taken from, vector
    #background = color for non-solid voxels
    x,y,z = support.shape
    print("Generating image stack...")
    os.mkdir(os.path.join(os.path.dirname(__file__),'Output',name+" image stack"))
    imageModel = setColor(model,modelColor,background)
    imageSupport = setColor(support,supportColor,background)
    completePicture = imageSupport+imageModel
    for val in sliceLocations:
        picture = completePicture[val,:,:,:]
        img = Image.fromarray(picture, 'RGB')
        img.save(os.path.join(os.path.dirname(__file__),'Output',name+" image stack",str(val)+name+'.png'))
    print("Image Stack Complete!")
    
@cuda.jit
def setColorKernel(d_u,d_v,color,background):
    i,j, k = cuda.grid(3)
    m,n,p = d_u.shape
    if i < m and j < n and k < p:
        if d_u[i,j,k] < 0:
            d_v[i,j,k] = color
        else:
            d_v[i,j,k] = background
    
def setColor(u,color,background):
    x,y,z = u.shape
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(np.ones([x,y,z],dtype=np.uint8))
    image = np.ones([x,y,z,3],dtype=np.uint8)
    gridDims = (x+TPBX-1)//TPBX, (y+TPBY-1)//TPBY, (z+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    for i in range(3):
        setColorKernel[gridDims,blockDims](d_u,d_v,color[i],background[i])
        image[:,:,:,i]= d_v.copy_to_host()
    return image