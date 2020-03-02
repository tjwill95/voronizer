import matplotlib.pyplot as plt
import os #Just used to set up file directory
import numpy as np
from numba import cuda
from PIL import Image
import userInput as u
try: TPB = u.TPB 
except: TPB = 8

def slicePlot(u,sliceLocation,titlestring='Plot',save=False,axis = "x"):
    #Plots a slice of matrix u cut at sliceLocation, with the negative values (voxels inside the object) set to teal.
    fig, ax = plt.subplots()
    if axis.upper()=="X":
        plt.contourf(u[sliceLocation,:,:], levels = [-1000,0])
    elif axis.upper()=="Y":
        plt.contourf(u[:,sliceLocation,:], levels = [-1000,0])
    elif axis.upper()=="Z":
        plt.contourf(u[:,:,sliceLocation], levels = [-1000,0])
    plt.title(titlestring)
    ax.set_aspect(1.0)
    plt.show()
    if save:
        fig.savefig(os.path.join(os.path.dirname(__file__),'Output',titlestring+'.png'))
        
def contourPlot(u,sliceLocation,titlestring='Plot',save=False,axis = "x"):
    #Plots a slice of matrix u cut at sliceLocation
    fig, ax = plt.subplots()
    if axis.upper()=="X":
        plt.contourf(u[sliceLocation,:,:])
    elif axis.upper()=="Y":
        plt.contourf(u[:,sliceLocation,:])
    elif axis.upper()=="Z":
        plt.contourf(u[:,:,sliceLocation])
    plt.title(titlestring)
    plt.colorbar()
    ax.set_aspect(1.0)
    plt.show()
    if save:
        fig.savefig(os.path.join(os.path.dirname(__file__),'Output',titlestring+'.png'))

def generateImageStack(model,modelColor,support,supportColor,sliceLocations=[],background=[0,0,0],name="Model"):
    #model = 3D voxel representation of model
    #modelColor = [R,G,B] color for model
    #support = 3D voxel representation of support
    #supportColor = [R,G,B] color for support
    #sliceLocations = x indices for slice to be taken from, vector, defaults to all
    #background = [R,G,B] color for non-solid voxels, defaults to black
    #name = name of the model, defaults to Model
    x,y,z = support.shape
    print("Generating image stack...")
    try: os.mkdir(os.path.join(os.path.dirname(__file__),'Output',name+" image stack"))
    except: pass
    imageModel = setColor(model,modelColor,background)
    imageSupport = setColor(support,supportColor,background)
    completePicture = imageSupport+imageModel
    if not sliceLocations:
        sliceLocations = range(x)
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
    #u = 3D voxel representation of model
    #color = [R,G,B] value desired for that model
    #background = [R,G,B] value desired for voxels outside of the model
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