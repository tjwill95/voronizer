from visualizeSlice import slicePlot, contourPlot
import Frep as f
from SDF3D import SDF3D, jumpFlood
from numba import cuda
import numpy as np
import userInput as u
try: TPB = u.TPB 
except: TPB = 8

def voronize(origObject, seedPoints, cellThickness, shellThickness, scale,
             name = "", sliceLocation = 0, sliceAxis = "X", order = 2):
    #origObject = voxel model of original object, negative = inside
    #seedPoints = same-size matrix with 0s at the location of each seed point, 1s elsewhere
    #wallThickness  = desired minimum thickness of cell walls (mm).
    #shellThickness = desired minimum thickness of shell (mm), 0 if no shell
    #name = If given a value, the name of the model, activates progress plots.
    resX, resY, resZ = origObject.shape
    if sliceLocation == 0:
        if sliceAxis == "X" or sliceAxis == "x":
            sliceLocation = resX//2
        elif sliceAxis == "Y" or sliceAxis =="y":
            sliceLocation = resY//2
        else:
            sliceLocation = resZ//2
    seedPoints = jumpFlood(seedPoints,order)
    if name !="":
        contourPlot(seedPoints[:,:,:,3],sliceLocation,titlestring="SDF of the Points for "+name,axis = sliceAxis)
    voronoi = wallFinder(seedPoints)
    voronoi = SDF3D(voronoi)
    if name !="":
        slicePlot(voronoi,sliceLocation,titlestring="Voronoi Structure for "+name,axis = sliceAxis)
    wallThickness=cellThickness/2-1
    voronoi = f.intersection(f.thicken(voronoi,wallThickness),origObject)
    if name !="":
        slicePlot(voronoi, sliceLocation, titlestring=(name+' Trimmed and Thinned'),axis = sliceAxis)
    if shellThickness>0:
        u_shell = f.shell(origObject,shellThickness)
        voronoi = f.union(u_shell,voronoi)
        if name !="":
            slicePlot(voronoi, sliceLocation, titlestring=name+' With Shell',axis = sliceAxis)
    if name =="":
        name = "Model"
    print("Voronize for " + name + " Complete!")
    return voronoi

@cuda.jit
def wallFinderKernel(d_points,d_walls):
    i,j,k = cuda.grid(3)
    dims = d_walls.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    m,n,p,d = d_points[i,j,k]
    for index in range(27):
        checkPos = (i+((index//9)%3-1),j+((index//3)%3-1),k+(index%3-1))
        if checkPos[0]<dims[0] and checkPos[1]<dims[1] and checkPos[2]<dims[2] and min(checkPos)>0:
            m1,n1,p1,d1 = d_points[checkPos]
            if m!=m1 or n!=n1 or p!=p1:
                d_walls[i,j,k]=-1
        
def wallFinder(voxel):
    #voxel = the original voxel model of the object
    #gradient = the gradient field of the object
    #Outputs a voxel model with material where the gradient was below the 
    #threshold.
    dims = voxel.shape
    d_points = cuda.to_device(voxel)
    d_walls = cuda.to_device(np.ones(dims[:3]))
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    wallFinderKernel[gridSize, blockSize](d_points,d_walls)
    return d_walls.copy_to_host()