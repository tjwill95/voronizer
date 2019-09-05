import os #Just used to set up file directory
import numpy as np
import Frep as f
from voronize import voronize
from SDF3D import SDF3D, xHeight
from pointGen import xRandPoints, explode, weightQuantityPoints
from meshExport import generateMesh
from analysis import findVol
from visualizeSlice import slicePlot, contourPlot
from voxelize import voxelize
from makeImage import generateImageStack

FILE_NAME = "" #Don't change, overwritten later as needed
MAT_DENSITY = 1.25 #g/cm^3 (material density)
MODEL = True #Generates model with infill
SUPPORT = True #Generates support structure
INVERSE = False #Also includes the inverse of the model
SEPARATE_SUPPORTS = True #Spits out two files, one for the support and one for the object
DISSOLVABLE = True #Perforates the support structure to allow fluids into the support cells
AESTHETIC = False #Removes all internal detail
RESOLUTION = 200
BUFFER = 4
MODEL_POINTS = 50
MODEL_SHELL = 4
MODEL_CELL = 1
SUPPORT_THRESH = 0.4
SUPPORT_CELL = 1
#FILE_NAME = "E.stl"
FILE_NAME = "3DBenchy_up.stl"
#FILE_NAME = "3DBenchy.stl"
#FILE_NAME = "hand_low.stl"
#FILE_NAME = "couch.stl"
#FILE_NAME = "bust_low.stl"
#FILE_NAME = "wavySurface.stl"
#FILE_NAME = "Bird.stl"

def main():
    modelImport = False
    scale = [1,1,1]
    if not MODEL and not SUPPORT:
        print("You need at least the model or the support structure.")
        return
    #This section allows the user to import STLs
    if FILE_NAME is not "":
        modelImport = True
        filepath = os.path.join(os.path.dirname(__file__), 'Input',FILE_NAME)
        res = RESOLUTION-BUFFER
        origShape, objectBox = voxelize(filepath, res)
        origShape = SDF3D(origShape)
        print(origShape.shape)
        gridResX, gridResY, gridResZ = origShape.shape
        scale[0] = objectBox[0]/(gridResX-BUFFER)
        scale[1] = max(objectBox[1:])/(gridResY-BUFFER)
        scale[2] = scale[1]
        #generateMesh(origShape,scale,modelName="Benchy_Voxel")
    """
    x0 = np.linspace(0.5,3.5,RESOLUTION)
    y0 = np.linspace(0.5,3.5,RESOLUTION)
    z0 = np.linspace(0.5,3.5,RESOLUTION)
    origShape = SDF3D(f.heart(x0,y0,z0,2,2,2))
    generateMesh(origShape,scale,filename="lowRes Heart") #For Paper
    return
    """
    """
    x0 = np.linspace(-50,50,RESOLUTION)
    y0, z0 = x0, x0
    origShape = SDF3D(f.union(f.sphere(x0,y0,z0,40),f.cylinderY(x0,y0,z0,-40,0,40))) #Silo
    #
    #origShape = SDF3D(f.subtract(f.union(f.translate(f.rect(x0,y0,z0,10,10,40),0,int(10/dx),0),/2
    #                  f.translate(f.rect(x0,y0,z0,10,10,40),0,int(-10/dx),0)),f.cylinderY(x0,y0,z0,20,-20,15))) #Cut Cylinder
    #
    #origShape = SDF3D(f.union(f.translate(f.rect(x0,y0,z0,20,20,10),0,-int(10/dx),int(15/dx)+1),
    #                          f.subtract(f.translate(f.rect(x0,y0,z0,40,20,40),0,-int(10/dx),0),
    #                          f.subtract(f.cylinderX(x0,y0,z0,-10,10,10),f.cylinderX(x0,y0,z0,-10,10,20))))) #J
    """
    if SUPPORT:
        projected = f.projection(origShape)
        #generateMesh(projected,scale,modelName="E_Projected") #For Paper
        support = f.subtract(f.thicken(origShape,1),projected)
        support = f.intersection(support, f.translate(support,-1,0,0))
        #generateMesh(support,scale,modelName="E_Support") #For Paper
        contourPlot(support,30,titlestring='Support',axis ="Z")
        supportPts = xRandPoints(xHeight(support),SUPPORT_THRESH)
        supportVoronoi = voronize(support, supportPts, SUPPORT_CELL, 0, scale,
                                  name = "E Support", sliceLocation=30, sliceAxis = "Z")
        #                          name = "Support")
        #generateMesh(supportVoronoi,scale,modelName="E_Support_Voronoi") #For Paper
        if DISSOLVABLE: 
            explosion = f.union(explode(supportPts), f.translate(explode(supportPts),-1,0,0))
            explosion = f.union(explosion,f.translate(explosion,0,1,0))
            explosion = f.union(explosion,f.translate(explosion,0,0,1))
            supportVoronoi = f.subtract(explosion,supportVoronoi)
            #generateMesh(supportVoronoi,scale,modelName="E_Support_Voronoi_Perforated") #For Paper
        table = f.subtract(f.thicken(origShape,1),f.intersection(f.translate(f.subtract(origShape,f.translate(origShape,-3,0,0)),-1,0,0),projected))
        #generateMesh(table,scale,modelName="E_Table") #For Paper
        supportVoronoi = f.union(table,supportVoronoi)
        #generateMesh(supportVoronoi,scale,modelName="E_Support_Complete") #For Paper
        findVol(supportVoronoi,scale,MAT_DENSITY,"Support")
    
    if MODEL:
        objectPts = weightQuantityPoints(origShape,MODEL_POINTS,25)
        print("Points Generated!")
        objectVoronoi = voronize(origShape, objectPts,MODEL_CELL,MODEL_SHELL, scale,
                                 name = "E Model", sliceLocation=30, sliceAxis = "Z")
        #                         name = "Heart", sliceAxis = "Y")
        #                         name = 
        findVol(objectVoronoi,scale,MAT_DENSITY,"E Model") #in mm^3
        if AESTHETIC:
            objectVoronoi = f.union(objectVoronoi,f.thicken(origShape,-8))
        else:
            objectVoronoi = objectVoronoi
    
    if SUPPORT and MODEL:
        complete = f.union(objectVoronoi,supportVoronoi)
    elif SUPPORT:
        complete = supportVoronoi
    elif MODEL:
        complete = objectVoronoi
    slicePlot(complete, origShape.shape[0]//2, titlestring='Full Model', axis = "X")
    slicePlot(complete, origShape.shape[1]//2, titlestring='Full Model', axis = "Y")
    slicePlot(complete, origShape.shape[2]//2, titlestring='Full Model', axis = "Z")
    generateImageStack(objectVoronoi,[255,0,0],supportVoronoi,[0,0,255],[50,100,150,200,250],name = "E")
    UIP = input("Would you like the .ply for this iteration? [Y/N]")
    #UIP = "N" #For Testing Purposes
    if UIP == "Y" or UIP == "y":
        if modelImport:
            fn = FILE_NAME[:-4]+"_Voronoi"
        else:
            fn = input("What would you like the file to be called?")
        print("Generating Model...")
        if SEPARATE_SUPPORTS and SUPPORT and MODEL:
            generateMesh(objectVoronoi,scale,modelName=fn)
            print("Generating Supports...")
            generateMesh(supportVoronoi,scale,modelName=fn+"Support")
        else:
            generateMesh(complete,scale,modelName=fn)
        if INVERSE and MODEL:
            print("Generating Inverse...")
            inv=f.subtract(objectVoronoi,origShape)
            print("Generating Mesh...")
            generateMesh(inv,scale,modelName=fn+"Inv")

if __name__ == '__main__':
    main()