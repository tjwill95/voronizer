import os #Just used to set up file directory
import time
import numpy as np
import Frep as f
import userInput as u
from voronize import voronize
from SDF3D import SDF3D, xHeight
from pointGen import genRandPoints, explode
from meshExport import generateMesh
from analysis import findVol
from visualizeSlice import slicePlot, contourPlot, generateImageStack
from voxelize import voxelize

BUFFER = 2

def main():
    start = time.time()
    try:    FILE_NAME = u.FILE_NAME
    except: FILE_NAME = ""
    try:    os.mkdir(os.path.join(os.path.dirname(__file__),'Output')) #Creates an output folder if there isn't one yet
    except: pass
    modelImport = False
    scale = [1,1,1]
    if not u.MODEL and not u.SUPPORT:
        print("You need at least the model or the support structure.")
        return
    if FILE_NAME != "":
        shortName = FILE_NAME[:-4]
        modelImport = True
        filepath = os.path.join(os.path.dirname(__file__), 'Input',FILE_NAME)
        res = u.RESOLUTION-BUFFER*2
        origShape, objectBox = voxelize(filepath, res, BUFFER)
        gridResX, gridResY, gridResZ = origShape.shape
        scale[0] = objectBox[0]/(gridResX-BUFFER*2)
        scale[1] = max(objectBox[1:])/(gridResY-BUFFER*2)
        scale[2] = scale[1]
    elif u.PRIMITIVE == True:
        shortName = u.PRIMITIVE_TYPE
        if u.PRIMITIVE_TYPE == "Heart":
            x0 = np.linspace(0.5,3.5,u.RESOLUTION)
            y0 = np.linspace(0.5,3.5,u.RESOLUTION)
            z0 = np.linspace(0.5,3.5,u.RESOLUTION)
            origShape = f.heart(x0,y0,z0,2,2,2)
        else:
            x0 = np.linspace(-50,50,u.RESOLUTION)
            y0, z0 = x0, x0
            if u.PRIMITIVE_TYPE == "Cube":
                origShape = f.rect(x0,y0,z0,80,80,80)
            elif u.PRIMITIVE_TYPE == "Silo":
                origShape = f.union(f.sphere(x0,y0,z0,40),f.cylinderY(x0,y0,z0,-40,0,40))
            elif u.PRIMITIVE_TYPE == "Cylinder":
                origShape = f.cylinderX(x0,y0,z0,-40,40,40)
            else: #Sphere
                origShape = f.sphere(x0,y0,z0,40)
    else:
        print("Provide either a file name or a desired primitive.")
        return
    
    print("Initial Bounding Box Dimensions: "+str(origShape.shape))
    origShape = SDF3D(f.condense(origShape,BUFFER))
    print("Condensed Bounding Box Dimensions: "+str(origShape.shape))
    
    if u.SUPPORT:
        projected = f.projection(origShape)
        support = f.subtract(f.thicken(origShape,1),projected)
        support = f.intersection(support, f.translate(support,-1,0,0))
        contourPlot(support,30,titlestring='Support',axis ="Z")
        supportPts = genRandPoints(xHeight(support), u.SUPPORT_THRESH)
        supportVoronoi = voronize(support, supportPts, u.SUPPORT_CELL, 0, scale, name = "Support", sliceAxis = "Z")
        if u.PERFORATE: 
            explosion = f.union(explode(supportPts), f.translate(explode(supportPts),-1,0,0))
            explosion = f.union(explosion,f.translate(explosion,0,1,0))
            explosion = f.union(explosion,f.translate(explosion,0,0,1))
            supportVoronoi = f.subtract(explosion,supportVoronoi)
        table = f.subtract(f.thicken(origShape,1),f.intersection(f.translate(f.subtract(origShape,f.translate(origShape,-3,0,0)),-1,0,0),projected))
        supportVoronoi = f.union(table,supportVoronoi)
        findVol(supportVoronoi,scale,u.MAT_DENSITY,"Support")
    
    if u.MODEL:
        objectPts = genRandPoints(origShape,u.MODEL_THRESH)
        print("Points Generated!")
        objectVoronoi = voronize(origShape, objectPts, u.MODEL_CELL, u.MODEL_SHELL, scale, name = "Object")
        findVol(objectVoronoi,scale,u.MAT_DENSITY,"Object") #in mm^3
        if u.AESTHETIC:
            objectVoronoi = f.union(objectVoronoi,f.thicken(origShape,-8))
    shortName = shortName+"_Voronoi"
    if u.SUPPORT and u.MODEL:
        complete = f.union(objectVoronoi,supportVoronoi)
        if u.IMG_STACK:
            generateImageStack(objectVoronoi,[255,0,0],supportVoronoi,[0,0,255],name = shortName)
    elif u.SUPPORT:
        complete = supportVoronoi
        if u.IMG_STACK:
            generateImageStack(supportVoronoi,[0,0,0],supportVoronoi,[0,0,255],name = shortName)
    elif u.MODEL:
        complete = objectVoronoi
        if u.IMG_STACK:
            generateImageStack(objectVoronoi,[255,0,0],objectVoronoi,[0,0,0],name = FILE_NAME[:-4])
    slicePlot(complete, origShape.shape[0]//2, titlestring='Full Model', axis = "X")
    slicePlot(complete, origShape.shape[1]//2, titlestring='Full Model', axis = "Y")
    slicePlot(complete, origShape.shape[2]//2, titlestring='Full Model', axis = "Z")
    
    print("That took "+str(round(time.time()-start,2))+" seconds.")
    UIP = input("Would you like the .ply for this iteration? [Y/N]")
    if UIP == "Y" or UIP == "y":
        if modelImport:
            fn = shortName
        else:
            fn = input("What would you like the file to be called?")
        print("Generating Model...")
        if u.SEPARATE_SUPPORTS and u.SUPPORT and u.MODEL:
            generateMesh(objectVoronoi,scale,modelName=fn)
            print("Generating Supports...")
            generateMesh(supportVoronoi,scale,modelName=fn+"Support")
        else:
            generateMesh(complete,scale,modelName=fn)
        if u.INVERSE and u.MODEL:
            print("Generating Inverse...")
            inv=f.subtract(objectVoronoi,origShape)
            print("Generating Mesh...")
            generateMesh(inv,scale,modelName=fn+"Inv")

if __name__ == '__main__':
    main()
