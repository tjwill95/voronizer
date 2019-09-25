import os #Just used to set up file directory
import time
import Frep as f
from voronize import voronize
from SDF3D import SDF3D, xHeight
from pointGen import genRandPoints, explode
from meshExport import generateMesh
from analysis import findVol
from visualizeSlice import slicePlot, contourPlot, generateImageStack
from voxelize import voxelize

FILE_NAME = "" #Don't change, overwritten later as needed
MAT_DENSITY = 1.25 #g/cm^3 (material density), for information only
MODEL = True #Generates model with infill
SUPPORT = True #Generates support structure
SEPARATE_SUPPORTS = True #Spits out two files, one for the support and one for the object
PERFORATE = True #Perforates the support structure to allow fluids into the support cells
IMG_STACK = False #Outputs an image stack of the model
AESTHETIC = False #Removes all internal detail, works best wtih INVERSE
INVERSE = False #Also includes the inverse of the model
RESOLUTION = 175
BUFFER = 4
MODEL_THRESH = 0.1
MODEL_SHELL = 3
MODEL_CELL = .7
SUPPORT_THRESH = 0.2
SUPPORT_CELL = .7
FILE_NAME = "E.stl"
#FILE_NAME = "3DBenchy_up.stl"
#FILE_NAME = "3DBenchy.stl"
#FILE_NAME = "hand_low.stl"
#FILE_NAME = "couch.stl"
#FILE_NAME = "bust_low.stl"
#FILE_NAME = "wavySurface.stl"
#FILE_NAME = "Bird.stl"

def main():
    start = time.time()
    try: os.mkdir(os.path.join(os.path.dirname(__file__),'Output')) #Creates an output folder if there isn't one yet
    except: pass
    modelImport = False
    scale = [1,1,1]
    if not MODEL and not SUPPORT:
        print("You need at least the model or the support structure.")
        return
    #This section allows the user to import STLs
    shortName = "Model"
    if FILE_NAME != "":
        shortName = FILE_NAME[:-4]
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
        support = f.subtract(f.thicken(origShape,1),projected)
        support = f.intersection(support, f.translate(support,-1,0,0))
        contourPlot(support,30,titlestring='Support',axis ="Z")
        supportPts = genRandPoints(xHeight(support),SUPPORT_THRESH)
        supportVoronoi = voronize(support, supportPts, SUPPORT_CELL, 0, scale,
                                  name = "Support",sliceAxis = "Z")
        if PERFORATE: 
            explosion = f.union(explode(supportPts), f.translate(explode(supportPts),-1,0,0))
            explosion = f.union(explosion,f.translate(explosion,0,1,0))
            explosion = f.union(explosion,f.translate(explosion,0,0,1))
            supportVoronoi = f.subtract(explosion,supportVoronoi)
        table = f.subtract(f.thicken(origShape,1),f.intersection(f.translate(f.subtract(origShape,f.translate(origShape,-3,0,0)),-1,0,0),projected))
        supportVoronoi = f.union(table,supportVoronoi)
        findVol(supportVoronoi,scale,MAT_DENSITY,"Support")
    
    if MODEL:
        objectPts = genRandPoints(SDF3D(origShape),MODEL_THRESH)
        print("Points Generated!")
        objectVoronoi = voronize(origShape, objectPts,MODEL_CELL,MODEL_SHELL, scale,
                                 name = "Object")
        findVol(objectVoronoi,scale,MAT_DENSITY,"E Model") #in mm^3
        if AESTHETIC:
            objectVoronoi = f.union(objectVoronoi,f.thicken(origShape,-8))
    shortName = shortName+"_Voronoi"
    if SUPPORT and MODEL:
        complete = f.union(objectVoronoi,supportVoronoi)
        if IMG_STACK:
            generateImageStack(objectVoronoi,[255,0,0],supportVoronoi,[0,0,255],name = shortName)
    elif SUPPORT:
        complete = supportVoronoi
        if IMG_STACK:
            generateImageStack(supportVoronoi,[0,0,0],supportVoronoi,[0,0,255],name = shortName)
    elif MODEL:
        complete = objectVoronoi
        if IMG_STACK:
            generateImageStack(objectVoronoi,[255,0,0],objectVoronoi,[0,0,0],name = FILE_NAME[:-4])
    slicePlot(complete, origShape.shape[0]//2, titlestring='Full Model', axis = "X")
    slicePlot(complete, origShape.shape[1]//2, titlestring='Full Model', axis = "Y")
    slicePlot(complete, origShape.shape[2]//2, titlestring='Full Model', axis = "Z")
    
    print("That took "+str(round(time.time()-start,2))+" seconds.")
    UIP = input("Would you like the .ply for this iteration? [Y/N]")
    #UIP = "N" #For Testing Purposes
    if UIP == "Y" or UIP == "y":
        if modelImport:
            fn = shortName
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