MAT_DENSITY = 1.25  #g/cm^3 (material density), for information only
MODEL = True        #Generates model with infill
SUPPORT = False      #Generates support structure
SEPARATE_SUPPORTS = True #Spits out two files, one for the support and one for the object
PERFORATE = False    #Perforates the support structure to allow fluids into the support cells
IMG_STACK = False   #Outputs an image stack of the model
AESTHETIC = False   #Removes all internal detail, works best with INVERSE
INVERSE = False     #Also includes the inverse of the model as a separate file
NET = False          #Only draws voronoi patterns at the surface of the objec
SMOOTH = True       #Smooths the output meshes, removes the voxelized texture
NET_THICKNESS = 4   #Sets the thickness of the net in voxels
BUFFER = 4          #Sets the empty voxels around the object
TPB = 8             #Threads per block, leave at 8 unless futzing.

RESOLUTION = 300    #Sets the resolution of the Y and Z axes
MODEL_THRESH = 0.1  #Influences the number of cells in the model, larger values lead to more cells
MODEL_SHELL = 3#3     #Sets the thickness of the model skin (in voxels), set to 0 for aesthetic models
MODEL_CELL = 0.9#0.9     #Sets the thickness of the cell walls within the model (in voxels)

SUPPORT_THRESH = 0.2#Influences the number of cells in the supports, larger values lead to more cells
SUPPORT_CELL = 0.7  #Sets the thickness of the cell walls within the supports (in voxels)

#Put the name of the desired file below, or uncomment one of the example files
#This file must be in the Input folder, set to be in the same directory as the
#python files.

#FILE_NAME = "E.stl"
#FILE_NAME = "3DBenchy_up.stl"
#FILE_NAME = "3DBenchy.stl"
#FILE_NAME = "bust_low.stl"
#FILE_NAME = "wavySurface.stl"

#If you would prefer a simple geometric object, uncomment the one you want and
#make sure that all FILE_NAME options are commented out.

#PRIMITIVE_TYPE = "Heart"
PRIMITIVE_TYPE = "Egg"
#PRIMITIVE_TYPE = "Cube"
#PRIMITIVE_TYPE = "Sphere"
#PRIMITIVE_TYPE = "Cylinder"
#PRIMITIVE_TYPE = "Silo"
