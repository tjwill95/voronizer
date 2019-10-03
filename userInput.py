MAT_DENSITY = 1.25 #g/cm^3 (material density), for information only
MODEL = True #Generates model with infill
SUPPORT = True #Generates support structure
SEPARATE_SUPPORTS = True #Spits out two files, one for the support and one for the object
PERFORATE = True #Perforates the support structure to allow fluids into the support cells
IMG_STACK = False #Outputs an image stack of the model
AESTHETIC = False #Removes all internal detail, works best wtih INVERSE
INVERSE = False #Also includes the inverse of the model
PRIMATIVE = False #
RESOLUTION = 150
MODEL_THRESH = 0.1
MODEL_SHELL = 3
MODEL_CELL = .7
SUPPORT_THRESH = 0.2
SUPPORT_CELL = .7
FILE_NAME = "E.stl"
#FILE_NAME = "3DBenchy_up.stl"
#FILE_NAME = "3DBenchy.stl"
#FILE_NAME = "bust_low.stl"
#FILE_NAME = "wavySurface.stl"
PRIMATIVE_TYPE = "Heart"
PRIMATIVE_TYPE = "Cube"
PRIMATIVE_TYPE = "Sphere"
PRIMATIVE_TYPE = "Cylinder"
PRIMATIVE_TYPE = "Silo"