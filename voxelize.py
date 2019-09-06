from numba import cuda
import numpy as np
import slice
from struct import unpack
from operator import itemgetter
TPB = 8
# From https://github.com/cpederkoff/stl-to-voxel

def voxelize(inputFilePath, resolution):
    mesh = list(read_stl_verticies(inputFilePath))
    modelSize = np.array([0,0,0])
    pointList = list(map(list,sum(mesh,())))
    for i in range(3):
        pointList = sorted(pointList, key=itemgetter(i))
        modelSize[2-i] = pointList[-1][i]-pointList[0][i]
    (scale, shift, bounding_box) = slice.calculateScaleAndShift(mesh, resolution)
    mesh = list(slice.scaleAndShiftMesh(mesh, scale, shift))
    vol = np.zeros((bounding_box[2],bounding_box[0],bounding_box[1]), dtype=bool)
    for height in range(bounding_box[2]):
        lines = slice.toIntersectingLines(mesh, height)
        prepixel = np.zeros((bounding_box[0], bounding_box[1]), dtype=bool)
        linesToVoxels(lines, prepixel)
        vol[height] = prepixel
    vol = padVoxelArray(vol,2)
    print("Voxelize complete!")
    return toFRep(vol), modelSize

def linesToVoxels(lineList, pixels):
    for x in range(len(pixels)):
        isBlack = False
        lines = list(findRelevantLines(lineList, x))
        targetYs = list(map(lambda line:int(generateY(line,x)),lines))
        for y in range(len(pixels[x])):
            if isBlack:
                pixels[x][y] = True
            if y in targetYs:
                for line in lines:
                    if onLine(line, x, y):
                        isBlack = not isBlack
                        pixels[x][y] = True
        if isBlack:
            print("an error has occured at x%sz%s"%(x,lineList[0][0][2]))

def findRelevantLines(lineList, x, ind=0):
    for line in lineList:
        same = False
        above = False
        below = False
        for pt in line:
            if pt[ind] > x:
                above = True
            elif pt[ind] == x:
                same = True
            else:
                below = True
        if above and below:
            yield line
        elif same and above:
            yield line

def generateY(line, x):
    if line[1][0] == line[0][0]:
        return -1
    ratio = (x - line[0][0]) / (line[1][0] - line[0][0])
    ydist = line[1][1] - line[0][1]
    newy = line[0][1] + ratio * ydist
    return newy


def onLine(line, x, y):
    newy = generateY(line, x)
    if int(newy) != y:
        return False
    if int(line[0][0]) != x and int(line[1][0]) != x and (max(line[0][0], line[1][0]) < x or min(line[0][0], line[1][0]) > x):
        return False
    if int(line[0][1]) != y and int(line[1][1]) != y and (max(line[0][1], line[1][1]) < y or min(line[0][1], line[1][1]) > y):
        return False
    return True

def BinarySTL(fname):
    fp = open(fname, 'rb')
    Header = fp.read(80)
    nn = fp.read(4)
    Numtri = unpack('i', nn)[0]
    record_dtype = np.dtype([
        ('normals', np.float32, (3,)),
        ('Vertex1', np.float32, (3,)),
        ('Vertex2', np.float32, (3,)),
        ('Vertex3', np.float32, (3,)),
        ('atttr', '<i2', (1,) )
    ])
    data = np.fromfile(fp, dtype=record_dtype, count=Numtri)
    fp.close()

    Normals = data['normals']
    Vertex1 = data['Vertex1']
    Vertex2 = data['Vertex2']
    Vertex3 = data['Vertex3']

    p = np.append(Vertex1, Vertex2, axis=0)
    p = np.append(p, Vertex3, axis=0)  #list(v1)
    Points = np.array(list(set(tuple(p1) for p1 in p)))
    return Header, Points, Normals, Vertex1, Vertex2, Vertex3

def AsciiSTL(fname):
    with open(fname, 'r') as input_data:
        # Skips text before the beginning of the interesting block:
        init = False
        triangles = []
        verticies = []
        for line in input_data:
            if line.strip() == 'outer loop':  # Or whatever test is needed
                init = True
                verticies = []
                continue
            # Reads text until the end of the block:
            elif line.strip() == 'endloop':
                init = False
                triangles.append(verticies)
                continue
            elif init:
                words = line.strip().split(' ')
                assert words[0] == 'vertex'
                verticies.append((float(words[1]), float(words[2]), float(words[3])))
    return triangles

def IsAsciiStl(fname):
    with open(fname,'rb') as input_data:
        line = input_data.readline()
        if line[:5] == b'solid':
            return True
        else:
            return False

def read_stl_verticies(fname):
    if IsAsciiStl(fname):
        for (i,j,k) in AsciiSTL(fname):
            yield (tuple(i),tuple(j),tuple(k))
    else:
        head, p, n, v1, v2, v3 = BinarySTL(fname)
        for i, j, k in zip(v1, v2, v3):
            yield (tuple(i), tuple(j), tuple(k))

def padVoxelArray(voxels,padding):
    shape = voxels.shape
    new_shape = (shape[0]+2*padding,shape[1]+2*padding,shape[2]+2*padding)
    vol = np.zeros(new_shape, dtype=float)
    for a in range(shape[0]):
        for b in range(shape[1]):
            for c in range(shape[2]):
                vol[a+padding,b+padding,c+padding] = voxels[a,b,c]
    return vol

@cuda.jit
def toFRepKernel(d_u,d_v):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i >= dims[0] or j >= dims[1] or k >= dims[2]:
        return
    if d_u[i,j,k]==1:
        d_v[i,j,k]=-0.01
    else:
        d_v[i,j,k]=0.01

def toFRep(u):
    d_u = cuda.to_device(u)
    dims = u.shape
    d_v = cuda.device_array(dims,dtype=np.float32)
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    toFRepKernel[gridSize, blockSize](d_u,d_v)
    return d_v.copy_to_host()