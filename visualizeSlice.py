import matplotlib.pyplot as plt
import os #Just used to set up file directory

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