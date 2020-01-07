# The Voronizer

The Voronizer is a python script written to generate support structures and infill structures for 3D models.  The structures are based on Voronoi foam, and allow the user a lot of control over the resulting model.

This code (and the algorithms within) are described by the paper freely available online: http://utw10945.utweb.utexas.edu/sites/default/files/2019/153%20Using%20Parallel%20Computing%20Techniques%20to%20Algorithmic.pdf

## Setting Up Your Machine

This script uses CUDA, a parallel processing library.  It requires that you have an Nvidia graphics card installed and that your computer is set up with the CUDA Toolkit.  Instructions are available here: https://developer.nvidia.com/how-to-cuda-python

The following packages are used:
numba, numpy, math, os, time, matplotlib, PIL, struct, operator, skimage, mpl_toolkits

I recommend using the Anaconda Python 3.X package, which comes with all the relevant packages pre-installed.
https://www.anaconda.com/distribution/

Some alternatives to having your own GPU installed are listed on the above website.

## Running the Script

First, place the STL of the file you want to produce the Voronoi infill and/or supports of into the 'Input' folder.

Next, open the userInput.py file in a text editor or scripting environment, such as Spyder (Which comes packaged with Anaconda) or Notepad.

Edit the options to best fit your needs.  If you only want the supports, set MODEL = False, if you only want the model set SUPPORT = False.  Edit line 15 to be the file name of your desired input STL, or delete the # symbol next to the demo model that you want to test.  It is recommended that all settings be left at their initial values at first to ensure compatability.

The script can be run through main.py.

After running the script, the model may need some post-processing to be compatable with your slicing software.  I recommend MeshLab, a free mesh-editing software available here: http://www.meshlab.net/.  To clean the model, I recommend using Filters > Cleaning and Repairing > Remove Non-Manifold Faces.  To smooth out the resulting model, I recommend the HC Laplacian Smooth filter, found under Filters > Smoothing Fairing and Deformation > HC Laplacian Smooth.  The HC Laplacian filter can be used iteratively to achieve the desired surface finish.  Finally, to export your model from MeshLab, go to File > Export Mesh As, and save it as a file type compatible with your slicing software.

## Troubleshooting

#### I got an error message

>UnicodeDecodeError: 'utf-8' codec can't decode byte 0xaa in position 80: invalid start byte

This (or something similar) means that the file is not formatted in a way that can be easily read by the STL reader.  If you get an error like this, try importing the file into MeshLab and then export it as an STL.

>CudaAPIError: Call to cuMemcpyDtoH results in UNKNOWN_CUDA_ERROR

This means that the resolution is off.  Often, it means that the input value makes the array too large for your graphics card to deal with.  To fix this, try decreasing the value assigned to RESOLUTION.  After changing the value, it's recommended that you restart the kernel before re-running the script, as this error can sometimes impact future iterations.

#### My file is taking a while to process

There are several things that can be done to speed up the software.  The fastest way is to decrease the resolution.  Another way to speed things up is to lower the triangle count of the input model.  I recommend going into MeshMixer with your input model, going to Filters > Remeshing, Simplification and Reconstruction > Simplification: Quadratic Edge Collapse Decimation, checking the 'Preserve Boundary of Mesh' box, and reducing the model to 50% of the initial triangle count (type 0.5 into the 'Percentage Reduction' text box).  For many models, there are many more triangles than are needed, so they can be simplified significantly with minimal loss of quality.
