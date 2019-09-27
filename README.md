# The Voronizer

!!WORK IN PROGRESS!!

COMPLETE INSTRUCTIONS WILL BE AVAILABLE BY THE END OF OCTOBER

The Voronizer is a python script written to generate support structures and infill structures for 3D models.  The structures are based on Voronoi foam, and allow the user a lot of control over the resulting model.

# Setting Up Your Machine

This script uses CUDA, a parallel processing library.  It requires that you have an Nvidia graphics card installed and that your computer is set up with the CUDA Toolkit.  Instructions are available here: https://developer.nvidia.com/how-to-cuda-python

I recommend using the Anaconda Python 3.X package.

Some alternatives to having your own GPU installed are listed on the above website.

# Running the Script

First, place the STL of the file you want to produce the Voronoi infill and/or supports of into the 'Input' folder.

Next, open the main.py file in a text editor or scripting environment, such as Spyder (Which comes packaged with Anaconda) or Notepad.

Edit lines 13 through 28 to best suit your needs.  If you just want the supports, set MODEL = False, if you just want the model set SUPPORT = False.  Edit line 28 to be the file name of your desired input STL, or delete the # symbol next to the demo model you want to test.  It is recommended that all settings be left at their initial values at first to ensure compatability.

After running the script, the model may need some post-processing to be compatable with your slicing software.  I recommend MeshLab, a free mesh-editing software available here: http://www.meshlab.net/.  To clean the model, I recommend using Filters > Cleaning and Repairing > Remove Non-Manifold Faces.  To smooth out the resulting model, I recommend the HC Laplacian Smooth filter, found under Filters > Smoothing Fairing and Deformation > HC Laplacian Smooth.  The HC Laplacian filter can be used iteratively to achieve the desired surface finish.  Finally, to export your model from MeshLab, go to File > Export Mesh As, and save it as a file type compatible with your slicing software.

# Troubleshooting

