import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import collections
import math

# Select parent folder to be analyzed
parent = '3D'

# Get all folders in parent
inf = list(os.walk(parent))
print(inf)
inf = inf[0][1]

# Create dictionary to store folders and after that sort them
angles= {}
angles_graph = {}

# Store folders in dictionary
for folder in inf:
    n_mes = int(folder.split('_')[0])
    angles[n_mes] = folder

# Sort dictionary by keys
angles = collections.OrderedDict(sorted(angles.items()))

# Iterate over all folders in parent (so iterate for all angles)
for n_folder in list(angles):
    folder = angles[n_folder]

    # Get all files in folder
    files = list(os.listdir(parent+'\\'+folder))

    # Get first file in folder
    # Only 1 file is evaluated but more files could be evaluated
    for file in files[:1]:
        # Path of the file
        path = parent+'\\'+folder+'\\'+file

        # Read matrix csv by separating with no header
        df= pd.read_csv(path, sep=';',header=None)

        # Create new copy with no vinculation
        C = np.array(df.values  ,copy=True)

        # Select portion of picture with no ropes
        C = C[200:300]

        # Create array where temperatures will be stored
        temp = []
        for a in range(0,C.shape[1]):
            # Get a column of pixels
            col = C[:,a]

            # Get the averate of the temperatures on the column to avoid noise
            mean = np.mean(col)

            # Append average
            temp.append(mean)

        filepath ='results'+parent+'\\'+folder.replace('.',',')



        # Create New figure
        fig = plt.figure(1)

        # Plot the image flipped in the vertical axis
        plt.imshow(df.values[:,::-1])

        # Determine axes titles
        plt.xlabel('x coordinate [px]')
        plt.ylabel('y coordinate [px]')

        # Resize plots
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

        # Add color bar
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        cbar = plt.colorbar(cax=cax)
        cbar.ax.get_yaxis().labelpad = -75
        cbar.ax.set_ylabel('Temperature [°C]', rotation=90)

        # Save figure
        plt.savefig(filepath+'_pic')

        # Create figure and add title
        fig = plt.figure(2)
        fig.suptitle("Results folder " + filepath, fontsize=16)

        # Create first subplot
        plt.subplot(2, 1, 1)
        C = C[:, ::-1]

        # Dinamically determine x-axis such that le falls in 0 and te falls in 100
        x0 = -68.62 + 4 - (1 - math.cos(math.radians(float(folder.split('_')[1].split(' ')[0])))) * 233 * 0.3
        x1 = 164.98 + 4 + (1 - math.cos(math.radians(float(folder.split('_')[1].split(' ')[0])))) * 233 * 0.85

        # Plot image and add title for axes
        plt.imshow(C, cmap='gray', interpolation='none', extent=[x0, x1, 40, 0])
        plt.xlabel('Chord percentage [%]')
        plt.ylabel('y coordinate [px]')
        plt.grid(True)

        # Create second subplot
        plt.subplot(2, 1, 2)
        xrange = np.linspace(x0, x1, len(temp))[::-1]
        plt.xlabel('Chord percentage [%]')
        plt.ylabel('Temperatuere [°C]')
        plt.grid(True)

        plt.plot(xrange, temp)
        plt.savefig(filepath + '_plot')

        # Show Figures
        plt.show()
