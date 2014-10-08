# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mplml
from matplotlib import pyplot as plt


def load_data_from_file(f):
    arrays = []
    for line in open(f):
        new_string_array = line.split(' ')
        new_string_array = [x for x in new_string_array if x != '']
        new_array = [float(elem) for elem in new_string_array]
        arrays.append(new_array)
    return (np.matrix(arrays))

thedata = load_data_from_file("phonedata.txt")
thelabels = load_data_from_file("activitylabels.txt")

# Some imports will be useful
# import numpy as np
# import scipy as sp
# May want some matplotlib, numpy.linalg or other imports


"""
Perform PCA, and visualize
Note that for PCA you may use built-in functions, you should not though for your k-means implementation
"""
B = mplml.PCA(thedata)
print B.numrows
print B.numcols
plt.scatter(B.Y[:,0],B.Y[:,1],c=np.array(thelabels.transpose()))
plt.show()
"""
#plt.plot(B.Y[:,0],B.Y[:,1],c=array(thelabels.transpose()))
plt.plot(B.Y[0:20,0],B.Y[0:20,1], 'o', markersize=7,\
                 color='blue', alpha=0.5, label='class1')
plt.plot(B.Y[20:40,0], B.Y[20:40,1], '^', markersize=7,\
                 color='red', alpha=0.5, label='class2')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()
plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')
plt.show()
"""


# For visualization, something like the following may help
# scatter(X_r[:,0],X_r[:,1],c=array(thelabels.transpose()))


"""
Now implement K-Means
There are built in functions for this - but the should be done by you!
"""

# May wish to initialize k-value, iterations and centroids
kvalue = 6
numiters = 10
centroid = np.zeros((kvalue,561))

# initialize cluster centroid positions

# Repeat steps of k-means until convergence or a fixed number of iterations is done

"""
Determine SSE values for clusters, and data
"""



"""
Visualize the relationship between known class labels, and cluster assignments
"""

# The challenge is that cluster assignments and labels both have discrete values
# If you plot them, many samples land on the same spot
# This can be solved by scattering the data a bit to see the density
# around different assignments


# Something like the following may be useful
# scatter(array(closestcenter)+np.random.normal(0,0.1,(closestcenter.shape[0])),array(thelabels)+np.random.normal(0,0.1,(thelabels.shape[0],thelabels.shape[1])))
