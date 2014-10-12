# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mplml
from matplotlib import pyplot as plt
import random


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
thelabelsmap = ['a','b','c','d','e','f']
thelabelsstrings = np.matrix([['a'] for x in thelabels])

thelabelsstrings = np.matrix(thelabelsstrings)
thelabelsmap = np.matrix(thelabelsmap)

"""
Perform PCA, and visualize
Note that for PCA you may use built-in functions, you should not though for your k-means implementation
"""

B = mplml.PCA(thedata)
index = 0
plt.scatter(B.Y[:,0],B.Y[:,1], c=np.array(thelabels.transpose()))#, label=np.array(thelabelsstrings.transpose()))
#plt.legend()
plt.title('PCA projection of phonedata in two dimensions')
plt.show()

# For visualization, something like the following may help
# scatter(X_r[:,0],X_r[:,1],c=array(thelabels.transpose()))


"""
Now implement K-Means
There are built in functions for this - but the should be done by you!
"""

#recalculates the center of each cluster in kclusters
def update_centroids(kclusters, centroids):
    index = 0
    for cluster in kclusters:
        if len(cluster) > 0:
            centroids[index] = np.mean(cluster, axis=0)
        index += 1

    return centroids

def SSE(centroids, kclusters):
    cluster_sse = [0 for x in range(len(kclusters))]
    sse = 0
    index = 0
    for centroid in centroids:
        for point in kclusters[index]:
            squared_error = np.square(np.linalg.norm(centroid - point))
            sse += squared_error
            cluster_sse[index] += squared_error
        index+= 1

    return sse, cluster_sse
def perform_kmeans(kvalue, numiters, thedata ):
    centroids = np.zeros((kvalue,561))

    #Assigns each point to a cluster
    cluster_assignments = [random.randint(1,kvalue) for _ in range(rows)]

    for iteration in range(numiters):

        #Partition clusters into a list of lists of points
        kclusters = [[] for x in range(kvalue)]
        index = 0
        for assignment in cluster_assignments:
            kclusters[assignment - 1].append(thedata[index])
            index += 1

        centroids = update_centroids(kclusters, centroids)

        index = 0
        for point in thedata:
            centroid_distances = [np.linalg.norm(point - x) for x in centroids]
            cluster_assignments[index] = np.argmin(centroid_distances) + 1
            index += 1

    sse, cluster_sse = SSE(centroids, kclusters)
    return cluster_assignments, sse, cluster_sse
# May wish to initialize k-value, iterations and centroids
kvalue = 6
numiters = 10
rows, columns = thedata.shape
centroids = np.zeros((kvalue,561))

cluster_assignments, sse, cluster_sse = perform_kmeans(kvalue,numiters,thedata)


plt.scatter(B.Y[:,0],B.Y[:,1], c=np.array(np.matrix(cluster_assignments).transpose()))#, c=np.array(thelabels.transpose()), label=np.array(thelabelsstrings.transpose()))
#plt.legend()
plt.title('PCA projection of phonedata in two dimensions')
plt.show()

    ##reassign points to closest centroid
    #for point in thedata
    ##find nearest cluster c_nearest and make cluster_assignments[point] = c_nearest


#np.linalg.norm(x1-x2)

#centroids: np.mean(myKMeans.thedata[:,:], axis=0)




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
