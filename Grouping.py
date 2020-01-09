####################################
######## For House Grouping ########
####################################
# By using the Spectral clustering.
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN


def Grouping(distance_matrix, threshold):
    
    #results = SpectralClustering(2).fit_predict(distance_matrix)
    # metric = 'precomputed' is to predict for the Distance Matrix.
    results = DBSCAN(eps = threshold, min_samples = 3, \
                     metric="precomputed").fit_predict(distance_matrix)
    
    
    return results

