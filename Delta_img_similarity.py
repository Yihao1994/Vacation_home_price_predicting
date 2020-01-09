##############################################
###### Check aerial images' similarity ######
#############################################
from scipy import spatial
import numpy as np


# FC2 vector to similarity value
def calculate_similarity(vector1, vector2):
    
    similarity = 1 - spatial.distance.cosine(vector1, vector2)
    
    return similarity



def similarity_Delta_matrix_calculation(nr_houses, img_Vectors):
    similarity_Delta_matrix = np.zeros([nr_houses, nr_houses])
    for ii in range(nr_houses):
        for jj in range(nr_houses):
            similarity_Delta_matrix[ii,jj] = calculate_similarity(img_Vectors[:,ii], img_Vectors[:,jj])
            
            
    return similarity_Delta_matrix






