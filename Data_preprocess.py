###############################################################
########## Data pre-process, including img processed ##########
###############################################################
# In[Package load] 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import date_transfer as dtf
import os
import glob
import img2vector  as i2v
import Delta_img_similarity as Dis
import Grouping as Gp

# In[Image process]
# [1].Image path
imgfolder_dir = 'DATA/aerial_photos'
paths = glob.glob(os.path.join(imgfolder_dir, '*.png'))
nr_houses = len(paths)     # How many different houses in this dataset

name = np.zeros(nr_houses, int)
for i in range(nr_houses):
    full_name = os.path.basename(paths[i])
    name[i] = (os.path.splitext(full_name))[0]
name.sort()


# [2].Image name into dict
# Create the dict for accommodations.
house_pk_dict = {name[i]:i for i in range(nr_houses)}


# [3].Implement the 'img2vector' function
img_Vectors = i2v.img2vector(paths, nr_houses)
# So the img vector of 'NAME' can be directly imported by the syntax like below:
# img_vector = img_Vectors[:,house_pk_dict[NAME]]. But it is  better to give an zero
# matrix to the dataframe, which leave a place to take image data.


# [4].Calculate the similarity matrix
similarity_Delta_matrix = Dis.similarity_Delta_matrix_calculation(nr_houses, img_Vectors)
distance_matrix = 1 - similarity_Delta_matrix

# [5].Return the house clustering results.
# By using the DBSCAN.
# This value here is chosen by the distance in (91750 between 91757 - 0.1).
threshold = distance_matrix[house_pk_dict[91750], house_pk_dict[91757]]-0.1    # Hyperparameters
clusters = Gp.Grouping(distance_matrix, threshold)


# [6].Track back the image house_pk of the clusters
cluster_label = np.unique(clusters)
labels_information = []
labels_house_number = []             # Will be used into the data split.
print('#####################')
print('Number of clusters: ', len(cluster_label))
for i, cluster in enumerate(cluster_label):
    # Find the houses_index in each cluster label.
    index_in_this_label = np.nonzero(clusters == cluster)[0]
    labels_information.append(index_in_this_label)
    
    # Track back the house_pk from the house_index above
    labels_house_nr = np.zeros(len(index_in_this_label), int)
    for j, index_house_nr in enumerate(index_in_this_label):
        labels_house_nr[j] = list(house_pk_dict.keys())[list(house_pk_dict.values()).index(index_house_nr)]
    labels_house_number.append(labels_house_nr)
        
    print('##########')
    print('Cluster %d: ' % cluster)
    print(labels_house_nr)
    print('')



# In[Tabular data process]
##################
# [1].Data loading
filename = 'DATA/data.csv'
dataframe  = pd.read_csv(filename)


###########################################
# [2].Handle the date information into days
date = dataframe.date_in.astype(str)      # Date information

# Since the minimum is '2016-01-16', and the maximum is '2019-04-17'
Base = '2016-01-01'
day_of_date = dtf.date_transfer(date, Base)
dataframe.date_in = day_of_date


##########################################################
# [3].Drop the 'agency_id'. Transfer database to an array.
database = dataframe.drop(['agency_id'], axis = 1)
attributeNames = np.asarray(database.columns)


##############################################
# [4].Print the attributes before combination.
print('#######################################')
print('Attributes used for recombine prediction:')
for j, attribute_name in enumerate(attributeNames):
    print('Attribute name:', attribute_name, ', ', 'column index:', j)
print('')

###########################################
# Make it array!
X = np.asarray(database, dtype = np.float64)


########################################################
# [4].Handle both the 'build_year' and 'renovation_year' 
# Since a later 'build_year' or 'renovation_year' will be more popular to customers
# So here they will be assign to corresponding int values. The higher value means the house is younger.
build_index = [i for i in range(len(attributeNames)) if \
               attributeNames[i] == 'build_year'][0]
renovation_index = [j for j in range(len(attributeNames)) if \
                    attributeNames[j] == 'renovation_year'][0]
build_year = X[:, build_index].astype(int)
renovation_year = X[:, renovation_index].astype(int)
year_class = np.unique(np.append(build_year, renovation_year))
classDict_year = dict(zip(year_class,range(len(year_class))))

build_year = np.array([classDict_year[j] for j in build_year])
X[:,build_index] = build_year

renovation_year = np.array([classDict_year[k] for k in renovation_year])
X[:,renovation_index] = renovation_year



# In[Data Recombined into time-series, and into house-groups]
###############################################################################
######### Recombine the data basing on the time-series and img-groups #########
date_index = [ii for ii in range(len(attributeNames)) if \
               attributeNames[ii] == 'date_in'][0]

K = 7   # An hyperparameters that can be learnt basing  on KNN
# K above means thehow many days closed to this particular date_information[jj] day.
date_information = np.unique(X[:,date_index])
date_information.sort()
date_up = np.zeros(len(date))
date_down = np.zeros(len(date))

# Find where house_pk is, in the attribute's column
house_pk_index = [pp for pp in range(len(attributeNames)) if \
               attributeNames[pp] == 'house_pk'][0]

X_timeseries = []
for jj, datedate in enumerate(date_information):
    # Date up and down limit
    date_down[jj] = datedate - K
    date_up[jj]   = datedate + K
    
    # [1].Split the entire data into time-series
    X_time = X[((X[:,date_index] >= date_down[jj]) & (X[:,date_index] <= date_up[jj])),:]    
    
    # [2].Split the data inside the dataseries basing on the img-grouping
    X_group = []
    for kk, house_names in enumerate(labels_house_number):
        
        X_one_group = []
        for iii, one_house_name in enumerate(house_names):
            # Tracking the index of where each house_pk in which part of the X_time.
            one_house_index = np.squeeze(np.where(X_time[:,house_pk_index] == one_house_name))
            data_for_this_house = X_time[one_house_index]
            X_one_group.append(data_for_this_house)
        
        X_one_group_all_together = X_one_group[0]
        for jjj in range(1, len(X_one_group)):
            X_one_group_all_together = np.vstack((X_one_group_all_together, X_one_group[jjj]))
            
        # Now, this X_group is the 'grouped' timeseries data basing on this X_time
        X_group.append(X_one_group_all_together)
        
    X_timeseries.append(X_group)
    
# The X_timeseries here is a data, who has 6 groups different kinds of house clusters, in one time cycle.
# And each time cycle take the neighboor day of ±7 days (A week). So when training with the further model,
# a time piece will be picked up from the X_timeseries, and the training process will be bsaed on 
# [each group] under this time piece. 


'''
# Code here for checking the which group this house is under...
house_nr_check = 27735
for i in range(len(labels_house_number)):
    KKK = np.where(labels_house_number[i] == house_nr_check)[0]
    
    if KKK.size == 0:
        continue
    
    print('House ' + str(house_nr_check) + ' is under Group_%d' % (i - 1))
    print('Index %d' % KKK)
'''


'''
# Code here is a plot for time-series data for 'house_pk_4_plot'.
house_pk_4_plot = 90076
data_4_plot = database[['house_pk', 'date_in', 'price']]
X_4_plot = np.asarray(data_4_plot, dtype = np.float64)
index_4_plot = np.where(X_4_plot[:,0] == house_pk_4_plot)[0]
X_4_this_house = X_4_plot[index_4_plot]  # Take this house from X_4_plot
X_4_this_house = X_4_this_house[:,1:]    # Remove house_pk
arg = np.argsort(X_4_this_house[:,0])    # Track the index
X_4_this_house_right_timeseries = X_4_this_house[arg]

# Plot time-series data for 'house_pk_4_plot'
plt.figure()
plt.plot(X_4_this_house_right_timeseries[:,0], X_4_this_house_right_timeseries[:,1])
plt.title('Price Curve for House_%d' % house_pk_4_plot, fontsize = 26)
plt.xlabel('Days away from 2016-01-01', fontsize = 23)
plt.ylabel('Price', fontsize = 23)
plt.tick_params(labelsize=18) # reduce the fontsize of the tickmarks
'''

