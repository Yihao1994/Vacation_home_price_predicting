################################################
###### XGBoost Model for price prediction ######
################################################
# In[Preprocess data loading]
from Data_preprocess import *
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
from datetime  import datetime
import time
import numpy as np
import pickle

# In[Pick up a time piece]

# [1].Transfer the base
sec_for_a_day = 24*60*60   # How many secs there ae for a day
Base_XGBoost = datetime.strptime(Base, '%Y-%m-%d')
un_time_Base_XGBoost = time.mktime(Base_XGBoost.timetuple())

# [2].Transfer the time_piece
##########################
##### Pick a day ! #######
##########################
time_piece = '2019-01-23'
##########################
date_XGBoost = datetime.strptime(time_piece, '%Y-%m-%d')
un_time_date_XGBoost = time.mktime(date_XGBoost.timetuple())

# [3].Calculate the difference between time-piece and Base
day_of_date_XGBoost = (un_time_date_XGBoost - un_time_Base_XGBoost)/sec_for_a_day

# [4].Find this time piece's index in time series. 
time_series_index = np.where(date_information == day_of_date_XGBoost)[0][0]

# [5].Attributes for XGboost
attributeNames_hat = np.asarray((database.drop(['house_pk', 'price'], axis = 1)).columns)
print('#####################################################################')
print('Attributes used for XGBoost prediction, only for importance checking:')
for j, attribute_name in enumerate(attributeNames_hat):
    print('Attribute name:', attribute_name, ', ', 'column index:', j)
print('')


# In[Pick up a group/cluster]
price_index = [k for k in range(len(attributeNames)) if \
                   attributeNames[k] == 'price'][0]
# Model Training
models_path = "Model_data/"
for nr_cluster in range(len(labels_information)):
    
    X_specific = (X_timeseries[time_series_index])[nr_cluster]
    
    # Just in case of 'EMPTY', which means, this group of house is not booked at
    # that time.
    if X_specific.size == 0:
        continue
    
    else:
        print('#####################')
        print('Under the ', time_piece)
        print('For Group %d' % (nr_cluster - 1))
        # [5].Split the real X and Y
        Y = X_specific[:,price_index]
        
        # Remobe the house_pk, since it has been shown through the groups
        # Remove the price_index, since thats y
        X_hat = np.delete(X_specific, [house_pk_index, price_index], axis = 1)
        
        # X_hat, Y and attributeNames_hat is what used here for XGBoost
        X_train, X_test, y_train, y_test = train_test_split(X_hat, Y, \
                                                            test_size = 0.1, random_state = 100)
        
        # DMatrix
        dtrain = xgb.DMatrix(data = X_train, label = y_train)
        dtest  = xgb.DMatrix(data = X_test, label = y_test)
        
        # Parameters of XGBoost setting
        param = {'max_depth': 7, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}
        param['nthread'] = 4
        param['seed'] = 100
        param['eval_metric'] = 'rmse'
        
        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        
        num_round = 10
        BST_with_evallist = xgb.train(param, dtrain, num_round, evallist)  
        
        pickle.dump(BST_with_evallist, open(models_path + 'BST_with_evallist_Group_' + str(nr_cluster-1) + \
                                            '_' + time_piece + '.pickle.dat', "wb"))

# BST is for traing, bst is for testing, respectively.
###############################################################################


# In[Load the model then test on Group -1]
test_group = 0  # Cluster -1
bst_with_evallist = pickle.load(open(models_path + "BST_with_evallist_Group_-1_2019-01-23.pickle.dat", "rb"))

X_test_now = (X_timeseries[time_series_index])[test_group]

# True value
y_predict = X_test_now[:,price_index]    
X_test_now = np.delete(X_test_now, [house_pk_index, price_index], axis = 1)

# predictd value
dpredict = xgb.DMatrix(X_test_now)
ypred_with_evallist = bst_with_evallist.predict(dpredict)

RMSE = np.sqrt(((ypred_with_evallist - y_predict) ** 2).mean())

print('######################################')
print("RMSE of bst_with_evallist ：", RMSE)
print('The r2 score for this Group %d is : %4f' % (test_group - 1, \
                                                   metrics.r2_score(y_predict, ypred_with_evallist)))
print('')
'''

# In[Importance plot & Trees plotted into .pdf]
# Attribute's importance plot
'''
xgb.plot_importance(bst_with_evallist)

# Tree plot and saved into pdf
num_trees = len(bst_with_evallist.get_dump())
for tree_index in range(num_trees):
    dot = xgb.to_graphviz(bst_with_evallist, num_trees = tree_index)
    dot.render("trees/tree{}".format(tree_index))
'''

