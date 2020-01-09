#For the presentation in 02.January.2020
###############################################################################
########################### Neural network Main doc ###########################
###############################################################################
import numpy as np
import matplotlib.pyplot as plt 
import scipy
from Data_preprocess import *
import pickle
import time
import math
from sklearn.model_selection import train_test_split

#Initialize the weight and bias for a L layers' NN
def initialize_parameters_deep(layer_dims, m_train):
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], \
                   layer_dims[l - 1])*np.sqrt(2/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros([layer_dims[l], 1])
        
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters



def relu(Z):
    
    A = np.maximum(0.03*Z, Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache



def minibatch(X, Y, mini_batch_size):
    global mini_batches
    m = X.shape[1]
    mini_batches = []
    
    #Shuffle the dataset
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[permutation]
    
    num_howmany_minibatches = math.floor(m/mini_batch_size)
    for i in range(0, num_howmany_minibatches):
        mini_batch_X = shuffled_X[:, i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    if m % mini_batch_size !=0:
        mini_batch_X = shuffled_X[:, (i+1)*mini_batch_size:m]
        mini_batch_Y = shuffled_Y[(i+1)*mini_batch_size:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches
        
    

#Deep NN forward
def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache_Z = relu(parameters['W' + str(l)] @ A_prev + parameters['b' + str(l)])
        cache = ((A_prev, parameters['W' + str(l)], parameters['b' + str(l)]), cache_Z)
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL = parameters['W' +  str(L)] @ A + parameters['b' + str(L)]     # No activation in the last stage
    cache = ((A, parameters['W' + str(L)], parameters['b' + str(L)]), AL)
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches



#Cost compuation
def compute_cost(AL, Y, caches, Lambd):
    
    m = Y.shape[0]
    Q = len(caches)         # How many W do we have. For calculating the L2_regularization
    L2_regularization = 0   # An initial value    
    
    # Compute loss from aL and y.
    cost = np.sqrt(np.sum((AL - Y)**2)/m)    # RMSE
    residual = AL - Y
    
    for q in range(Q):
        L2_regularization = L2_regularization + (Lambd/(2*m))*(np.sum(np.square(caches[q][0][1])))
    
    
    cost_with_regular = cost + L2_regularization
    cost = np.squeeze(cost)
    cost_with_regular = np.squeeze(cost_with_regular)
    assert(cost.shape == ())
    
    return cost_with_regular



def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True)
    
    # dZL = dAL*gL'(ZL)
    dZ[Z <= 0] *= 0.03
    
    assert (dZ.shape == Z.shape)
    
    return dZ



#Deep NN backwrds
def L_model_backward(AL, Y, caches, Lambd):
    
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = 2*(AL - Y)/m    # Derivative of MSE
    
    current_cache = caches[L-1]
    AL_minus1 = current_cache[0][0]
    m = AL_minus1.shape[1]           #m is the nr of the trainging example
    WL = current_cache[0][1]
    bL = current_cache[0][2]
    ZL = current_cache[1]
    gZL_dash  = 1
    dZL = dAL*gZL_dash
    dWL = dZL@(AL_minus1.T)/m + Lambd*WL/m
    dbL = np.sum(dZL, axis = 1, keepdims =  True)/m 
    dAL_prev = (WL.T)@dZL
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = dAL_prev, dWL, dbL
    
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        AL_prev = current_cache[0][0]
        WL = current_cache[0][1]
        bL = current_cache[0][2]
        ZL = current_cache[1]
        dZL = relu_backward(dAL_prev, ZL)
        dWL = dZL@(AL_prev.T)/m + Lambd*WL/m
        dbL = np.sum(dZL, axis = 1, keepdims =  True)/m
        dAL_prev = (WL.T)@dZL
        dA_prev_temp, dW_temp, db_temp = dAL_prev, dWL, dbL
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads




###############################################################################
### Adam algorithm ###
######################
def Adam_init(parameters):
    L = len(parameters) // 2
    
    #Initialize the weight & Bias
    V = {}
    S = {}
    
    for l in range(L):
        V["dW" + str(l + 1)] = np.zeros([parameters["W" + str(l + 1)].shape[0], \
         parameters["W" + str(l + 1)].shape[1]])
    
        V["db" + str(l + 1)] = np.zeros([parameters["b" + str(l + 1)].shape[0], \
         parameters["b" + str(l + 1)].shape[1]])
    
        S["dW" + str(l + 1)] = np.zeros([parameters["W" + str(l + 1)].shape[0], \
         parameters["W" + str(l + 1)].shape[1]])
    
        S["db" + str(l + 1)] = np.zeros([parameters["b" + str(l + 1)].shape[0], \
         parameters["b" + str(l + 1)].shape[1]])
        
    return V, S



#Parameters updating with Adm algorithm
def update_parameters_Adam(parameters, grads, beta_1, beta_2, V, S, T, learning_rate, epsilon):
    
    L = len(parameters) // 2 # number of layers in the neural network
    V_corr = {}
    S_corr = {}
    
    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        #Momentum
        V["dW" + str(l + 1)] = beta_1*V["dW" + str(l + 1)] + (1-beta_1)*grads["dW" + str(l + 1)]
        V["db" + str(l + 1)] = beta_1*V["db" + str(l + 1)] + (1-beta_1)*grads["db" + str(l + 1)]
        
        #RMS prop
        S["dW" + str(l + 1)] = beta_2*S["dW" + str(l + 1)] + (1-beta_2)*np.power(grads["dW" + str(l + 1)], 2)
        S["db" + str(l + 1)] = beta_2*S["db" + str(l + 1)] + (1-beta_2)*np.power(grads["db" + str(l + 1)], 2)
        
        #Bias correction
        V_corr["dW" + str(l + 1)] = V["dW" + str(l + 1)]/(1-np.power(beta_1, T))
        V_corr["db" + str(l + 1)] = V["db" + str(l + 1)]/(1-np.power(beta_1, T))
        
        S_corr["dW" + str(l + 1)] = S["dW" + str(l + 1)]/(1-np.power(beta_2, T))
        S_corr["db" + str(l + 1)] = S["db" + str(l + 1)]/(1-np.power(beta_2, T))
        
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - \
        learning_rate*V_corr["dW" + str(l + 1)]/(np.sqrt(S_corr["dW" + str(l + 1)]) + epsilon)
        
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - \
        learning_rate*V_corr["db" + str(l + 1)]/(np.sqrt(S_corr["db" + str(l + 1)]) + epsilon)
        
    return parameters
###############################################################################



###############################################################################################
###################################### @ Main function @ ######################################
###############################################################################################
def L_layer_model(X, Y, layer_dims, m_train, decay_coeff, alpha0, num_epochs, Lambd, mini_batch_size, \
                  beta_1, beta_2, epsilon, print_cost = True):  
    
    Cost = []                         # keep track of cost
    ##################################
    # Weight W & Bias b initialization
    parameters = initialize_parameters_deep(layer_dims, m_train)
    
    ########################################
    # Momentum v & RMS prop s initialization    
    V, S = Adam_init(parameters)
    
    # Title print
    print('There are %d hidden layers inside this neural network' % (len(layer_dims) - 2))
    print('It will iterate for %d times' % (num_epochs))
    # Loop for different epochs
    for i in range(0, num_epochs):
        mini_batches = minibatch(X, Y, mini_batch_size)
        cost_with_regular = 0
        learning_rate = np.power(decay_coeff, i)*alpha0
        
        # Loop for mini-batch
        for t, mini_batch in enumerate(mini_batches):
            (mini_batch_X, mini_batch_Y) = mini_batch

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches =  L_model_forward(mini_batch_X, parameters)
            
            # Compute cost.
            cost_with_regular += compute_cost(AL, mini_batch_Y, caches, Lambd)
        
            # Backward propagation.
            grads = L_model_backward(AL, mini_batch_Y, caches, Lambd)
     
            # Update parameters.
            T = t + 1
            parameters = update_parameters_Adam(parameters, grads, beta_1, beta_2, \
                                                V, S, T, learning_rate, epsilon)
                    
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost_with_regular))
    
        if print_cost and i % 100 == 0:
            Cost.append(cost_with_regular)
            
    # plot the cost
    plt.plot(np.squeeze(Cost))
    plt.ylabel('Cost with regularization', fontsize = 28)
    plt.xlabel('Iterations (per hundreds)', fontsize = 28)
    plt.title("Final Learning rate = %3f, L2 Regularized fator = %3f" % (learning_rate, Lambd), fontsize = 25)
    plt.show()
    
    return parameters, Cost


################################
### Hyperparameters' setting ###
n_h_1 = 17
n_h_2 = 13
n_h_3 = 9
n_h_4 = 5
'''
n_h_5 = 4
n_h_6 = 13
n_h_7 = 9
n_h_8 = 7
n_h_9 = 5
'''
n_y = 1        # Since this is a cat or not classification
layer_dims = (X.shape[1], n_h_1, n_h_2, n_h_3, n_h_4, n_y)  # n_h_5, n_h_6, n_h_7, n_h_8, n_h_9, 
Lambd = 0                      # Regularization coeff
decay_coeff = 0.997              # Decay coeff for learning_rate
alpha0 = 0.01                    # Initial learning_rate
mini_batch_size = np.power(2,12)          # np.power(2, 5)   # Cell for a mini batch
num_epochs = 600                          # Go through entire batch for this times
beta_1 = 0.9                           # First moment coeff
beta_2 = 0.9                             # Second moment coeff
epsilon = 1e-8                            # Epsilon


train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(X, Y, test_size = 0.1, random_state=1)
train_set_x = train_set_x.T
test_set_x  = test_set_x.T
m_train = train_set_x.shape[1]

# Model training
parameters, Cost = L_layer_model(train_set_x, train_set_y, layer_dims, m_train, decay_coeff, alpha0, \
                           num_epochs, Lambd, mini_batch_size, beta_1, beta_2, epsilon, print_cost = True)


''' 
# Split the data using 10-Fold
K = 10
CV = model_selection.KFold(n_splits = K, shuffle = True)
k = 0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    train_set_x, train_set_y = X[:, train_index], Y[train_index]
    test_set_x, test_set_y = X[:, test_index], Y[test_index]
    m_train = train_set_x.shape[1]
    k+=1    
'''    

    

'''
#Function for testing the accuracy.
def predict(X, y, parameters, layer_dims, threshold):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > threshold:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print('For this %d hidden units neural network, the accuracy is:' % (len(layer_dims) - 2))
    print(str(np.sum((p == y)/m)))
        
    return p

#See how the model works during the training process~
threshold = 0.85
p = predict(train_set_x, train_set_y, parameters, layer_dims, threshold)
parameters['Layer_dimension'] = layer_dims
'''
'''
# Save the well-trained data
pickle.dump(parameters, open('well_trained_parameters_NN_Moons.txt', 'wb'))
'''