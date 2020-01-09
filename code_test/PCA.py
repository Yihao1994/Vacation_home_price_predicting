######
# PCA
from Data_preprocess import *
from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np

# Normalization
X_PCA = X[:,1:len(attributeNames) - 1]
N_PCA, M_PCA = X_PCA.shape
attributeNames_PCA = attributeNames[1:len(attributeNames) - 1]
X_prime = X_PCA - np.ones((N_PCA, 1))*X_PCA.mean(0)
X_prime = X_prime*(1/np.std(X_prime, 0))      # Normalization
print('########################')
print('Attributes used for PCA:')
for j, attribute_name in enumerate(attributeNames_PCA):
    print('Attribute name:', attribute_name, ', ', 'column index:', j)

# PCA by svd
U, S, Vh = svd(X_prime, full_matrices = False)
V = Vh.T
rho = (S*S)/((S*S).sum())
threshold = 0.95

#Projection
Z = X_prime @ V


##############
# Plot PCA rho
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components', fontsize = 25);
plt.xlabel('Principal component', fontsize = 20);
plt.ylabel('Variance distribution', fontsize = 20);
plt.legend(['Individual','Cumulative','Threshold'])
plt.tick_params(labelsize = 20)      #tickmarks
plt.grid()
plt.show()

####################
# Plot PCA histogram 
PCA_pickup = [0,1,2,3]
legendStrs = ['PC'+str(e+1) for e in PCA_pickup]
c = ['r','g','b','m']
bw = 0.2
r = np.arange(1,M_PCA+1)
plt.figure()
for i in PCA_pickup:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, range(0, len(attributeNames_PCA)))
plt.xlabel('Attributes', fontsize = 20)
plt.ylabel('Component coefficients', fontsize = 20)
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients', fontsize = 25)
plt.tick_params(labelsize = 15)      #tickmarks
plt.show()
