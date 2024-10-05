"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        """YOUR CODE HERE FOR Q1"""
        t, d = X_hat.shape  # T x D
        n, d = self.X.shape # N x D
    
        difference_list = utils.euclidean_dist_squared( self.X, X_hat) # N x T
        y_hat = np.zeros(t)
        for i in range(t):
            difference_list_sort = np.sort(difference_list[:,i]) # sort entire
            difference_list_sort = difference_list_sort[:self.k] # slice form index 0 to k
            
            # get y's value of the distance of sorted list using the index 
            indice_list = np.isin(difference_list[:,i], difference_list_sort)
            outputs = np.bincount(self.y[indice_list])
            y_hat[i] = np.argmax(outputs)
        return y_hat
        # raise NotImplementedError()

