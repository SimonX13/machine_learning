from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np

import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(
            self, max_depth=max_depth, stump_class=RandomStumpInfoGain
        )

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:
    """
    YOUR CODE HERE FOR Q4
    Hint: start with the constructor __init__(), which takes the hyperparameters.
    Hint: you can instantiate objects inside fit().
    Make sure predict() is able to handle multiple examples.
    """
    max_depth = None
    num_trees = None
    forest = None
    X = None 
    y = None
    
    def __init__(self, num_trees, max_depth):
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.forest = np.zeros(num_trees)   
        #raise NotImplementedError()


    def fit(self, X, y):
        self.X = X
        self.y = y
        
        forest = []
        for i in range(self.num_trees):
            model = RandomTree(max_depth=self.max_depth)
            model.fit(X,y)
            forest.append(model)
        self.forest = forest
        #raise NotImplementedError()


    def predict(self, X_pred):
        n = X_pred.shape[0]
        y = np.zeros((self.num_trees, n))
        
        for i in range(self.num_trees):
            #print("the item in forest looks like")
            result = self.forest[i].predict(X_pred)
            y[i] = result
               
        y_final = np.zeros(n)

        for i in range(n):
            val, count = np.unique(y[:,i],return_counts=True)   
            y_final[i] = val[np.argmax(count)]
        return y_final

        #raise NotImplementedError()

