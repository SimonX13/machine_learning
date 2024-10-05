import numpy as np


class NaiveBayes:
    """
    Naive Bayes implementation.
    Assumes the feature are binary.
    Also assumes the labels go from 0,1,...k-1
    """

    p_y = None
    p_xy = None

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y):
        n, d = X.shape
         
        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n
        print("p_y")
        print(p_y)

        """YOUR CODE HERE FOR Q3.3"""

        # Compute the conditional probabilities i.e.
        # p(x_ij=1 | y_i==c) as p_xy[j, c]
        # p(x_ij=0 | y_i==c) as 1 - p_xy[j, c]
        # print(np.ones((d, k)))
        #p_xy = 0.5 * np.ones((d, k))
        p_xy = np.zeros((d, k)) # 100 x 4
        
        for i in range(d):
            for j in range(k):
                numerator = np.bincount(X[:,i])
                number = 0 
                for z in range(n): # among all the rows, the column that has 1 and y[this row] is this organization
                    if X[z, i]==1:
                        if y[z] == j:
                            number = number + 1
                numerator = number / n # the probability that column i has this probability of 1 such that the newsgroup is y[z]
                p_xy[i, j] = numerator / p_y[j]
        # TODO: replace the above line with the proper code

        #raise NotImplementedError()


        self.p_y = p_y
        self.p_xy = p_xy

    def predict(self, X):
        n, d = X.shape
        k = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(n)
        for i in range(n):

            probs = p_y.copy()  # initialize with the p(y) terms
            for j in range(d):
                if X[i, j] != 0:
                    probs *= p_xy[j, :]
                else:
                    probs *= 1 - p_xy[j, :]

            y_pred[i] = np.argmax(probs)

        return y_pred


class NaiveBayesLaplace(NaiveBayes):
    def __init__(self, num_classes, beta=0):
        super().__init__(num_classes)
        self.beta = beta

    def fit(self, X, y):
        """YOUR CODE FOR Q3.4"""
        
        
        #raise NotImplementedError()

        n, d = X.shape
         
        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n
        print("p_y")
        print(p_y)


        # Compute the conditional probabilities i.e.
        # p(x_ij=1 | y_i==c) as p_xy[j, c]
        # p(x_ij=0 | y_i==c) as 1 - p_xy[j, c]
        # print(np.ones((d, k)))
        #p_xy = 0.5 * np.ones((d, k))
        p_xy = np.zeros((d, k)) # 100 x 4
        
        for i in range(d):
            for j in range(k):
                numerator = np.bincount(X[:,i])
                number = 0 
                for z in range(n): # among all the rows, the column that has 1 and y[this row] is this organization
                    if X[z, i]==1:
                        if y[z] == j:
                            number = number + 1
                numerator = number / n # the probability that column i has this probability of 1 such that the newsgroup is y[z]
                p_xy[i, j] = (numerator + self.beta ) / ( p_y[j] + k*self.beta)
        
        print(p_xy[:,0])
        self.p_y = p_y
        self.p_xy = p_xy
