import numpy as np
import utils


class DecisionStumpEquality:
    """
    This is a decision stump that branches on whether the value of X is
    "almost equal to" some threshold.

    This probably isn't a thing you want to actually do, it's just an example.
    """

    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape # 400 x 2
       
        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)
        # print(count)
        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)
        # print(y_mode)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)
        # print(minError)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = np.round(X[i, j])
                

                # Find most likely class for each split
                is_almost_equal = np.round(X[:, j]) == t
                y_yes_mode = utils.mode(y[is_almost_equal])
                y_no_mode = utils.mode(y[~is_almost_equal])  # ~ is "logical not"

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[np.round(X[:, j]) != t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode

    def predict(self, X):
        n, d = X.shape
        X = np.round(X)

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] == self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat


class DecisionStumpErrorRate:
    
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None
    
    def fit(self, X, y):
        """YOUR CODE HERE FOR Q6.2"""
        n, d = X.shape
        # print(X)
        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = X[i, j]

                # Find most likely class for each split
                is_almost_equal = X[:, j] > t
                
                y_yes_mode = utils.mode(y[is_almost_equal])
                y_no_mode = utils.mode(y[~is_almost_equal])  # ~ is "logical not"

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[X[:, j] < t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode

        #raise NotImplementedError()

    def predict(self, X):
        """YOUR CODE HERE FOR Q6.2"""
        n, d = X.shape
        # X = np.round(X)

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] > self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat
        #raise NotImplementedError()


def entropy(p):
    """
    A helper function that computes the entropy of the
    discrete distribution p (stored in a 1D numpy array).
    The elements of p should add up to 1.
    This function ensures lim p-->0 of p log(p) = 0
    which is mathematically true, but numerically results in NaN
    because log(0) returns -Inf.
    """
    plogp = 0 * p  # initialize full of zeros
    plogp[p > 0] = p[p > 0] * np.log(p[p > 0])  # only do the computation when p>0
    return -np.sum(plogp)


class DecisionStumpInfoGain(DecisionStumpErrorRate):
    # This is not required, but one way to simplify the code is
    # to have this class inherit from DecisionStumpErrorRate.
    # Which methods (init, fit, predict) do you need to overwrite?
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    """YOUR CODE HERE FOR Q6.3"""
    def fit(self, X, y):
        """YOUR CODE HERE FOR Q6.2"""
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)
        
        maxInfoGain = 0.0
        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                t = X[i, j]
                is_larger = X[:, j] > t
                y_mode_yes = utils.mode(y[is_larger])
                y_mode_no = utils.mode(y[~is_larger]) 

                if len(y[is_larger])==0 or len(y[~is_larger])==0:
                    continue

                y_leaf_1 = entropy(np.bincount(y[is_larger], minlength=2) / len(y[is_larger])) 
                y_leaf_2 = entropy(np.bincount(y[~is_larger], minlength=2) / len(y[~is_larger])) 
                infoGain = entropy(np.bincount(y, minlength=2) / n) - len(y[is_larger]) / n * y_leaf_1 - len(y[~is_larger]) / n * y_leaf_2

                if infoGain > maxInfoGain:
                    maxInfoGain= infoGain
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_mode_yes
                    self.y_hat_no = y_mode_no
        
    # def predict(self, X):
    #     n, d = X.shape
    #     y = np.zeros(n)

    #     for i in range(n):
    #         if X[i,0] > -80.305106 or X[i,1] > 37.669007:
    #             y[i] = 0
    #         else:
    #             y[i] = 1

    #     return y
