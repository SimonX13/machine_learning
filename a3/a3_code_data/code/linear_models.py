import numpy as np
from numpy.linalg import solve


# Ordinary Least Squares
class LeastSquares:
    def fit(self, X, y):
        self.w = solve(X.T @ X, X.T @ y)

    def predict(self, X):
        return X @ self.w


# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares):
    # inherits the predict() function from LeastSquares
    def fit(self, X, y, v):
        """YOUR CODE HERE FOR Q2.1"""
        self.w = solve(X.T @  np.diag(v) @ X, X.T @ np.diag(v) @ y)
        
    def predict(self, X):
        return X @ self.w

class LinearModel:
    """
    Generic linear model optimizing custom function objects.
    A combination of:
    (1) optimizer and
    (2) function object
    prescribes the behaviour of the parameters, although prediction is
    always performed exactly the same: y_hat = X @ w.

    See optimizers.py for optimizers.
    See loss_fn.py for function objects, which must implement evaluate()
    and return f and g values corresponding to current parameters.
    """

    def __init__(self, loss_fn, optimizer, check_correctness=False):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.bias_yes = True
        self.check_correctness = check_correctness

        # For debugging and making learning curves
        self.fs = []
        self.nonzeros = []
        self.ws = []

    def optimize(self, w_init, X, y):
        """
        Perform gradient descent using the optimizer.
        """
        n, d = X.shape

        # Initial guess
        w = np.copy(w_init)
        f, g = self.loss_fn.evaluate(w, X, y)

        # Reset the optimizer state and tie it to the new parameters.
        # See optimizers.py for why reset() is useful here.
        self.optimizer.reset()
        self.optimizer.set_fun_obj(self.loss_fn)
        self.optimizer.set_parameters(w)
        self.optimizer.set_fun_obj_args(X, y)

        # Collect training information for debugging
        fs = [f]
        gs = [g]
        ws = []

        # Use gradient descent to optimize w
        while True:
            f, g, w, break_yes = self.optimizer.step()
            fs.append(f)
            gs.append(g)
            ws.append(w)
            if break_yes:
                break

        return w, fs, gs, ws

    def fit(self, X, y):
        """
        Generic fitting subroutine in triplet:
        1. Make initial guess
        2. Check correctness of function object
        3. Use gradient descent to optimize
        """
        n, d = X.shape

        # Correctness check
        if self.check_correctness:
            w = np.random.rand(d)
            self.loss_fn.check_correctness(w, X, y)

        # Initial guess
        w = np.zeros(d)

        # Optimize
        self.w, self.fs, self.gs, self.ws = self.optimize(w, X, y)

    def predict(self, X):
        """
        By default, implement linear regression prediction
        """
        return X @ self.w


class LeastSquaresBias:
    "Least Squares with a bias added"

    def fit(self, X, y):
        """YOUR CODE HERE FOR Q3.1"""
        bias_vector = np.ones(shape=(X.shape[0],1))
        X = np.hstack((bias_vector,X))
        self.w = solve(X.T @ X, X.T @ y)
        #raise NotImplementedError()

    def predict(self, X_pred):
        """YOUR CODE HERE FOR Q3.1"""
        # predict the Z 
        # use Z and v to predict y
        bias = np.ones((X_pred.shape[0] , 1))
        X_pred = np.hstack((bias, X_pred))
        return X_pred @ self.w
        #raise NotImplementedError()
       

class LeastSquaresPoly:
    "Least Squares with polynomial basis"

    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self, X, y):
        """YOUR CODE HERE FOR Q3.2"""
        X = self._poly_basis (X)
        self.w = solve(X.T @ X, X.T @ y)
       

    def predict(self, X_pred):
        """YOUR CODE HERE FOR Q3.2"""
        X = self._poly_basis (X_pred)
        return X @ self.w
        # raise NotImplementedError()

    # A private helper function to transform any X with d=1 into
    # the polynomial basis defined by this class at initialization.
    # Returns the matrix Z that is the polynomial basis of X.
    def _poly_basis(self, X):
        
        """YOUR CODE HERE FOR Q3.2"""
        poly_X = np.ones ((X. shape [0],1))
        for i in range(1, self.p+1):
            poly_X = np.hstack( (poly_X, X**i))
        return poly_X
        # raise NotImplementedError()
