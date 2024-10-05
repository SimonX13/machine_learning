#!/usr/bin/env python
import os
from pathlib import Path

import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from fun_obj import (
    LogisticRegressionLoss,
    LogisticRegressionLossL0,
    LogisticRegressionLossL2,
    SoftmaxLoss,
)
import linear_models
from optimizers import (
    GradientDescentLineSearch,
    GradientDescentLineSearchProxL1,
)
from utils import load_dataset, classification_error, handle, run, main


@handle("2")
def q2():
    print("2.0")
    data = load_dataset("logisticData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    fun_obj = LogisticRegressionLoss()
    optimizer = GradientDescentLineSearch(max_evals=400, verbose=False)
    model = linear_models.LinearClassifier(fun_obj, optimizer)
    model.fit(X, y)

    train_err = classification_error(model.predict(X), y)
    print(f"Linear Training error: {train_err:.3f}")

    val_err = classification_error(model.predict(X_valid), y_valid)
    print(f"Linear Validation error: {val_err:.3f}")

    print(f"# nonZeros: {np.sum(model.w != 0)}")
    print(f"# function evals: {optimizer.num_evals}")


@handle("2.1")
def q2_1():
    data = load_dataset("logisticData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    fun_obj = LogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch(max_evals=400, verbose=False)
    model = linear_models.LinearClassifier(fun_obj, optimizer)
    model.fit(X, y)

    train_err = classification_error(model.predict(X), y)
    print(f"Linear Training error: {train_err:.3f}")

    val_err = classification_error(model.predict(X_valid), y_valid)
    print(f"Linear Validation error: {val_err:.3f}")

    print(f"# nonZeros: {np.sum(model.w != 0)}")
    print(f"# function evals: {optimizer.num_evals}")


@handle("2.2")
def q2_2():
    data = load_dataset("logisticData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    """YOUR CODE HERE FOR Q2.2"""
    for lambda_value in [0.01, 0.1, 1, 10]:
        print(f"lambda = {lambda_value}")
        optimizer = GradientDescentLineSearchProxL1(max_evals=400, verbose=False, lammy=lambda_value)
        model = linear_models.LinearClassifier(LogisticRegressionLoss(), optimizer)
        model.fit(X, y)
        train_error = classification_error(model.predict(X), y)
        print(f"Training error: {train_error:.3f}")
        val_error = classification_error(model.predict(X_valid), y_valid)
        print(f"Validation error: {val_error:.3f}")
        print(f"# feature used: {np.sum(model.w != 0)}")
        print(f"# gradient descent iteration: {optimizer.num_evals}")
        print("\n")
    


@handle("2.3")
def q2_3():
    data = load_dataset("logisticData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    local_loss = LogisticRegressionLoss()
    global_loss = LogisticRegressionLossL0(1)
    optimizer = GradientDescentLineSearch(max_evals=400, verbose=False)
    model = linear_models.LinearClassifierForwardSel(local_loss, global_loss, optimizer)
    model.fit(X, y)

    train_err = classification_error(model.predict(X), y)
    print(f"Linear training 0-1 error: {train_err:.3f}")

    val_err = classification_error(model.predict(X_valid), y_valid)
    print(f"Linear validation 0-1 error: {val_err:.3f}")

    print(f"# nonZeros: {np.sum(model.w != 0)}")
    print(f"total function evaluations: {model.total_evals:,}")


@handle("3")
def q3():
    data = load_dataset("multiData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    model = linear_models.LeastSquaresClassifier()
    model.fit(X, y)

    train_err = classification_error(model.predict(X), y)
    print(f"LeastSquaresClassifier training 0-1 error: {train_err:.3f}")

    val_err = classification_error(model.predict(X_valid), y_valid)
    print(f"LeastSquaresClassifier validation 0-1 error: {val_err:.3f}")

    print(f"model predicted classes: {np.unique(model.predict(X))}")


@handle("3.2")
def q3_2():
    data = load_dataset("multiData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    fun_obj = LogisticRegressionLoss()
    optimizer = GradientDescentLineSearch(max_evals=500, verbose=False)
    model = linear_models.LinearClassifierOneVsAll(fun_obj, optimizer)
    model.fit(X, y)

    train_err = classification_error(model.predict(X), y)
    print(f"LinearClassifierOneVsAll training 0-1 error: {train_err:.3f}")

    val_err = classification_error(model.predict(X_valid), y_valid)
    print(f"LinearClassifierOneVsAll validation 0-1 error: {val_err:.3f}")

    print(f"model predicted classes: {np.unique(model.predict(X))}")


@handle("3.4")
def q3_4():
    data = load_dataset("multiData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    fun_obj = SoftmaxLoss()
    optimizer = GradientDescentLineSearch(max_evals=1_000, verbose=True)
    model = linear_models.MulticlassLinearClassifier(fun_obj, optimizer)
    model.fit(X, y)

    train_err = classification_error(model.predict(X), y)
    print(f"SoftmaxLoss training 0-1 error: {train_err:.3f}")

    val_err = classification_error(model.predict(X_valid), y_valid)
    print(f"SoftmaxLoss validation 0-1 error: {val_err:.3f}")

    print(f"model predicted classes: {np.unique(model.predict(X))}")


@handle("3.5")
def q3_5():
    from sklearn.linear_model import LogisticRegression

    data = load_dataset("multiData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    """YOUR CODE HERE FOR Q3.5"""
    

    # Train scikit-learn models
    sk_one_vs_all = LogisticRegression(multi_class='ovr', penalty='none', fit_intercept=False)
    sk_one_vs_all.fit(X, y)
    sk_softmax = LogisticRegression(multi_class='multinomial', penalty='none', fit_intercept=False)
    sk_softmax.fit(X, y)
    
    def calculate_error(model, X, y):
        predictions = model.predict(X)
        return np.mean(predictions != y)

    # Calculate and print errors
    results = {
        "Scikit-learn One-vs-All": {
            "Training Error": calculate_error(sk_one_vs_all, X, y),
            "Validation Error": calculate_error(sk_one_vs_all, X_valid, y_valid)
        },
        "Scikit-learn Softmax": {
            "Training Error": calculate_error(sk_softmax, X, y),
            "Validation Error": calculate_error(sk_softmax, X_valid, y_valid)
        }
    }
    print(results)

if __name__ == "__main__":
    main()
