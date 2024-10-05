#!/usr/bin/env python
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"] # correct result 

    """YOUR CODE HERE FOR Q1. Also modify knn.py to implement KNN predict."""
    temp = [1,3,10]
    for k in temp:
        
        model = KNN(k)
        model.fit(X,y)
        result = model.predict(X_test)
        print(result)
    
        error = np.mean(result != y_test)
        print("KNN test error: %.3f" % error)
    if k==1:  
        plot_classifier(model, X, y)
        fname = Path("..", "figs", "q1_2_knnplot.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
    #raise NotImplementedError()



@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = list(range(1, 30, 4))
    """YOUR CODE HERE FOR Q2"""
    cv_accs = []
    for i in ks:
        model = KNN(i)
        mean_accucracy = []
        for j in range(10):
            total_rows = X.shape[0]
            start_percentage = 10*j  # Start percentage
            end_percentage = 10+10*j    # End percentage
            start_index = int(total_rows * start_percentage / 100)
            end_index = int(total_rows * end_percentage / 100)
            # Create a mask to select rows within the specified percentage range
            mask = np.logical_and(np.arange(total_rows) >= start_index, np.arange(total_rows) <= end_index) #mark the 10% to be 1 so that only 10% part is left
            filter_train_X = X[~mask]
            filter_train_y = y[~mask]
            
            filter_test_X = X[mask]
            filter_test_y = y[mask]
            model.fit(filter_train_X, filter_train_y)
            
            result = model.predict(filter_test_X)
            error = np.mean(result != filter_test_y)
            mean_accucracy.append(error)
        cv_accs.append(np.mean(mean_accucracy)) # for this k value, it generates this error value
        
    avg_list = []
    for i in ks:
        model = KNN(i)
        model.fit(X,y)
        result = model.predict(X_test)
        error = np.mean(result != y_test)
        avg_list.append(error)
        
    accucracy_list= []
    accucracy_list_validation = []
    
    for i in avg_list:
        accucracy_list.append(1-i)
    for i in cv_accs :
        accucracy_list_validation.append(1-i)
    plt.plot(ks, accucracy_list, label="test error")
    plt.plot(ks, accucracy_list_validation, label = "cross-validation")
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Accruacy")
    plt.title("cross-validation and test accuracies as a function of k")
    plt.grid(axis='y')
    fname = Path("..", "figs", "q2_2_plot.pdf")
    plt.savefig(fname)
    print (f"Figure saved as {fname}")
    return cv_accs
    #raise NotImplementedError()



@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    """YOUR CODE HERE FOR Q3.2"""
    print(wordlist[72])
    row_42 = X[802]  
    indices_of_ones = np.where(row_42 == 1)[0] # Find the indices where the values are equal to 1
    print(wordlist[indices_of_ones])
    print(groupnames[y[802]])
    #raise NotImplementedError()



@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    """CODE FOR Q3.4: Modify naive_bayes.py/NaiveBayesLaplace"""

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayesLaplace(num_classes=4, beta=1)
    model.fit(X, y)

    """YOUR CODE HERE FOR Q3.4. Also modify naive_bayes.py/NaiveBayesLaplace"""
    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Laplace Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Laplace Naive Bayes validation error: {err_valid:.3f}")
   # raise NotImplementedError()



@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf,stump_class=DecisionStumpInfoGain))
    print("Random tree info gain")
    evaluate_model(RandomTree(max_depth=np.inf))
    print("Random forest info gain")
    evaluate_model(RandomForest(50, max_depth=np.inf))
    # print("Decision tree info gain")
    # evaluate_model(DecisioTnTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

    """YOUR CODE FOR Q4. Also modify random_tree.py/RandomForest"""
    #raise NotImplementedError()



@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.1. Also modify kmeans.py/Kmeans"""
    raise NotImplementedError()



@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.2"""
    raise NotImplementedError()



if __name__ == "__main__":
    main()
