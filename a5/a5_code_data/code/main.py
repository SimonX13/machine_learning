#!/usr/bin/env python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from encoders import PCAEncoder
from kernels import GaussianRBFKernel, LinearKernel, PolynomialKernel
from linear_models import (
    LinearModel,
    LinearClassifier,
    KernelClassifier,
)
from optimizers import (
    GradientDescent,
    GradientDescentLineSearch,
    StochasticGradient,
)
from fun_obj import (
    LeastSquaresLoss,
    LogisticRegressionLossL2,
    KernelLogisticRegressionLossL2,
)
from learning_rate_getters import (
    ConstantLR,
    InverseLR,
    InverseSqrtLR,
    InverseSquaredLR,
)
from utils import (
    load_dataset,
    load_trainval,
    load_and_split,
    plot_classifier,
    savefig,
    standardize_cols,
    handle,
    run,
    main,
)


@handle("1")
def q1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # Standard (regularized) logistic regression
    loss_fn = LogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    lr_model = LinearClassifier(loss_fn, optimizer)
    lr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(lr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(lr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(lr_model, X_train, y_train)
    savefig("logRegPlain.png", fig)

    # kernel logistic regression with a linear kernel
    loss_fn = KernelLogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    kernel = LinearKernel()
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegLinear.png", fig)


@handle("1.1")
def q1_1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    """YOUR CODE HERE FOR Q1.1"""
    pass


@handle("1.2")
def q1_2():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    sigmas = 10.0 ** np.array([-2, -1, 0, 1, 2])
    lammys = 10.0 ** np.array([-4, -3, -2, -1, 0, 1, 2])

    # train_errs[i, j] should be the train error for sigmas[i], lammys[j]
    train_errs = np.full((len(sigmas), len(lammys)), 100.0)
    val_errs = np.full((len(sigmas), len(lammys)), 100.0)  # same for val

    """YOUR CODE HERE FOR Q1.2"""
    pass

    # Make a picture with the two error arrays. No need to worry about details here.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))
    for (name, errs), ax in zip([("training", train_errs), ("val", val_errs)], axes):
        cax = ax.matshow(errs, norm=norm)

        ax.set_title(f"{name} errors")
        ax.set_ylabel(r"$\sigma$")
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([str(sigma) for sigma in sigmas])
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(range(len(lammys)))
        ax.set_xticklabels([str(lammy) for lammy in lammys])
        ax.xaxis.set_ticks_position("bottom")
    fig.colorbar(cax)
    savefig("logRegRBF_grids.png", fig)


@handle("3.2")
def q3_2():
    data = load_dataset("animals.pkl")
    X_train = data["X"]
    animal_names = data["animals"]
    trait_names = data["traits"]

    # Standardize features
    X_train_standardized, mu, sigma = standardize_cols(X_train)
    n, d = X_train_standardized.shape

    # Matrix plot
    fig, ax = plt.subplots()
    ax.imshow(X_train_standardized)
    savefig("animals_matrix.png", fig)
    plt.close(fig)

    # 2D visualization
    np.random.seed(3164)  # make sure you keep this seed
    j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
    random_is = np.random.choice(n, 15, replace=False)  # choose random examples

    fig, ax = plt.subplots()
    ax.scatter(X_train_standardized[:, j1], X_train_standardized[:, j2])
    for i in random_is:
        xy = X_train_standardized[i, [j1, j2]]
        ax.annotate(animal_names[i], xy=xy)
    savefig("animals_random.png", fig)
    plt.close(fig)

    """YOUR CODE HERE FOR Q3"""
    # pass
    # pca = PCAEncoder(k=2)
    # pca.fit(X_train)
    # X_train_pca = pca.encode(X_train)

    # fig, ax = plt.subplots()
    # ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
    # for i in range(animal_names.shape[0]):
    #     xy = X_train_pca[i]
    #     ax.annotate(animal_names[i], xy=xy)
    # savefig("animals_pca.png", fig)
    # plt.close(fig)
     
    # largest_influence_pc1 = np.argmax(np.abs(pca.W[0]))
    # largest_influence_pc2 = np.argmax(np.abs(pca.W[1]))
    # print("Largest influence on first principal component:", trait_names[largest_influence_pc1])
    # print("Largest influence on second principal component:", trait_names[largest_influence_pc2])
    
    # X_c = X_train - pca.mu
    
    # variance = 1 - np.sum((X_train_pca @ pca.W - X_c) ** 2) / np.sum(X_c ** 2)
    # print("Variance explained by 2-dimensional representation:", variance)

    # for k in range(1, X_train_standardized.shape[1] + 1):
    #     pca = PCAEncoder(k)
    #     pca.fit(X_train)
    #     X_train_pca = pca.encode(X_train)
    #     X_c = X_train - pca.mu
    #     variance = 1 - np.sum((X_train_pca @ pca.W - X_c) ** 2) / np.sum(X_c ** 2)
    #     if variance >= 0.5:
    #         break
        
    # print(f"Number of principal components needed to explain 50% of the variance: {k}")
    # Initialize PCA encoder with 2 components
    encoder_pca = PCAEncoder(k=2)
    encoder_pca.fit(X_train)
    X_train_transformed = encoder_pca.encode(X_train)

    # Plotting the transformed data
    fig, ax = plt.subplots()
    ax.scatter(X_train_transformed[:, 0], X_train_transformed[:, 1])
    for idx in range(animal_names.shape[0]):
        position = X_train_transformed[idx]
        ax.annotate(animal_names[idx], xy=position)
    savefig("animals_pca_plot.png", fig)
    plt.close(fig)

    # Identifying features with the most influence on each principal component
    most_influential_feature_pc1 = np.argmax(np.abs(encoder_pca.W[0]))
    most_influential_feature_pc2 = np.argmax(np.abs(encoder_pca.W[1]))
    print(f"Largest influence on PC1: {trait_names[most_influential_feature_pc1]}")
    print(f"Largest influence on PC2: {trait_names[most_influential_feature_pc2]}")

    # Calculating and printing variance explained by the PCA
    X_centered = X_train - encoder_pca.mu
    variance_explained = 1 - np.sum((X_train_transformed @ encoder_pca.W - X_centered) ** 2) / np.sum(X_centered ** 2)
    print(f"Variance explained by 2D representation: {variance_explained}")

    # Determining the minimum number of components to explain at least 50% variance
    for num_components in range(1, X_train_standardized.shape[1] + 1):
        encoder_pca = PCAEncoder(num_components)
        encoder_pca.fit(X_train)
        X_train_transformed = encoder_pca.encode(X_train)
        X_centered = X_train - encoder_pca.mu
        variance_explained = 1 - np.sum((X_train_transformed @ encoder_pca.W - X_centered) ** 2) / np.sum(X_centered ** 2)
        if variance_explained >= 0.5:
            break

    print(f"Minimum principal components for 50% variance: {num_components}")

    



@handle("4")
def q4():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    # Train ordinary regularized least squares
    loss_fn = LeastSquaresLoss()
    optimizer = GradientDescentLineSearch()
    model = LinearModel(loss_fn, optimizer, check_correctness=False)
    model.fit(X_train, y_train)
    print(model.fs)  # ~700 seems to be the global minimum.

    print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
    print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")

    # Plot the learning curve!
    fig, ax = plt.subplots()
    ax.plot(model.fs, marker="o")
    ax.set_xlabel("Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    savefig("gd_line_search_curve.png", fig)


@handle("4.1")
def q4_1():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    """YOUR CODE HERE FOR Q4.1"""
    # pass
    loss_fn = LeastSquaresLoss()
    base_optimizer = GradientDescent()
    batchSize = [1, 10, 100]
    for i in batchSize:
        sgd_optimizer = StochasticGradient(
        base_optimizer=base_optimizer,
        learning_rate_getter=ConstantLR(0.0003),
        batch_size= i
        )
        model = LinearModel(loss_fn, sgd_optimizer, check_correctness=False)
       
        model.fit(X_train, y_train)
        print(model.fs)  # ~700 seems to be the global minimum.

        print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
        print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")

    # Plot the learning curve!
    fig, ax = plt.subplots()
    ax.plot(model.fs, marker="o")
    ax.set_xlabel("Sto Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    savefig("Batch Size of SGD.png", fig)


@handle("4.3")
def q4_3():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    """YOUR CODE HERE FOR Q4.3"""
    loss_fn = LeastSquaresLoss()
    learning_rate_getters = [
        ConstantLR(0.1),
        InverseLR(0.1),
        InverseSquaredLR(0.1),
        InverseSqrtLR(0.1)
    ]
    colors = ['blue', 'green', 'red','orange']
    labels = ['ConstantLR', 'InverseLR', 'InverseSquaredLR','InverseSquareRootLR']

    # plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()
    for lr_getter, color, label in zip(learning_rate_getters, colors, labels):
        sgd_optimizer = StochasticGradient(
            base_optimizer=GradientDescent(),
            learning_rate_getter=lr_getter,
            batch_size=10
        )
        model = LinearModel(loss_fn, sgd_optimizer, check_correctness=False)
        model.fit(X_train, y_train)

        # Assuming model.fs stores the loss values
        ax.plot(model.fs, color=color, label=label)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Learning Curves for Different SGD Optimizers')
    ax.legend()
    # ax.show()
    savefig("Batch Size of SGD.png", fig)


if __name__ == "__main__":
    main()
