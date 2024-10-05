import numpy as np


def example(x):
    return np.sum(x ** 2)


def example_grad(x):
    return 2 * x


def foo(x):
    result = 1
    λ = 4  # this is here to make sure you're using Python 3
    # ...but in general, it's probably better practice to stick to plaintext
    # names. (Can you distinguish each of λ𝛌𝜆𝝀𝝺𝞴 at a glance?)
    for x_i in x:
        result += x_i ** λ
    return result

def foo_grad(x):
    # Your implementation here...
    # compute the gradient of the function x
    result = []
    λ = 4  # this is here to make sure you're using Python 3
    # ...but in general, it's probably better practice to stick to plaintext
    # names. (Can you distinguish each of λ𝛌𝜆𝝀𝝺𝞴 at a glance?)
    for x_i in x:
        result.append(λ* x_i ** (λ-1))
    return result
    #raise NotImplementedError()

def bar(x):
    return np.prod(x)

def bar_grad(x):
    # Your implementation here...
    # Hint: This is a bit tricky - what if one of the x[i] is zero?
    # result = []
    # temp = np.prod(x)
    # for x_i in x:
    #     result.append(temp / x_i)
    # return result
    bar_x = np.prod(x)
    gradient = bar_x / x
    return gradient
    # raise NotImplementedError()


