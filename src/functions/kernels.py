import numpy as np
import pandas as pd


def gaussian_kernel(X, Y):
    """
    Known as Radial Basis Function, 'rbf' kernel in sklearn.svm.
    :param X:
    :param Y:
    :return: kernel function
    """
    return np.exp(-abs(abs(X-Y))**2)
