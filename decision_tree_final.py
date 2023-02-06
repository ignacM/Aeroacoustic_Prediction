import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics, linear_model, ensemble, model_selection, tree

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from graphviz import Source
from decision_tree import plot_regression_outcome, print_regression_solutions, \
    print_regression_residuals, print_actual_vs_real, print_tree


def runTree(X, Y):
    """
    Add so that it can take arguments as a model, (max_depth, min_sample_split) parameters of the model
    :param X:
    :param Y:
    :return:
    """
    # Splitting test and train data into test_size %
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    # Give parameters of optimized tree:
    tre = DecisionTreeRegressor(max_depth=6, min_samples_split=2, ccp_alpha=0.01)
    #tre.fit(x_train, y_train)
    # Print evaluation metrics
    print_regression_solutions(x_train, y_train, x_test, y_test, tre, 'Optimized Decision Tree')
    ypred = tre.predict(x_test)

    # Plot results
    plot_regression_outcome(y_test, ypred, 'Optimized Decision Tree')
    print_regression_residuals(y_test, ypred, 'Optimized Decision Tree')
    print_actual_vs_real(y_test, ypred)

    # Plot tree
    print_tree(tre)
