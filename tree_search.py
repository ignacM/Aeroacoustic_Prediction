import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics, linear_model, ensemble, model_selection
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from decision_tree import print_regression_solutions, print_regression_residuals, plot_regression_outcome


def runTree(X, Y):
    """
    Add so that the model can be chosen, so that it doesn't need to be tree itself.
    :param X:
    :param Y:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    """Baseline Tree"""
    parameters = {'max_depth': np.arange(1, 19),
                  'max_features': ('sqrt', 'log2'),
                  'min_samples_split': np.arange(1, 10)
                  }
    tree = DecisionTreeRegressor(random_state=42)
    """ Training Error and Test Error is visualized for different parameters:"""
    def mae_scores(arguments):
        regressor = DecisionTreeRegressor(**arguments)
        scores = -1 * cross_val_score(regressor, x_train, y_train, cv=3,
                                      scoring='neg_mean_absolute_error')
        regressor.fit(x_train, y_train)
        ypred_train = regressor.predict(x_train)
        ypred_test = regressor.predict(x_test)

        train_error = mae(y_train, ypred_train)
        test_error = mae(y_test, ypred_test)

        return np.mean(scores), train_error, test_error

    def parameter_vs_mae(start, end, step, parameter_name):
        results = {}
        train_errors = {}
        test_errors = {}
        for i in range(start, end, step):
            results[i], train_errors[i], test_errors[i] = mae_scores({parameter_name: i})
        """ Plotting Parameter vs Mean Absolute Error"""
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(list(results.keys()), list(results.values()))
        plt.title("%s vs MAE" % parameter_name, fontweight="bold", fontsize=25)
        plt.xlabel("%s" % parameter_name)
        plt.ylabel("Mean Absolute Error")
        plt.grid(True)
        plt.show()

        """ Plotting Train vs Test Error"""
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(list(train_errors.keys()), list(train_errors.values()))
        plt.plot(list(test_errors.keys()), list(test_errors.values()))
        plt.title("Train vs Test Error for %s" % parameter_name, fontweight="bold", fontsize=25)
        plt.xlabel("%s" % parameter_name)
        plt.ylabel("Error")
        plt.legend(["Training Error", "Test Error"])
        plt.grid(True)
        plt.show()
        return

    parameter_vs_mae(1, 20, 1, 'max_depth')
    parameter_vs_mae(2, 100, 1, 'min_samples_split')

    """Then, a grid search of the optimization of minimum samples split is done,
    we know that lower minimum_samples split results in lower test error: """
    parameter_grid = {"min_samples_split": np.arange(2, 10, 1)}
    min_samples_split_gsearch = GridSearchCV(estimator=tree, param_grid=parameter_grid, cv=5)
    min_samples_split_gsearch.fit(x_train, y_train)
    min_samples_split_gsearch.best_params_
    """Fit tree and present results:"""
    best_min_samples_split_tree = min_samples_split_gsearch.best_estimator_
    print_regression_solutions(x_train, y_train, x_test, y_test,
                               best_min_samples_split_tree, 'Best Min Sample Split')
    """ Then repeat with another parameter"""
    parameter_grid = {"max_depth": np.arange(1, 10, 1)}
    tree = DecisionTreeRegressor(random_state=42)
    depth_gsearch = GridSearchCV(estimator=tree, param_grid=parameter_grid, cv=5)
    depth_gsearch.fit(x_train, y_train)
    depth_gsearch.best_params_
    best_depth_tree = depth_gsearch.best_estimator_
    """fit tree and print evaluation"""
    print_regression_solutions(x_train, y_train, x_test, y_test,
                               best_depth_tree, 'Best Max Depth')
    """ Then do gird search over all parameters giving a list of significant parametrs"""
    """param_grid = {'max_depth': np.arange(1, 19),
                  'max_features': ('sqrt', 'log2'),
                  'min_samples_split': np.arange(2, 10)
                  }
    tree = DecisionTreeRegressor(random_state=42)
    model = model_selection.GridSearchCV(estimator=tree, param_grid=param_grid, verbose=10, n_jobs=1, cv=5)
    model.fit(x_train, y_train)

    # Re-build the model with best estimated tree
    best_model = model.best_estimator_
    tree = best_model
    tree.fit(x_train, y_train)"""
    """Print the regression metrics"""
    return

""" Should be returning best max depth and best minimum sample split"""


